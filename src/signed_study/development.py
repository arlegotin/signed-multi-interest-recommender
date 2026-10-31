"""Complete-cohort development selection and bounded one-time tuning evidence."""
from dataclasses import asdict, replace
import math
import uuid

import numpy as np
from threadpoolctl import threadpool_limits

from .budget import BudgetExceeded, atomic_write_json
from .calibration import _fit_cost, _score_units, planned_comparison_schedule, select_calibration_users, require_full_cohort
from .metrics import history_statistics, macro_average, planned_half_width, select_family, select_comparator
from .models import endpoint_requests, initial_configurations, model_configuration
from .protocol import canonical_sha256
from .stage_evidence import plain, config_record
from .workload import measure_users


ALIASES = {'Centroid':('SignedMax',lambda p:p=={'K':1,'lambda':0}),
           'PositiveMax':('SignedMax',lambda p:p.get('lambda')==0),
           'RocchioDirection':('MeanNegative',lambda p:p.get('K')==1)}


def aggregate_configurations(entries, attempts, users):
    """An incomplete configuration has no selectable macro AUC."""
    expected=set(users)
    rows=[]
    for entry in entries:
        runs=[a for a in attempts if a['configuration_id']==entry.id and a.get('kind','fit_and_score')=='fit_and_score']
        complete=(len(runs)==len(users) and {a['user_id'] for a in runs}==expected
                  and all(a['status']=='completed' for a in runs))
        valid={a['user_id']:a for a in runs if a['status']=='completed'}
        rows.append(dict(configuration=entry,status='completed' if complete else 'failed',complete=complete,
                         failed_share=1-len(valid.keys() & expected)/len(users),
                         macro_auc=macro_average([a['metrics'] for a in runs])['auc'] if complete else None,
                         user_prototype_identities={a['user_id']:a['prototype_identity'] for a in runs if 'prototype_identity' in a},
                         user_auc={u:a['metrics']['auc'] for u,a in valid.items()}))
    return rows


def choose_families(rows):
    choices={}
    families=sorted({r['configuration'].family for r in rows})
    missing=[]
    for family in families:
        candidates=[r for r in rows if r['configuration'].family==family]
        try:
            chosen=select_family(candidates)
        except ValueError:
            missing.append(family)
            continue
        choices[family]=next(r for r in candidates if r['configuration']==chosen)
    for name,(source,predicate) in ALIASES.items():
        candidates=[r for r in rows if r['configuration'].family==source and predicate(r['configuration'].parameters)]
        try:
            selected=select_family(candidates)
        except ValueError:
            missing.append(name)
            continue
        source_row=next(r for r in candidates if r['configuration']==selected)
        choices[name]={**source_row,'configuration':model_configuration(name,selected.parameters),
                       'source_configuration_id':selected.id,'fit_family':source}
    return choices,sorted(missing)


def projected_cost(episodes, entries, measurements, config, *, exact_parameters=False):
    """Empirical upper-envelope pricing, using the actual canonical chunk.

    Endpoint and selected solver costs require their actual measured parameters.
    """
    total=0.
    for entry in entries:
        rows=[r for r in measurements if r['family']==entry.family]
        if exact_parameters and entry.family in ('LinearLogistic','RBFSVM'):
            rows=[r for r in rows if r.get('parameters')==entry.parameters]
        chunk=config.numerics['score_chunk_size']
        priced=[]
        for row in rows:
            timing=row if row.get('chunk_size')==chunk else row.get('score_measurements',{}).get(str(chunk))
            if timing is None or (timing is not row and not timing.get('numerical_accepted')):
                raise ValueError('Missing measured canonical scoring chunk cost')
            value={**row,**{k:timing.get(k,0.) for k in ('score_seconds','write_seconds','overhead_seconds')}}
            if any(not math.isfinite(value[k]) or value[k]<0 for k in ('fit_seconds','score_seconds','write_seconds','overhead_seconds')):
                raise ValueError('Invalid measured development cost')
            priced.append(value)
        if not priced:
            raise ValueError('No measured cost for required family')
        for e in episodes:
            h=e.history
            units=_score_units(entry.family,len(h.positive_ids),len(h.negative_ids),len(e.candidate_ids),entry.parameters.get('K',4))
            rate=max(r['score_seconds']/_score_units(entry.family,r['positive_count'],r['negative_count'],r['candidate_count'],r.get('K',4)) for r in priced)
            total+=_fit_cost(priced,len(h.positive_ids)+len(h.negative_ids))+units*rate+max(r['write_seconds']+r['overhead_seconds'] for r in priced)
    return total


def _selection_record(row, config):
    entry=row['configuration']
    family=row.get('fit_family',entry.family)
    constructor=('learned_classifier' if family in ('LinearLogistic','RBFSVM') else
                 'all_positive_exemplars' if family in ('AllExemplarMargin','ExemplarRatio') else 'spherical_lloyd')
    return {**asdict(entry),'macro_auc':row['macro_auc'],
            'source_configuration_id':row.get('source_configuration_id',entry.id),
            'fit_family':row.get('fit_family',entry.family),
            'development_prototype_identities':row.get('user_prototype_identities',{}),
            'prototype_identity':canonical_sha256(dict(algorithm=constructor,policy=plain(config.prototype),
                                                    K=entry.parameters.get('K'))) }


def _boundary_coverage(choices,rows,config):
    coverage=[]
    grids={'C':list(config.selection['expanded_C'][:1])+list(config.C_grid)+list(config.selection['expanded_C'][1:]),
           'g':list(config.selection['expanded_g'][:1])+list(config.g_grid)+list(config.selection['expanded_g'][1:]),
           'lambda':list(config.lambda_grid)+[config.selection['expanded_lambda']]}
    for family,row in choices.items():
        if family in ALIASES: continue
        p=row['configuration'].parameters
        for key,grid in grids.items():
            if key not in p or (key=='lambda' and p[key]==0): continue
            initial=getattr(config,key+'_grid')
            if p[key] not in (initial[0],initial[-1],grid[0],grid[-1]): continue
            index=grid.index(p[key])
            adjacent=grid[index+1 if index==0 or p[key]==initial[0] else index-1]
            match=next((r for r in rows if r['configuration'].family==family and r['complete']
                        and r['configuration'].parameters=={**p,key:adjacent}),None)
            coverage.append(dict(family=family,parameter=key,value=p[key],adjacent=adjacent,
                gain=None if match is None else row['macro_auc']-match['macro_auc'],
                persistent_outer=p[key] in (grid[0],grid[-1])))
    return coverage


def refresh_selected_forecast(result,store,timings,config,ledger,calibration):
    """Price the concrete thirteen slots, including timing and geometric checks."""
    schedule=result['comparison_schedule']
    entries=[model_configuration(s['fit_family'],s['parameters']) for s in schedule]
    confirmation={e.history.user_id:e for e in store.confirmation}
    planned=[confirmation[u] for u in result['confirmation_user_ids']]
    development={e.history.user_id:e for e in store.development}
    timing_users=calibration['calibration_user_ids']
    catalog=calibration['forecast'].get('catalog_items',calibration.get('catalog_items'))
    if type(catalog) is not int or catalog<=0:
        raise ValueError('Selected-method timing requires measured catalog cardinality')
    timing_episodes=[replace(development[u],candidate_ids=tuple(range(catalog))) for u in timing_users]
    numerical=planned[:64]
    repeats=config.timing['warmups']+config.timing['repetitions']
    geometric=[entry for entry,slot in zip(entries,schedule) if slot['geometric']]
    seconds={'G4':projected_cost(planned,entries,timings,config,exact_parameters=True),
             'G5':repeats*projected_cost(timing_episodes,entries,timings,config,exact_parameters=True)
                  +projected_cost(numerical,geometric,timings,config,exact_parameters=True)+2}
    evaluations={'G4':len(planned)*len(entries),
                 'G5':len(timing_users)*len(entries)*repeats+len(numerical)*len(geometric)}
    state=ledger.snapshot(); caps=config.resources; margin=caps['safety_multiplier']
    stage_limits=state.get('stage_limits')
    if stage_limits is None:
        if state.get('reallocations'):
            raise ValueError('Transfer history requires effective stage limits')
        stage_limits=caps['stage_seconds']  # Legacy synthetic ledger fixtures.
    from .budget import time_forecast_fits
    feasible=(state['attempted_evaluations']+sum(evaluations.values())<=caps['evaluation_limit']
              and time_forecast_fits(state,config,seconds))
    # The frozen schedule cannot exceed the preselected slot allowances.
    allowance=calibration['forecast']['planning']
    feasible=feasible and len(entries)<=allowance['confirmation_entries'] and (
        evaluations['G5']<=allowance['timing_evaluations']+allowance['numerical_diagnostic_evaluations'])
    from .diagnostics import APPLICABILITY, CONTAMINATION_METHODS
    contamination_users=planned[:config.contamination['max_users']]
    contamination_schedule=[s for s in schedule if s['name'] in CONTAMINATION_METHODS]
    contamination=dict(user_ids=[e.history.user_id for e in contamination_users],
        seeds=list(config.contamination_seeds),schedule=contamination_schedule,
        seconds=len(config.contamination_seeds)*projected_cost(contamination_users,
            [model_configuration(s['fit_family'],s['parameters']) for s in contamination_schedule],timings,config,exact_parameters=True),
        evaluations=len(contamination_users)*len(config.contamination_seeds)*len(contamination_schedule),
        applicability=APPLICABILITY,minimum_subgroup_users=100,numerical_auc_threshold=config.delta,
        planned_sample_counts=dict(contamination_users=len(contamination_users),numerical_users=len(numerical),timing_users=len(timing_users)),
        gain_predicate='lower CI > 0; diagnostic gain only, not delta practical success',
        minimum_extra_seconds=config.contamination['minimum_extra_seconds'],
        policy='conditional only; first hash-ordered users; no retuning; effective ledger execution policy')
    return dict(status='feasible' if feasible else 'infeasible',seconds=seconds,evaluations=evaluations,contamination=contamination,
                execution_policy=state.get('execution_policy',{'mode':'bounded'}),
                comparison_entries=len(entries),comparison_schedule=schedule,timing_user_ids=list(timing_users),
                numerical_user_ids=[e.history.user_id for e in numerical],catalog_items=catalog,
                cost_chunk_size=config.numerics['score_chunk_size'],safety_multiplier=margin,
                stop_reason=None if feasible else 'Concrete selected schedule exceeds remaining budget or admitted allowances')


def endpoint_campaign_forecast(cohort, entries, requests, timings, store, calibration, config, ledger, expansion_seconds):
    """Outcome-free upper bound for every possible selected/matched slot.

    Each slot uses the largest cost across all eligible initial/endpoint
    parameters. The maximizers may differ by stage; this remains conservative
    without learning an endpoint winner from outcomes.
    """
    candidates=list(entries)+list(requests)
    for entry in list(candidates):
        if entry.family=='SignedMax':
            candidates.extend([model_configuration('MeanNegative',entry.parameters),
                               model_configuration('SignedMax',{'K':1,'lambda':entry.parameters['lambda']})])
    candidates=list({e.id:e for e in candidates}.values())
    schedule=planned_comparison_schedule()
    confirmation={e.history.user_id:e for e in store.confirmation}
    planned=[confirmation[u] for u in calibration['forecast']['planned_user_ids']['confirmation']]
    development={e.history.user_id:e for e in store.development}
    timing_users=calibration['calibration_user_ids']
    catalog=calibration['forecast'].get('catalog_items',calibration.get('catalog_items'))
    if type(catalog) is not int or catalog<=0:
        raise ValueError('Endpoint campaign requires measured catalog cardinality')
    timing_episodes=[replace(development[u],candidate_ids=tuple(range(catalog))) for u in timing_users]
    def bound(episodes,geometric=False):
        maxima={family:max(projected_cost(episodes,[entry],timings,config,exact_parameters=True)
                           for entry in candidates if entry.family==family)
                for family in {slot['fit_family'] for slot in schedule if not geometric or slot['geometric']}}
        return sum(maxima[slot['fit_family']] for slot in schedule if not geometric or slot['geometric'])
    repeats=config.timing['warmups']+config.timing['repetitions']
    numerical=planned[:64]
    seconds={'G3':expansion_seconds,'G4':bound(planned),
             'G5':repeats*bound(timing_episodes)+bound(numerical,True)+2}
    evaluations={'G3':len(cohort)*len(requests),'G4':len(planned)*len(schedule),
                 'G5':len(timing_users)*len(schedule)*repeats+len(numerical)*sum(s['geometric'] for s in schedule)}
    state=ledger.snapshot();caps=config.resources;margin=caps['safety_multiplier']
    limits=state.get('stage_limits',caps['stage_seconds'])
    allowance=calibration['forecast']['planning']
    from .budget import time_forecast_fits
    feasible=(state['attempted_evaluations']+sum(evaluations.values())<=caps['evaluation_limit']
        and time_forecast_fits(state,config,seconds)
        and len(schedule)<=allowance['confirmation_entries']
        and evaluations['G5']<=allowance['timing_evaluations']+allowance['numerical_diagnostic_evaluations'])
    return dict(status='feasible' if feasible else 'infeasible',seconds=seconds,evaluations=evaluations,
                execution_policy=state.get('execution_policy',{'mode':'bounded'}),
                safety_multiplier=margin,comparison_entries=len(schedule),
                basis='maximum measured cost per possible selected/matched slot; no endpoint outcomes',
                stop_reason=None if feasible else 'Complete remaining endpoint campaign exceeds remaining capacity')


def execute_development(config,ledger,store,cache,calibration,*,attempt_ids=None):
    """Execute admitted fixed cohorts; callers verify provenance and smoke first."""
    require_full_cohort(config,store,calibration['forecast']['planned_user_ids'],ledger.snapshot())
    users=calibration['forecast']['planned_user_ids']['development']
    episodes={e.history.user_id:e for e in store.development}
    if not users or len(set(users))!=len(users) or not set(users)<=episodes.keys():
        raise ValueError('Development requires fixed calibrated development IDs')
    cohort=[episodes[u] for u in users]
    entries=initial_configurations(config)
    attempts=[]; batches=[]; timings=[]; pilot_attempts=[]
    attempt_ids=[] if attempt_ids is None else attempt_ids
    batch_active=False
    prefix=f'development-{uuid.uuid4().hex}'
    def batch(declared,seconds,kind,*,batch_users=None,timing_only=False):
        nonlocal batch_active
        selected_users=users if batch_users is None else batch_users
        path=config.paths['results']/f'{prefix}-{kind}.json'
        with ledger.work('G3',seconds,len(selected_users)*len(declared)) as aid:
            batch_active=True
            attempt_ids.append(aid)
            with threadpool_limits(limits=config.timing['blas_threads']):
                result=measure_users(store,cache,selected_users,config,ledger,numerical_checks=False,
                                     output_path=path,declared_entries=declared,timing_only=timing_only)
            (pilot_attempts if timing_only else attempts).extend(result['attempts'])
            timings.extend(result['user_measurements'])
            batches.append(dict(path=str(path),journal_path=str(path.with_suffix('.jsonl')),kind=kind,
                                attempt_id=aid,actual_evaluations=result['actual_evaluations']))
        batch_active=False
        return result
    with ledger.work('G3',0,0) as aid:
        attempt_ids.append(aid)
        initial_seconds=projected_cost(cohort,entries,calibration['user_measurements'],config)
    batch(entries,initial_seconds,'initial')
    with ledger.work('G3',0,0) as aid:
        attempt_ids.append(aid)
        rows=aggregate_configurations(entries,attempts,users)
        choices,missing=choose_families(rows)
        requests=endpoint_requests(choices,config) if not missing else []
        initial_boundary_winners=_boundary_coverage(choices,rows,config)
    coverage=dict(requested=[asdict(e) for e in requests],admitted=False,unavailable=[],expansion_batches=0)
    if requests:
        # Official fixed cohorts have >=62 users; tiny synthetic fixtures use
        # their entire cohort rather than manufacture additional pilot users.
        pilot_users=select_calibration_users(cohort) if len(cohort)>=32 else list(users)
        try:
            with ledger.work('G3',0,0) as aid:
                attempt_ids.append(aid)
                pilot_seconds=projected_cost([episodes[u] for u in pilot_users],requests,timings,config)
            pilot= batch(requests,pilot_seconds,'endpoint-profile',batch_users=pilot_users,timing_only=True)
            pilot_measurements=pilot['user_measurements']
            if len(pilot_measurements)!=len(pilot_users)*len(requests):
                raise ValueError('Incomplete endpoint timing pilot; full expansion unavailable')
            with ledger.work('G3',0,0) as aid:
                attempt_ids.append(aid)
                seconds=projected_cost(cohort,requests,pilot_measurements,config,exact_parameters=True)
                coverage['campaign_forecast']=endpoint_campaign_forecast(
                    cohort,entries,requests,[*calibration['user_measurements'],*timings],
                    store,calibration,config,ledger,seconds)
                if coverage['campaign_forecast']['status']!='feasible':
                    raise ValueError(coverage['campaign_forecast']['stop_reason'])
            batch(requests,seconds,'endpoints')
            entries+=requests
            coverage.update(admitted=True,expansion_batches=1)
        except (BudgetExceeded,ValueError) as error:
            # A failed admission has no started endpoint batch. Runtime failures
            # remain durable and propagate rather than masquerade as unavailable.
            if batch_active: raise
            coverage.update(unavailable=[e.id for e in requests],reason=str(error))
    with ledger.work('G3',0,0) as aid:
        attempt_ids.append(aid)
        with threadpool_limits(limits=config.timing['blas_threads']):
            return _finalize(config,ledger,store,cache,calibration,users,cohort,entries,attempts,
                             batches,attempt_ids,timings,pilot_attempts,coverage,prefix,initial_boundary_winners)


def _finalize(config,ledger,store,cache,calibration,users,cohort,entries,attempts,
              batches,attempt_ids,timings,pilot_attempts,coverage,prefix,initial_boundary_winners):
    rows=aggregate_configurations(entries,attempts,users)
    choices,missing=choose_families(rows)
    result=dict(schema_version=1,status='blocked' if missing else 'completed',development_user_ids=users,
                confirmation_user_ids=calibration['forecast']['planned_user_ids']['confirmation'],
                configurations=[asdict(e) for e in entries],attempts=attempts,batches=batches,
                attempt_ids=attempt_ids,actual_evaluations=len(attempts)+len(pilot_attempts),pilot_attempts=pilot_attempts,
                missing_required_families=missing,
                incomplete_configuration_ids=[r['configuration'].id for r in rows if not r['complete']],
                initial_boundary_winners=initial_boundary_winners,
                stop_reason='Required family has no complete eligible configuration' if missing else None,
                repair_decisions=[dict(family=f,status='requires_bounded_repair',scope='full_family_fixed_cohort') for f in missing],
                configuration_results=[{**r,'configuration':asdict(r['configuration'])} for r in rows],
                families={f:_selection_record(r,config) for f,r in choices.items()},endpoint_coverage=coverage,
                confirmation_outcomes_accessed=False)
    if not missing:
        signed=choices['SignedMax']; baseline=select_comparator(choices)
        result.update(S_star=_selection_record(signed,config), B_star=_selection_record(choices[baseline.family],config))
        p=signed['configuration'].parameters
        controls={'matched_MeanNegative':('MeanNegative',dict(p)),
                  'matched_PositiveMax':('SignedMax',{'K':p['K'],'lambda':0}),
                  'matched_K1_SignedMax':('SignedMax',{'K':1,'lambda':p['lambda']})}
        result['matched_controls']={name:{**asdict(model_configuration(family,parameters)),
            'fit_family':family,'prototype_identity':canonical_sha256(dict(algorithm='spherical_lloyd',policy=plain(config.prototype),K=parameters['K']))}
            for name,(family,parameters) in controls.items()}
        K1=next(r for r in rows if r['configuration'].family=='SignedMax'
                and r['configuration'].parameters=={'K':1,'lambda':0})
        for name,control in result['matched_controls'].items():
            control['development_prototype_identities']=(K1 if name=='matched_K1_SignedMax' else signed)['user_prototype_identities']
        result['comparison_schedule']=[{**slot,**(result['matched_controls'][slot['name']] if slot['role']=='matched_control'
                                                   else result['families'][slot['name']])} for slot in planned_comparison_schedule()]
        differences=[signed['user_auc'][u]-choices[baseline.family]['user_auc'][u] for u in users]
        width=planned_half_width(differences,len(result['confirmation_user_ids']))
        result['planned_precision']=dict(delta=config.delta,half_width=width,modest_effects_may_be_inconclusive=width>config.delta,
                                         basis='paired development SD; optimistic planning estimate')
        histories=[]
        for e in cohort:
            histories.append(dict(user_id=e.history.user_id, **history_statistics(e.history,cache)))
        result['history_statistics']=histories
        result['history_thresholds']={key:float(np.median([r[key] for r in histories])) for key in ('H','O')}
        result['boundary_winners']=_boundary_coverage(choices,rows,config)
        result['complete_suite_eligible']=(not coverage['unavailable'] and not result['incomplete_configuration_ids']
                                           and not any(r['persistent_outer'] for r in result['boundary_winners']))
        result['selected_forecast']=refresh_selected_forecast(result,store,timings,config,ledger,calibration)
        if result['selected_forecast']['status']!='feasible':
            result.update(status='blocked',stop_reason=result['selected_forecast']['stop_reason'])
    result['artifact_name']=f'{prefix}.json'
    result['input_sha256']=canonical_sha256(dict(config=config_record(config),calibration=calibration['output_sha256']))
    result['output_sha256']=canonical_sha256(plain(result))
    atomic_write_json(config.paths['results']/result['artifact_name'],plain(result))
    return result
