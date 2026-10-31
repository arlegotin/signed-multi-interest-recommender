"""Outcome-independent calibration sampling and empirical resource forecasts."""
import hashlib
import math

from .data import Episode, EpisodeStore
from .models import initial_configurations
from .budget import results_first, time_forecast_fits


def planned_comparison_schedule():
    """All ten selected families and three controls before parameter selection.

    Aliases retain declared evaluation slots even when computation is shared.
    Later freeze/diagnostic consumers must resolve the actual parameters and
    refresh costs; this preselection allowance does not certify a frozen plan.
    """
    aliases={'Centroid':'SignedMax','PositiveMax':'SignedMax','RocchioDirection':'MeanNegative'}
    families=('Centroid','PositiveMax','RocchioDirection','MeanNegative','SignedMax',
              'AllExemplarMargin','PrototypeRatio','ExemplarRatio','LinearLogistic','RBFSVM')
    schedule=[{'name':name,'fit_family':aliases.get(name,name),'role':'selected'} for name in families]
    schedule.extend({'name':name,'fit_family':family,'role':'matched_control'} for name,family in (
        ('matched_MeanNegative','MeanNegative'),('matched_PositiveMax','SignedMax'),
        ('matched_K1_SignedMax','SignedMax')))
    return [{**slot,'geometric':slot['fit_family'] not in ('LinearLogistic','RBFSVM')} for slot in schedule]


def user_digest(user_id):
    return hashlib.sha256(f'signed-study:{user_id}'.encode()).hexdigest()


def require_full_cohort(config,cohort,planned,state):
    """Backstop policy against audited episode IDs, never forecast-claimed counts."""
    if not results_first(state):
        return
    dataset=cohort.audit['dataset']
    counts={'ml-1m':(418,1776),'ml-100k':(62,225)}
    if dataset not in counts:
        raise ValueError('Results-first full cohort requires the original official dataset')
    for role,count in zip(('development','confirmation'),counts[dataset]):
        episodes=getattr(cohort,role)
        expected=sorted([e.history.user_id for e in episodes],key=user_digest)
        supplied=planned.get(role)
        if (len(expected)!=count or len(set(expected))!=count or supplied!=expected
                or not isinstance(supplied,list) or any(type(uid) is not int for uid in supplied)
                or any(type(e.history.user_id) is not int or e.history.role!=role for e in episodes)):
            raise ValueError(f'Results-first requires the full audited digest-ordered {role} cohort')
    if {e.history.user_id for e in cohort.development} & {e.history.user_id for e in cohort.confirmation}:
        raise ValueError('Results-first full cohort has overlapping user partitions')
    try:
        _verify_cohort_binding(config,cohort,state)
    except (OSError,KeyError,TypeError) as error:
        raise ValueError('Results-first full cohort requires complete verified acquisition/audit evidence') from error


def _verify_cohort_binding(config,cohort,state):
    """Permit only the already acquired/audited dataset, retaining invocation.

    A dataset string and the expected counts are insufficient: verify the exact
    terminal acquisition/audit receipts, official access evidence and immutable
    episode records. This opens no outcome arrays.
    """
    import json
    from pathlib import Path
    from .data import episode_store_digest, load_episode_store
    from .stage_evidence import read_stage, file_reference
    from .stages import _official
    if config.dataset!='ml-1m':
        raise ValueError('Results-first full cohort dataset substitution is unauthorized')
    acquired=read_stage(config,'acquire')
    audited=read_stage(config,'audit',{'acquire':acquired['output_sha256']})
    for record,allowed in ((acquired,('G0','setup')),(audited,('G0',))):
        if not record['attempt_ids'] or any(state['attempts'].get(a,{}).get('status')!='completed'
                or state['attempts'].get(a,{}).get('stage') not in allowed for a in record['attempt_ids']):
            raise ValueError('Full cohort requires completed acquisition/audit accounting')
    manifest=json.loads(Path(acquired['artifacts']['data_manifest.json']['path']).read_text())
    _official(manifest,config)
    if (manifest['dataset']!=cohort.audit['dataset']
            or acquired['inputs']['config']['dataset']!=config.dataset):
        raise ValueError('Full cohort is not bound to the original acquisition')
    if manifest['dataset']=='ml-100k' and acquired['artifacts'].get('access_evidence')!=file_reference(config.paths['results']/'data_acquisition.json'):
        raise ValueError('Fallback access evidence is not bound to the original acquisition')
    bound=load_episode_store(audited['artifacts']['audit']['path'],audited['artifacts']['episodes']['path'])
    if (bound.audit!=cohort.audit or episode_store_digest(bound)!=episode_store_digest(cohort)
            or file_reference(cohort.paths['audit'])!=audited['artifacts']['audit']
            or file_reference(cohort.paths['episodes'])!=audited['artifacts']['episodes']):
        raise ValueError('Full cohort differs from the verified audited episode store')


def select_calibration_users(episodes: list[Episode]) -> list[int]:
    if len(episodes) < 32:
        raise ValueError('Calibration requires at least 32 development users')
    if any(e.history.role != 'development' for e in episodes):
        raise ValueError('Calibration accepts development episodes only')
    if len({e.history.user_id for e in episodes}) != len(episodes):
        raise ValueError('Duplicate development user')
    ranked = sorted(episodes, key=lambda e: (
        len(e.history.positive_ids) + len(e.history.negative_ids), user_digest(e.history.user_id)))
    chosen = []
    for q in range(4):
        part = ranked[q*len(ranked)//4:(q+1)*len(ranked)//4]
        small = sorted(part, key=lambda e: user_digest(e.history.user_id))[:4]
        remaining = [e for e in part if e not in small]
        chosen.extend(small + sorted(remaining, key=lambda e: (
            -len(e.candidate_ids), user_digest(e.history.user_id)))[:4])
    return sorted((e.history.user_id for e in chosen), key=user_digest)


def _fit_cost(rows, support):
    # Upper envelope of observed size bins; extrapolate only the measured slope,
    # never a claimed SVM quadratic/cubic complexity law.
    bins = {}
    for r in rows:
        n = r['support_count']
        bins[n] = max(bins.get(n, 0.), r['fit_seconds'])
    sizes = sorted(bins)
    upper = [bins[n] for n in sizes if n >= support]
    if upper:
        return max(upper)
    slopes = [max(0., (bins[b]-bins[a])/(b-a)) for a, b in zip(sizes, sizes[1:])]
    slope = max(slopes, default=bins[sizes[-1]] / sizes[-1])
    return max(bins.values()) + slope*(support-sizes[-1])


def _score_units(family, positive, negative, candidates, K):
    if family == 'LinearLogistic':
        references = 1
    elif family in ('AllExemplarMargin', 'ExemplarRatio', 'RBFSVM'):
        references = positive + negative
    else:
        references = min(K, positive) + (1 if family == 'MeanNegative' else negative)
    return 384 * candidates * references


def forecast_scale(measurements: dict, cohort: EpisodeStore, ledger) -> dict:
    """Forecast remaining G3/G4/G5 from actual capped support/candidate sizes.

    Reserve 92 development entries, 13 selected/matched confirmation entries,
    eight all-grid smoke users and 32x13x(1+5) timing evaluations. These are
    conservative planning reservations, not claims of executed fits.
    """
    config, state = ledger.config, ledger.snapshot()
    registry = initial_configurations(config)
    schedule=planned_comparison_schedule()
    comparison_entries=len(schedule)
    timing_evaluations=config.timing['users']*comparison_entries*(config.timing['warmups']+config.timing['repetitions'])
    numerical_evaluations=64*sum(slot['geometric'] for slot in schedule)
    families = {entry.family for entry in registry}
    rows = measurements['user_measurements']
    by_family = {family: [r for r in rows if r['family'] == family] for family in families}
    if any(not r for r in by_family.values()):
        raise ValueError('Every comparator family needs measured calibration work')
    for r in rows:
        if any(not math.isfinite(r[key]) or r[key] < 0 for key in
               ('fit_seconds', 'score_seconds', 'write_seconds')):
            raise ValueError('Invalid calibration duration')
        if min(r['support_count'], r['candidate_count'], r['positive_count'], r['negative_count']) < 1:
            raise ValueError('Invalid calibration sizes')

    def price_chunk(chunk):
        priced={family:[] for family in families}
        for row in rows:
            if chunk==row.get('chunk_size',config.numerics['score_chunk_size']):
                timing=row
            else:
                timing=row.get('score_measurements',{}).get(str(chunk))
                if timing is None or timing.get('numerical_accepted') is not True:
                    return None
            if any(not math.isfinite(timing.get(key,float('nan'))) or timing[key]<0
                   for key in ('score_seconds','write_seconds')):
                return None
            overhead=timing.get('overhead_seconds',0.)
            if not math.isfinite(overhead) or overhead<0:
                return None
            priced[row['family']].append({**row,'score_seconds':timing['score_seconds'],
                                          'write_seconds':timing['write_seconds'],'overhead_seconds':overhead})
        return priced

    by_family=price_chunk(config.numerics['score_chunk_size'])
    if by_family is None:
        raise ValueError('Canonical scoring chunk requires measured costs')

    from .costs import FamilyCost
    prepared={family:FamilyCost.prepare(observed,family) for family,observed in by_family.items()}

    def cost(e, entry, candidate_count=None):
        return prepared[entry.family].cost(e,entry.family,entry.parameters.get('K',4),candidate_count)

    def comparison_cost(e,candidate_count=None,*,geometric_only=False):
        maxima={family:max(cost(e,c,candidate_count) for c in registry if c.family==family) for family in families}
        return sum(maxima[slot['fit_family']] for slot in schedule if not geometric_only or slot['geometric'])

    dev = sorted(cohort.development, key=lambda e: user_digest(e.history.user_id))
    confirm = sorted(cohort.confirmation, key=lambda e: user_digest(e.history.user_id))
    scales = ([list(config.calibration['fallback_scale_pair'])] if cohort.audit['dataset'] == 'ml-100k'
              else [list(p) for p in config.calibration['primary_scale_pairs']])
    if results_first(state):
        scales = scales[:1]  # The authorized full cohort cannot shrink for elapsed time.
    if cohort.audit['dataset'] not in ('ml-1m', 'ml-100k'):
        raise ValueError('Production forecast requires official dataset provenance')
    caps = config.resources
    stage_limits = state.get('stage_limits')
    if stage_limits is None:
        if state.get('reallocations'):
            raise ValueError('Transfer history requires effective stage limits')
        stage_limits = caps['stage_seconds']  # Legacy synthetic ledger fixtures.
    margin = caps['safety_multiplier']
    result = dict(status='infeasible', scale=None, safety_multiplier=margin, tested_scales=[],
                  execution_policy=state.get('execution_policy', {'mode':'bounded'}),
                  cost_chunk_size=config.numerics['score_chunk_size'],
                  adjustments=[], policy=dict(embedding_batch_size=config.encoder['batch_size'],
                  score_chunk_size=config.numerics['score_chunk_size'], blas_threads=config.timing['blas_threads'],
                  inference_workers=1), planning=dict(development_entries=92, confirmation_entries=comparison_entries,
                  comparison_schedule=schedule,smoke_users=8, smoke_entries=78, timing_evaluations=timing_evaluations,
                  numerical_diagnostic_evaluations=numerical_evaluations),
                  support_model='upper observed size-bin envelope plus measured nonnegative slope',
                  stop_reason='Smallest permitted scale cannot fit remaining allocations and reserve')
    fixed = measurements.get('fixed_seconds', 0.)
    for value in (fixed, measurements.get('optional_seconds', 0.)):
        if type(value) not in (int,float) or not math.isfinite(value) or value < 0:
            raise ValueError('Invalid calibration duration')
    catalog_items=measurements.get('catalog_items',3883 if cohort.audit['dataset']=='ml-1m' else 1682)
    if type(catalog_items) is not int or catalog_items<=0:
        raise ValueError('Measured catalog size must be positive')
    timing_users=set(select_calibration_users(list(dev)))
    def reprice():
        development=[sum(cost(e,c) for c in registry)+14*max(cost(e,c) for c in registry) for e in dev]
        confirmation=[comparison_cost(e) for e in confirm]
        timing=(config.timing['warmups']+config.timing['repetitions'])*sum(
            comparison_cost(e,catalog_items) for e in dev if e.history.user_id in timing_users)
        numerical=sum(comparison_cost(e,geometric_only=True) for e in confirm[:64])
        return development,confirmation,timing,numerical,sum(development[:8])
    dev_cost,conf_cost,timing,numerical,smoke_seconds=reprice()

    def projection(pair, optional):
        d, c = pair
        seconds = {'G3': sum(dev_cost[:d]) + smoke_seconds + fixed,
                   'G4': sum(conf_cost[:c]) + fixed, 'G5': timing + numerical + fixed + optional}
        evaluations = d*92 + c*comparison_entries + 624 + timing_evaluations + numerical_evaluations + (4200 if optional else 0)
        fits = (d <= len(dev) and c <= len(confirm)
                and state['attempted_evaluations']+evaluations <= caps['evaluation_limit']
                and time_forecast_fits(state,config,seconds))
        return fits, seconds, evaluations

    optional = measurements.get('optional_seconds', 0.)
    first_fits, _, _ = projection(scales[0], optional)
    if measurements.get('memory_bound') or not first_fits:
        optional = 0.
        result['adjustments'].append('remove_optional_work')
        if measurements.get('memory_bound'):
            result['policy']['embedding_batch_size'] = max(1, config.encoder['batch_size']//2)
            result['adjustments'].append('reduce_embedding_batch')
            # A smaller batch is a proposed policy, never invented evidence of
            # lower RSS. An actual remeasurement is needed before proceeding.
            if not measurements.get('reduced_batch_memory_verified', False):
                result['stop_reason'] = 'Reduced embedding batch needs measured memory verification'
        else:
            result['adjustments'].append('embedding_batch_unchanged_not_memory_bound')
        result['policy']['score_chunk_size'] = min(7, config.numerics['score_chunk_size'])
        result['adjustments'].append('reduce_scoring_chunk')
        if not measurements.get('chunk_invariance'):
            result['stop_reason'] = 'Reduced scoring chunk lacks invariance evidence'
            return result
        proposed=price_chunk(result['policy']['score_chunk_size'])
        if proposed is None:
            result['stop_reason']='Proposed scoring chunk needs complete valid timing remeasurement'
            return result
        by_family=proposed
        prepared={family:FamilyCost.prepare(observed,family) for family,observed in by_family.items()}
        dev_cost,conf_cost,timing,numerical,smoke_seconds=reprice()
        result['cost_chunk_size']=result['policy']['score_chunk_size']
    for pair in scales:
        result['tested_scales'].append(pair)
        fits, seconds, evaluations = projection(pair, optional)
        if measurements.get('memory_bound') and not measurements.get('reduced_batch_memory_verified'):
            fits = False
        if fits:
            d,c = pair
            result.update(status='feasible', scale=pair, seconds=seconds,
                          smoke_seconds=smoke_seconds,catalog_items=catalog_items,
                          reserved_evaluations=evaluations, optional_work=bool(optional), stop_reason=None,
                          planned_user_ids={'development': [e.history.user_id for e in dev[:d]],
                                            'confirmation': [e.history.user_id for e in confirm[:c]]},
                          interval_width_estimate={'half_width': 1.96/math.sqrt(c),
                             'basis': 'conservative SD bound 1 for paired AUC difference; development estimate follows G3'})
            break
    return result
