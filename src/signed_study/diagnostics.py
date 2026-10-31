"""Fixed secondary mechanism, precision, feedback and catalog-cost diagnostics.

No selection or refitting for numerical checks. Optional feedback work is priced
before freeze and admitted separately, after the mandatory work, without reserve.
"""
from dataclasses import replace
import json
import os
from pathlib import Path
import time

import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits

from . import confirmation, data, protocol_lock
from .budget import BudgetExceeded
from .metrics import assign_stratum, evaluate_user, history_statistics, paired_analysis
from .models import fit_user, score_user
from .protocol import ProtocolMismatch, canonical_sha256
from .stage_evidence import plain, verify_record, file_reference


CONTAMINATION_METHODS = ('SignedMax', 'MeanNegative', 'LinearLogistic', 'RBFSVM')
APPLICABILITY = 'gain_primary_or_full_cohort_mean_controls_or_precise_high_high_or_material_float32_auc_change'


def contaminate_history(history, seed: int):
    """Swap exactly q distinct signs, retaining support IDs and class counts."""
    p, n = list(history.positive_ids), list(history.negative_ids)
    if not p or not n or len(set(p+n)) != len(p+n):
        raise ValueError('contamination requires disjoint nonempty support classes')
    q = max(1, int(np.floor(.1 * min(len(p), len(n)))))
    rng = np.random.default_rng(seed)
    pi, ni = rng.choice(len(p), q, replace=False), rng.choice(len(n), q, replace=False)
    for i, j in zip(pi, ni):
        p[i], n[j] = n[j], p[i]
    return replace(history, positive_ids=tuple(p), negative_ids=tuple(n))


def float32_geometry_scores(model, candidates, *, chunk_size=512):
    """Cast fixed X/P/N once; NumPy float32 throughout, without normalization.

    Direct rowwise float32 distances avoid scipy cdist's float64 backend and a
    candidate-by-support-by-dimension tensor. No fit or canonical scorer call.
    """
    X, P, N = (np.asarray(a, dtype=np.float32) for a in (candidates, model.state['P'], model.state['N']))
    scores = np.empty(len(X), dtype=np.float32)
    penalty = np.float32(model.parameters.get('lambda', 0))
    for start in range(0, len(X), chunk_size):
        x = X[start:start+chunk_size]
        if model.method in ('PrototypeRatio', 'ExemplarRatio'):
            def nearest(rows):
                best = np.full(len(x), np.inf, dtype=np.float32)
                for row in rows:
                    diff = x-row
                    best = np.minimum(best, np.sqrt(np.sum(diff*diff, axis=1, dtype=np.float32)))
                return best
            dp, dn = nearest(P), nearest(N)
            denom = dp+dn
            values = np.full(len(x), np.float32(.5), dtype=np.float32)
            np.divide(dn, denom, out=values, where=denom != 0)
        else:
            values = np.max(x@P.T, axis=1)
            if len(N) and penalty != 0:
                negative = x@N.mean(axis=0, dtype=np.float32) if model.method in ('MeanNegative','RocchioDirection') else np.max(x@N.T, axis=1)
                values -= penalty*negative
        scores[start:start+len(x)] = values
    if not np.isfinite(scores).all():
        raise ValueError('nonfinite float32 diagnostic scores')
    return scores


def mechanism_strata(users, episodes, cache, selected, saved, config):
    stats = {u:history_statistics(episodes[u].history, cache) for u in users}
    auc = {s['name']:{} for s in selected['comparison_schedule']}
    for row in saved['attempts']:
        if row['status']=='completed': auc[row['slot']][row['user_id']]=row['metrics']['auc']
    def summarize(ids):
        result = dict(n_users=len(ids),user_ids=ids,imprecise=len(ids)<100,
                      status='available' if ids else 'unavailable',interpretation='secondary_unadjusted')
        result['comparisons'] = {name:paired_analysis(ids,{u:v for u,v in auc['SignedMax'].items() if u in ids},
            {u:v for u,v in values.items() if u in ids},delta=config.delta,seed=config.bootstrap_seed,
            resamples=config.bootstrap_resamples,batch_size=config.bootstrap_batch_size)
            for name,values in auc.items() if name!='SignedMax'} if ids else {}
        return result
    names = [f'{h}_H_{o}_O' for h in ('low','high') for o in ('low','high')]
    strata = {'H_O':{name:summarize([u for u in users if assign_stratum(stats[u]['H'],stats[u]['O'],selected['history_thresholds'])==name]) for name in names},
              'negative_count':{name:summarize([u for u in users if lo<=stats[u]['negative_count']<=hi])
                                for name,lo,hi in [('5–8',5,8),('9–16',9,16),('17–64',17,64)]}}
    norms=[stats[u]['negative_mean_norm'] for u in users]
    pairs=[(stats[u]['negative_mean_norm'],auc['SignedMax'][u]-auc['matched_MeanNegative'][u])
           for u in users if u in auc['SignedMax'] and u in auc['matched_MeanNegative']]
    correlation = None
    if len(pairs)>1:
        x,y=np.asarray(pairs).T
        if np.std(x)>0 and np.std(y)>0: correlation=float(np.corrcoef(x,y)[0,1])
    strata['negative_mean_norm']=dict(values=norms,user_ids=users,quantiles=np.quantile(norms,[0,.25,.5,.75,1]).tolist(),
        paired_users=len(pairs),signed_minus_matched_mean_correlation=correlation,
        interpretation='descriptive_scale_confound; no subgroup selection')
    associations={}
    for name in ('MeanNegative','matched_MeanNegative'):
        paired=[(stats[u]['negative_mean_norm'],auc['SignedMax'][u]-auc[name][u]) for u in users if u in auc['SignedMax'] and u in auc[name]]
        corr=None
        if len(paired)>1:
            x,y=np.asarray(paired).T
            if np.std(x)>0 and np.std(y)>0: corr=float(np.corrcoef(x,y)[0,1])
        associations[name]=dict(n_users=len(paired),pearson_correlation=corr,paired_values=[list(p) for p in paired])
    strata['negative_mean_norm']['associations']=associations
    strata['history_statistics']=[dict(user_id=u,**stats[u]) for u in users]
    strata['thresholds']=selected['history_thresholds']
    return strata


def _rank(scores, ids):
    return np.asarray(ids)[np.lexsort((np.asarray(ids),-scores))].tolist()


def numerical_measurements(users, schedule, rows, cache, lock_hash, config, ledger, record, labels_by_user):
    by_key={(r['user_id'],r['slot']):r for r in rows}
    results=[]
    for uid in users:
        for slot in schedule:
            if not slot['geometric']: continue
            ledger.checkpoint()
            row=dict(kind='numerical',user_id=uid,slot=slot['name'],configuration_id=slot['id'],status='started',
                     dtype='float32',backend='numpy.float32; direct rowwise float32 Euclidean distances',refitted=False)
            record(row)
            try:
                original=by_key.get((uid,slot['name']))
                if not original or original['status']!='completed' or 'fitted_state' not in original:
                    row.update(status='unavailable',reason='Original successful frozen geometry unavailable')
                else:
                    model=confirmation.load_frozen_geometry(original['fitted_state'],cache,lock_hash)
                    scores=float32_geometry_scores(model,cache.rows(original['candidate_ids']),chunk_size=config.numerics['score_chunk_size'])
                    canonical=np.asarray(original['scores'],dtype=np.float64)
                    row.update(status='completed',max_absolute_score_difference=float(np.max(np.abs(scores.astype(np.float64)-canonical))),
                        rank_positions_changed=sum(a!=b for a,b in zip(_rank(scores,original['candidate_ids']),_rank(canonical,original['candidate_ids']))),
                        canonical_dtype='float64',accuracy_claim=False,
                        absolute_auc_change=abs(evaluate_user(scores,labels_by_user[uid],original['candidate_ids'])['auc']-original['metrics']['auc']))
            except (ValueError,KeyError,OSError) as error:
                row.update(status='failed',reason=f'{type(error).__name__}: {error}')
            record(row); results.append(row)
    return results


def timing_measurements(episodes, users, schedule, cache, config, ledger, record):
    """Raw support -> fit -> resident full-catalog score -> deterministic rank."""
    ids=[int(i) for i in cache.item_ids]
    # Materialize once outside score boundaries, keeping candidate residence equal.
    resident=cache.rows(ids)
    index={i:j for j,i in enumerate(ids)}
    class Resident:
        def rows(self, requested):
            return resident[[index[int(i)] for i in requested]]
    rows=[]
    for uid in users:
        for repeat in range(config.timing['warmups']+config.timing['repetitions']):
            offset=repeat % len(schedule)
            for slot in schedule[offset:]+schedule[:offset]:
                ledger.checkpoint()
                row=dict(kind='timing',user_id=uid,slot=slot['name'],configuration_id=slot['id'],parameters=slot['parameters'],
                         repeat=repeat,warmup=repeat<config.timing['warmups'],candidate_ids=ids,status='started')
                start=time.perf_counter_ns(); record(row)
                try:
                    fit_start=time.perf_counter_ns()
                    model=fit_user(slot['fit_family'],episodes[uid].history,cache,slot['parameters'],config=config)
                    fit_end=time.perf_counter_ns()
                    if model.diagnostics['status']!='ok': raise ValueError('Failed timing fit')
                    scores=score_user(model,ids,Resident(),chunk_size=config.numerics['score_chunk_size'])
                    score_end=time.perf_counter_ns()
                    _rank(scores,ids)
                    rank_end=time.perf_counter_ns()
                    row.update(status='completed',fit_seconds=(fit_end-fit_start)/1e9,score_seconds=(score_end-fit_end)/1e9,
                               rank_seconds=(rank_end-fit_end)/1e9,diagnostics=plain(model.diagnostics))
                except Exception as error:
                    if isinstance(error,BudgetExceeded): raise
                    row.update(status='failed',reason=f'{type(error).__name__}: {error}')
                row['pipeline_before_final_write_seconds']=(time.perf_counter_ns()-start)/1e9
                write_start=time.perf_counter_ns(); record(row)
                write_seconds=(time.perf_counter_ns()-write_start)/1e9
                row=dict(row,final_write_seconds=write_seconds,total_evaluation_seconds=(time.perf_counter_ns()-start)/1e9)
                if row['status']=='completed':
                    row['pipeline_overhead_seconds']=max(0.,row['total_evaluation_seconds']-row['fit_seconds']-row['rank_seconds']-write_seconds)
                    k=len(model.state.get('P',[])); n=len(episodes[uid].history.negative_ids); d=resident.shape[1]; I=len(ids)
                    terms=(k+1 if slot['fit_family']=='MeanNegative' else k+n)
                    row['standalone_formula_operations']=dict(catalog_items=I,dimension=d,positive_representatives=k,negative_support=n,
                        dot_multiply_adds=0 if 'Ratio' in slot['fit_family'] else I*d*(1 if slot['fit_family']=='LinearLogistic' else len(model.state.get('support_vectors',[])) if slot['fit_family']=='RBFSVM' else terms),
                        distance_coordinate_differences=I*d*(k+n) if 'Ratio' in slot['fit_family'] else 0)
                rows.append(row)
    return rows


def timing_summary(rows, calibration, config):
    result={}
    for name in dict.fromkeys(r['slot'] for r in rows):
        complete=[r for r in rows if r['slot']==name and not r['warmup'] and r['status']=='completed']
        summary=dict(n_completed=len(complete),n_failed=sum(r['slot']==name and r['status']!='completed' for r in rows))
        for key in ('fit_seconds','score_seconds','rank_seconds','total_evaluation_seconds','final_write_seconds','pipeline_overhead_seconds'):
            values=[r[key] for r in complete]
            per_user=[np.median([r[key] for r in complete if r['user_id']==u]) for u in dict.fromkeys(r['user_id'] for r in complete)]
            summary[key]=dict(median=float(np.median(per_user)) if per_user else None,p95=float(np.percentile(per_user,95)) if per_user else None,
                repeat_sd=float(np.std(values)) if values else None,
                within_user_repeat_sd={str(u):float(np.std([r[key] for r in complete if r['user_id']==u])) for u in dict.fromkeys(r['user_id'] for r in complete)})
        fit,rank=summary['fit_seconds']['median'],summary['rank_seconds']['median']
        summary['amortization_1000_requests']=dict(per_user_fit_seconds=fit,request_rank_seconds=rank,
            fit_each_request_seconds=1000*(fit+rank) if fit is not None else None,
            one_fit_reused_seconds=fit+1000*rank if fit is not None else None)
        result[name]=summary
    cold=calibration.get('timings',{})
    cold_total=sum(cold.get(k,0.) for k in ('cold_model_load_seconds','catalog_encode_seconds','cache_load_verify_seconds'))
    return dict(methods=result,cold_shared_seconds=cold,cold_total_seconds=cold_total,
        shared_item_embedding_seconds=cold.get('catalog_encode_seconds'),threadpools=threadpool_info(),
        policy=dict(config.timing),rss_method='Ledger psutil process-tree sampling every 100 ms plus OS maximum RSS',
        interpretation='computational cost only; warm repeats are not accuracy seeds; fixed active catalog only',
        formula_operations='Standalone per request: SignedMax/AllExemplarMargin O(I*d*(K+h_neg)); MeanNegative O(I*d*(K+1)+h_neg*d); ratio O(I*d*(K+h_neg)); logistic O(I*d); RBF O(I*d*n_support). No cross-lambda score reuse in this benchmark.')


def contamination_applicability(saved, strata, numerical, plan, config):
    if plan['applicability']!=APPLICABILITY: raise ProtocolMismatch('Unknown frozen applicability rule')
    reasons=[]
    if saved['primary']['low']>0: reasons.append('primary lower CI > 0; diagnostic gain, not practical success')
    for name in ('MeanNegative','matched_MeanNegative'):
        if saved['comparisons'][name]['low']>0: reasons.append(f'full-cohort {name} lower CI > 0')
    high=strata['H_O']['high_H_high_O']
    if high['n_users']>=plan['minimum_subgroup_users'] and high['comparisons'].get('matched_MeanNegative',{}).get('low',-1)>0:
        reasons.append('high_H_high_O matched MeanNegative lower CI > 0 with at least 100 users')
    material={}
    for slot in dict.fromkeys(r['slot'] for r in numerical):
        values=[r['absolute_auc_change'] for r in numerical if r['slot']==slot and r['status']=='completed']
        material[slot]=dict(n_users=len(values),mean_absolute_auc_change=float(np.mean(values)) if values else None)
        if values and np.mean(values)>=plan['numerical_auc_threshold']:
            reasons.append(f'{slot} mean absolute float32 AUC change >= frozen delta')
    return dict(applicable=bool(reasons),reasons=reasons,numerical=material,high_high_users=high['n_users'],
                interpretation='diagnostic trigger only; canonical float64 validity and practical-success threshold unchanged')


def contamination_applicable(saved, strata, numerical, plan, config):
    return contamination_applicability(saved,strata,numerical,plan,config)['applicable']


def contamination_skip_reason(applicable, state, plan, config):
    if not applicable: return 'Diagnostic question not applicable under frozen rule'
    remaining=config.resources['experimental_seconds']-config.resources['report_reserve_seconds']-state['experimental_seconds']
    if remaining<config.contamination['minimum_extra_seconds']: return 'Fewer than 20 minutes remain beyond reporting reserve'
    margin=config.resources['safety_multiplier']
    if (remaining<margin*plan['seconds'] or state['stage_seconds']['G5']+margin*plan['seconds']>state['stage_limits']['G5']
        or state['attempted_evaluations']+plan['evaluations']>config.resources['evaluation_limit']):
        return 'Frozen conditional schedule does not fit remaining time/evaluations beyond reserve'
    return None


def contamination_measurements(plan, episodes, cache, config, permit, store, ledger, record):
    # Build once per (user,seed); every method consumes this exact cached history.
    swaps={(u,seed):contaminate_history(episodes[u].history,seed) for u in plan['user_ids'] for seed in plan['seeds']}
    rows=[]
    for uid in plan['user_ids']:
        labels=data.load_confirmation_outcomes(store,uid,permit)
        labels.setflags(write=False)
        for seed in plan['seeds']:
            history=swaps[(uid,seed)]
            for slot in plan['schedule']:
                ledger.checkpoint()
                row=dict(kind='contamination',user_id=uid,seed=seed,slot=slot['name'],configuration_id=slot['id'],parameters=slot['parameters'],
                         positive_ids=list(history.positive_ids),negative_ids=list(history.negative_ids),status='started')
                record(row)
                try:
                    model=fit_user(slot['fit_family'],history,cache,slot['parameters'],config=config)
                    row['diagnostics']=plain(model.diagnostics)
                    if model.diagnostics['status']!='ok': raise ValueError('Failed contamination fit')
                    scores=score_user(model,episodes[uid].candidate_ids,cache,chunk_size=config.numerics['score_chunk_size'])
                    row.update(status='completed',metrics=evaluate_user(scores,labels,episodes[uid].candidate_ids))
                except Exception as error:
                    if isinstance(error,BudgetExceeded): raise
                    row.update(status='failed',reason=f'{type(error).__name__}: {error}')
                record(row); rows.append(row)
    return dict(status='completed' if all(r['status']=='completed' for r in rows) else 'partial',attempts=rows,swap_assignments=[dict(user_id=u,seed=s,positive_ids=list(h.positive_ids),negative_ids=list(h.negative_ids)) for (u,s),h in swaps.items()],
                interpretation='secondary stress diagnostic; original hyperparameters and evaluation labels; no retuning')


def _diagnostic_inputs(lock_hash,payload,saved,events):
    """Stable scientific identity, independent of current qualification hashes."""
    finished=[e for e in events if e['kind']=='run_finished' and e['run_id']==saved['run_id']]
    if len(finished)!=1 or finished[0]['lock_sha256']!=lock_hash:
        raise ProtocolMismatch('Diagnostic confirmation identity lacks a unique frozen run artifact')
    reference=finished[0]['artifact']
    if Path(reference['path'])!=Path(saved['artifact_path']):
        raise ProtocolMismatch('Diagnostic confirmation artifact identity changed')
    return dict(lock_sha256=lock_hash,confirmation_run_id=saved['run_id'],confirmation_artifact=reference,
                confirmation_record_sha256=saved.get('original_output_sha256',saved['output_sha256']),
                frozen_development_sha256=canonical_sha256(payload['development']),
                config_sha256=canonical_sha256(payload['config']),
                cache_sha256=payload['identities']['cache_sha256'],
                episodes_sha256=payload['data_access']['episodes_sha256'])


def _load_saved_diagnostics(path,ledger,expected_inputs):
    """Require external publication binding and exact run inputs before qualification.

    A terminal resource failure may still have published immutable evidence; its
    noncompleted ledger status is qualified by the caller after identity checks.
    """
    try:
        result=json.loads(Path(path).read_text()); verify_record(result)
        actual=file_reference(path)
        accounting=ledger.snapshot()['attempts']
        anchors=[accounting[aid] for aid in result['attempt_ids'] if aid in accounting
                 and accounting[aid]['stage']=='G5' and accounting[aid]['event']=='finish'
                 and accounting[aid].get('outputs',{}).get('diagnostics')==actual]
        if len(anchors)!=1:
            raise ProtocolMismatch('Diagnostic artifact differs from its permanent ledger publication')
        if (Path(result['artifact_path'])!=Path(path)
                or result['lock_sha256']!=expected_inputs['lock_sha256']
                or result['confirmation_run_id']!=expected_inputs['confirmation_run_id']
                or result['confirmation_artifact']!=expected_inputs['confirmation_artifact']
                or result['inputs']!=expected_inputs
                or result['input_sha256']!=canonical_sha256(expected_inputs)):
            raise ProtocolMismatch('Diagnostic identity or frozen inputs belong to another confirmation run')
        protocol_lock.evidence.verify_reference(result['confirmation_artifact'])
        protocol_lock.evidence.verify_reference(result['attempt_journal'])
        return result
    except (ValueError,KeyError,OSError,TypeError) as error:
        raise ProtocolMismatch(f'Diagnostic provenance verification failed: {error}') from error


def _qualify(result,ledger,saved,verification_id=None):
    original=result
    result=dict(result)
    if verification_id is not None and verification_id not in result['attempt_ids']:
        result.update(attempt_ids=result['attempt_ids']+[verification_id],read_verification_attempt_id=verification_id)
    accounting=ledger.snapshot()['attempts']
    bad=[accounting[a]['status'] for a in result['attempt_ids'] if accounting[a]['status']!='completed']
    if bad or result['confirmation_status']!=saved['status'] or result['confirmation_interpretation']!=saved['interpretation']:
        result.update(confirmation_status=saved['status'],confirmation_interpretation=saved['interpretation'])
        if bad: result.update(status='partial',accounting_status=bad)
    if result!=original:
        result.setdefault('original_output_sha256',result.pop('output_sha256'))
        result['output_sha256']=canonical_sha256(result)
    return result


def diagnose(lock_path, ledger):
    """Only saved qualified confirmation; exclusive diagnostics never resume fits."""
    return confirmation._serialized(lock_path,_diagnose,ledger)


def _diagnose(lock_path,ledger):
    digest,payload=protocol_lock.read_lock(lock_path)
    # Heavy current-context/cache verification is separately charged, including
    # on saved diagnostic reads; it never reserves evaluations or resumes G4.
    with ledger.work('G5',1,0) as verification_id:
        live,resolved,store,cache,_,_=protocol_lock._context(ledger.config,ledger)
        events=confirmation.read_events(lock_path)
        authorization=next((e for e in reversed(events) if e['kind']=='correction_authorized'),None)
        permit=(confirmation._verify_correction(lock_path,live,authorization) if authorization else protocol_lock.verify_protocol(lock_path,live))
        saved=confirmation.load_saved_confirmation(lock_path,ledger,permit=permit)
        inputs=_diagnostic_inputs(digest,payload,saved,events)
    path=Path(lock_path).with_name(f"diagnostics-{saved['run_id']}.json")
    if path.exists():
        result=_load_saved_diagnostics(path,ledger,inputs)
        return _qualify(result,ledger,saved,verification_id)
    journal=path.with_suffix('.jsonl')
    if journal.exists(): raise ProtocolMismatch('Interrupted diagnostics retained; no automatic repeat')
    selected=payload['development']; forecast=selected['selected_forecast']
    attempts=[verification_id]
    def record(row):
        event=dict(schema_version=1,lock_sha256=digest,**plain(row))
        event['output_sha256']=canonical_sha256(event)
        with journal.open('a') as handle:
            handle.write(protocol_lock.canonical_bytes(event).decode()+'\n'); handle.flush(); os.fsync(handle.fileno())
    episodes={e.history.user_id:e for e in store.confirmation}
    result=dict(schema_version=1,lock_sha256=digest,confirmation_sha256=saved['output_sha256'],confirmation_status=saved['status'],
                confirmation_run_id=saved['run_id'],confirmation_artifact=inputs['confirmation_artifact'],
                inputs=inputs,input_sha256=canonical_sha256(inputs),
                confirmation_interpretation=saved['interpretation'],interpretation='secondary_unadjusted',status='completed',attempt_ids=attempts,artifact_path=str(path))
    with ledger.work('G5',forecast['seconds']['G5'],forecast['evaluations']['G5']) as aid:
        attempts.append(aid); record(dict(kind='started',ledger_attempt_id=aid))
        with threadpool_limits(limits=resolved.timing['blas_threads']):
            result['strata']=mechanism_strata(selected['confirmation_user_ids'],episodes,cache,selected,saved,resolved)
            labels={u:data.load_confirmation_outcomes(store,u,permit) for u in forecast['numerical_user_ids']}
            result['numerical']=numerical_measurements(forecast['numerical_user_ids'],selected['comparison_schedule'],saved['attempts'],cache,digest,resolved,ledger,record,labels)
            dev={e.history.user_id:e for e in store.development}
            if len(cache.item_ids)!=forecast['catalog_items']: raise ProtocolMismatch('Full active timing catalog changed')
            rows=timing_measurements(dev,forecast['timing_user_ids'],selected['comparison_schedule'],cache,resolved,ledger,record)
            calibration=json.loads((resolved.paths['results']/'calibration.json').read_text())
            result['timing']=timing_summary(rows,calibration,resolved)
            result['timing']['attempts']=rows
        result['resources']=ledger.checkpoint()
    plan=forecast['contamination']
    result['contamination_applicability']=contamination_applicability(saved,result['strata'],result['numerical'],plan,resolved)
    reason=contamination_skip_reason(contamination_applicable(saved,result['strata'],result['numerical'],plan,resolved),ledger.snapshot(),plan,resolved)
    if reason: result['contamination']=dict(status='skipped',reason=reason,plan=plan)
    else:
        try:
            with ledger.work('G5',plan['seconds'],plan['evaluations']) as aid:
                attempts.append(aid); record(dict(kind='contamination_started',ledger_attempt_id=aid))
                with threadpool_limits(limits=resolved.timing['blas_threads']):
                    result['contamination']=contamination_measurements(plan,episodes,cache,resolved,permit,store,ledger,record)
        except BudgetExceeded as error:
            result['contamination']=dict(status='unperformed_or_partial',reason=str(error),attempt_journal=str(journal))
    # A final zero-evaluation admission covers summary serialization and binds its
    # immutable output reference after all benchmark/optional admissions finish.
    try:
        with ledger.work('G5',1,0) as aid:
            attempts.append(aid)
            if 'attempts' in result['contamination']:
                clean={(r['user_id'],r['slot']):r['metrics']['auc'] for r in saved['attempts'] if r['status']=='completed'}
                result['contamination']['paired_by_seed']={}
                for seed in plan['seeds']:
                    comparisons={}
                    for slot in plan['schedule']:
                        corrupted={r['user_id']:r['metrics']['auc'] for r in result['contamination']['attempts']
                                   if r['seed']==seed and r['slot']==slot['name'] and r['status']=='completed'}
                        baseline={u:clean[(u,slot['name'])] for u in plan['user_ids'] if (u,slot['name']) in clean}
                        comparisons[slot['name']]=paired_analysis(plan['user_ids'],corrupted,baseline,delta=resolved.delta,
                            seed=resolved.bootstrap_seed,resamples=resolved.bootstrap_resamples,batch_size=resolved.bootstrap_batch_size)
                    result['contamination']['paired_by_seed'][str(seed)]=comparisons
            accounting=ledger.snapshot()['attempts']
            result['resource_measurements']={aid:accounting[aid].get('measurements',{}) for aid in attempts}
            result['timing']['peak_process_tree_rss_bytes']=max((r.get('rss_bytes',0) for r in result['resource_measurements'].values()),default=0)
            result['timing']['os_max_rss_bytes']=max((r.get('os_max_rss_bytes',0) for r in result['resource_measurements'].values()),default=0)
            result['resources']=ledger.checkpoint()
            if any(r['status']=='failed' for r in result['numerical']) or any(r['status']!='completed' for r in rows) or result['contamination']['status'] in ('partial','unperformed_or_partial'):
                result['status']='partial'
            result['attempt_journal']=file_reference(journal)
            result=plain(result)
            result['output_sha256']=canonical_sha256(result)
            protocol_lock.exclusive_json(path,result)
            ledger.finish(aid,'completed',{}, {'diagnostics':file_reference(path)})
    except BudgetExceeded:
        if not path.exists(): raise
    return _qualify(result,ledger,saved,verification_id)
