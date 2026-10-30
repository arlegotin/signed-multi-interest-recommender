"""Bounded development fits, scoring checks, and durable actual-attempt evidence."""
import time
import json
import os

import numpy as np

from .budget import BudgetExceeded, atomic_write_json
from .data import load_development_outcomes, episode_store_digest
from .metrics import evaluate_user
from .models import fit_user, score_user, initial_configurations
from .protocol import canonical_sha256
from .stage_evidence import plain


def calibration_entries(config):
    """All seven families, maximum geometric work, both solver endpoints."""
    entries=[]
    for entry in initial_configurations(config):
        p=entry.parameters
        if entry.family=='LinearLogistic':
            include=p['C'] in (config.C_grid[0],config.C_grid[-1])
        elif entry.family=='RBFSVM':
            include=p['C'] in (config.C_grid[0],config.C_grid[-1]) and p['g'] in (config.g_grid[0],config.g_grid[-1])
        else:
            include=p.get('K',4)==4 and p.get('lambda',8)==8
        if include:
            entries.append(entry)
    return entries


def measure_users(store,cache,users,config,ledger,*,numerical_checks,output_path):
    """One admitted batch. Load each user's development labels exactly once.

    Before each actual user/configuration evaluation, persist a started record.
    Repeated scoring consumes a separate evaluation; unstarted reservations are
    never represented as actual evaluations. Caller owns the batch admission.
    """
    episodes={e.history.user_id:e for e in store.development}
    if len(set(users))!=len(users) or any(uid not in episodes for uid in users):
        raise PermissionError('Workload users must be unique admitted development users')
    entries=calibration_entries(config) if numerical_checks else initial_configurations(config)
    result=dict(status='completed',attempts=[],actual_evaluations=0,user_measurements=[],resource_trace=[],
                chunk_invariance=True,repair_decisions=[],fixed_seconds=0.,optional_seconds=0.,memory_bound=False)
    inputs={'episodes':episode_store_digest(store),'user_ids':users,
            'entries':[e.id for e in entries],'numerical_checks':numerical_checks}

    def persist():
        atomic_write_json(output_path,dict(schema_version=1,input_sha256=canonical_sha256(inputs),
                          output_sha256=canonical_sha256(plain(result)),measurements=plain(result)))

    journal_path=output_path.with_suffix('.jsonl')
    journal_path.parent.mkdir(parents=True,exist_ok=True)

    def journal(row):
        # Each small record is durable before the next fit, avoiding quadratic
        # rewrites of every earlier user's score array on every new attempt.
        event={'schema_version':1,'input_sha256':canonical_sha256(inputs),'attempt':plain(row)}
        event['output_sha256']=canonical_sha256(event)
        with journal_path.open('a') as handle:
            handle.write(json.dumps(event,sort_keys=True,allow_nan=False)+'\n')
            handle.flush()
            os.fsync(handle.fileno())

    def start_attempt(uid,entry,kind,chunk):
        row=dict(evaluation_index=result['actual_evaluations'],user_id=uid,configuration_id=entry.id,family=entry.family,parameters=entry.parameters,
                 kind=kind,chunk_size=chunk,status='started',diagnostics={})
        result['attempts'].append(row)
        result['actual_evaluations']+=1
        journal(row)
        return row

    def repair(kind,reason):
        if not any(r['kind']==kind for r in result['repair_decisions']):
            result['repair_decisions'].append(dict(kind=kind,scope='all_methods_and_users',reason=reason,
                status='required',pre_freeze=True,applied=False))
        result['status']='repair_required'

    cap_users=set()
    for uid in users:
        user_start=time.perf_counter_ns()
        ledger.checkpoint()
        episode=episodes[uid]
        labels=load_development_outcomes(store,uid)
        input_seconds=(time.perf_counter_ns()-user_start)/1e9
        for entry in entries:
            boundary_start=time.perf_counter_ns()
            ledger.checkpoint()
            chunk=config.numerics['score_chunk_size']
            row=start_attempt(uid,entry,'fit_and_score',chunk)
            start=time.perf_counter_ns()
            model=None
            try:
                model=fit_user(entry.family,episode.history,cache,entry.parameters,config=config)
                fit_seconds=(time.perf_counter_ns()-start)/1e9
                row['diagnostics']=model.diagnostics
                if model.diagnostics['status']!='ok':
                    raise ValueError('Failed fit; see persisted model diagnostics')
                if model.diagnostics.get('prototype',{}).get('cap_hit'):
                    cap_users.add(uid)
                start=time.perf_counter_ns()
                scores=score_user(model,episode.candidate_ids,cache,chunk_size=chunk)
                metrics=evaluate_user(scores,labels,episode.candidate_ids)
                score_seconds=(time.perf_counter_ns()-start)/1e9
                row.update(status='completed',fit_seconds=fit_seconds,score_seconds=score_seconds,
                           metrics=metrics,scores=scores.tolist(),candidate_ids=list(episode.candidate_ids))
                start=time.perf_counter_ns()
                journal(row)
                write_seconds=(time.perf_counter_ns()-start)/1e9
                overhead_seconds=max(0.,(time.perf_counter_ns()-boundary_start)/1e9-fit_seconds-score_seconds-write_seconds)
                result['user_measurements'].append(dict(user_id=uid,family=entry.family,
                    configuration_id=entry.id,parameters=entry.parameters,K=entry.parameters.get('K',4),
                    support_count=len(episode.history.positive_ids)+len(episode.history.negative_ids),
                    positive_count=len(episode.history.positive_ids),negative_count=len(episode.history.negative_ids),
                    candidate_count=len(episode.candidate_ids),fit_seconds=fit_seconds,score_seconds=score_seconds,
                    write_seconds=write_seconds,overhead_seconds=overhead_seconds+input_seconds/len(entries)))
                row['write_seconds']=write_seconds
                if numerical_checks:
                    for alternate in config.numerics['score_chunk_sizes_check']:
                        if alternate==chunk:
                            continue
                        check=start_attempt(uid,entry,'score_repeat',alternate)
                        try:
                            start=time.perf_counter_ns()
                            repeated=score_user(model,episode.candidate_ids,cache,chunk_size=alternate)
                            repeated_metrics=evaluate_user(repeated,labels,episode.candidate_ids)
                            difference=float(np.max(np.abs(scores-repeated)))
                            tolerance=config.numerics['score_tolerance_factor']*(1+2*entry.parameters.get('lambda',0))
                            accepted=difference<=tolerance
                            check.update(status='completed' if accepted else 'failed',
                                score_seconds=(time.perf_counter_ns()-start)/1e9,max_absolute_difference=difference,
                                tolerance=tolerance,metrics=repeated_metrics,
                                metric_difference={k:repeated_metrics[k]-metrics[k] for k in metrics})
                            if not accepted:
                                result['chunk_invariance']=False
                                repair('numerical_investigation','Chunk score variation exceeds the fixed tolerance')
                        except BudgetExceeded:
                            check.update(status='interrupted',diagnostics=model.diagnostics)
                            raise
                        except Exception as error:
                            check.update(status='failed',exception=str(error),diagnostics=model.diagnostics)
                            result['chunk_invariance']=False
                            repair('numerical_investigation','Repeated scoring failed')
                        finally:
                            journal(check)
            except BudgetExceeded:
                row.update(status='interrupted',diagnostics={} if model is None else model.diagnostics)
                result['status']='interrupted'
                persist()
                raise
            except Exception as error:
                row.update(status='failed',exception=str(error),elapsed_seconds=(time.perf_counter_ns()-start)/1e9,
                           diagnostics={} if model is None else model.diagnostics)
                failures=row['diagnostics'].get('failures',[])
                kind='solver_iterations' if entry.family in ('LinearLogistic','RBFSVM') and any(
                    f in failures for f in ('iterations','warning','status')) else 'implementation_investigation'
                repair(kind,'Failed calibration fit/scoring; inspect complete attempt diagnostics')
            finally:
                journal(row)
        write_start=time.perf_counter_ns()
        persist()
        resource=ledger.checkpoint()
        result['resource_trace'].append({'user_id':uid,**resource})
        overhead=(time.perf_counter_ns()-write_start)/1e9/len(entries)
        for measured in result['user_measurements']:
            if measured['user_id']==uid:
                measured['overhead_seconds']+=overhead
    result['prototype_cap_user_fraction']=len(cap_users)/len(users) if users else 0.
    if result['prototype_cap_user_fraction']>config.prototype['cap_hit_inspection_fraction']:
        repair('prototype_trace_inspection','More than 5% of calibration users hit prototype cap; inspect objective traces')
    result['resources']=ledger.checkpoint()
    persist()
    return result
