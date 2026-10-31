"""Bounded development fits, scoring checks, and durable actual-attempt evidence."""
import time
from pathlib import Path

import numpy as np

from .budget import BudgetExceeded, atomic_write_json
from .data import load_development_outcomes, episode_store_digest
from .metrics import evaluate_user
from .models import fit_user, score_user, initial_configurations
from .protocol import canonical_sha256
from .stage_evidence import plain, config_record
from .workload_journal import (JournalWriteError, WorkloadJournal, initial_measurements,
                               replay_workload_journal)


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


def numerical_variation(reference,repeated,item_ids,*,eta):
    """Check the declared score/order contract using O(n) scratch memory.

    Pair ordering includes creation/removal of exact score ties. Log bounded
    item-ID examples and exact counts, rather than an unbounded pair matrix or
    a list of every near-tie change. Reference gaps exactly 2*eta are near ties.
    """
    reference=np.asarray(reference,dtype=np.float64)
    repeated=np.asarray(repeated,dtype=np.float64)
    ids=np.asarray(item_ids)
    if (reference.ndim!=1 or repeated.shape!=reference.shape or ids.shape!=reference.shape
            or not np.isfinite(reference).all() or not np.isfinite(repeated).all()
            or not np.isfinite(eta) or eta<0):
        raise ValueError('Numerical comparison needs aligned finite vectors and nonnegative eta')
    difference=float(np.max(np.abs(reference-repeated),initial=0.))
    near=distant=0
    examples=[]
    for i in range(len(reference)-1):
        gaps=reference[i]-reference[i+1:]
        repeated_gaps=repeated[i]-repeated[i+1:]
        changed=np.sign(gaps)!=np.sign(repeated_gaps)
        close=np.abs(gaps)<=2*eta
        near_mask=changed & close
        near+=int(np.count_nonzero(near_mask))
        distant+=int(np.count_nonzero(changed & ~close))
        if len(examples)<32:
            for offset in np.flatnonzero(near_mask)[:32-len(examples)]:
                examples.append({'item_ids':[int(ids[i]),int(ids[i+1+offset])],
                                 'reference_gap':float(gaps[offset]),
                                 'repeated_gap':float(repeated_gaps[offset])})
    return {'accepted':difference<=eta and distant==0,'tolerance':eta,
            'max_absolute_difference':difference,'pairwise_order_accepted':distant==0,
            'distant_order_changes':distant,'near_tie_order_changes':near,
            'near_tie_examples':examples,'near_tie_examples_truncated':near>len(examples)}


def measure_users(store,cache,users,config,ledger,*,numerical_checks,output_path,declared_entries=None,timing_only=False):
    """Measure a fresh admitted workload with durable starts and bounded snapshots.

    Version-2 journal replay recovers all final timing/resource fields without
    rerunning a fit. Partial geometric checkpoints never replace the final file;
    publication remains incomplete until the journal's final digest is durable.
    Caller owns admission and reservations, including unused reservations.
    """
    episodes={e.history.user_id:e for e in store.development}
    users=list(users)
    if len(set(users))!=len(users) or any(uid not in episodes for uid in users):
        raise PermissionError('Workload users must be unique admitted development users')
    entries=(calibration_entries(config) if numerical_checks else initial_configurations(config)) if declared_entries is None else list(declared_entries)
    if not entries or len({e.id for e in entries}) != len(entries):
        raise ValueError('Declared entries must be nonempty and unique')
    if numerical_checks and timing_only:
        raise ValueError('Numerical metric checks require outcome-enabled development work')
    output_path=Path(output_path)
    journal_path=output_path.with_suffix('.jsonl')
    partial_path=output_path.with_suffix('.partial.json')
    if output_path.exists() or partial_path.exists():
        raise FileExistsError('Existing workload evidence must be retained; choose a fresh admitted output path')
    result=initial_measurements()
    inputs={'episodes':episode_store_digest(store),'user_ids':users,
            'entries':[e.id for e in entries],'numerical_checks':numerical_checks,'timing_only':timing_only}
    identity=dict(inputs=inputs,config=config_record(config),
        entries=[dict(id=e.id,family=e.family,parameters=plain(e.parameters)) for e in entries],
        score_chunk_size=config.numerics['score_chunk_size'],
        score_chunk_sizes_check=list(config.numerics['score_chunk_sizes_check']),
        users=[dict(user_id=uid,positive_count=len(episodes[uid].history.positive_ids),
                    negative_count=len(episodes[uid].history.negative_ids),candidate_count=len(episodes[uid].candidate_ids),
                    candidate_ids_sha256=canonical_sha256({'item_ids':list(episodes[uid].candidate_ids)})) for uid in users])
    journal=WorkloadJournal(journal_path,inputs,identity)
    input_digest=canonical_sha256(inputs)
    terminal=set()
    timed=set()
    measured_indices={}
    finalized=False

    def persist(*,final=False):
        value=plain(result)
        if not final:
            value['status']='interrupted'
        atomic_write_json(output_path if final else partial_path,
                          dict(schema_version=1,input_sha256=input_digest,
                               output_sha256=canonical_sha256(value),measurements=value))

    def state():
        return plain({k:v for k,v in result.items() if k not in
                      ('attempts','actual_evaluations','user_measurements','resource_trace')})

    def start_attempt(uid,entry,kind,chunk):
        row=dict(evaluation_index=result['actual_evaluations'],user_id=uid,configuration_id=entry.id,family=entry.family,parameters=plain(entry.parameters),
                 kind=kind,chunk_size=chunk,status='started',diagnostics={})
        journal.append('attempt_started',attempt=plain(row))
        result['attempts'].append(row)
        result['actual_evaluations']+=1
        return row

    def finish_attempt(row):
        index=row['evaluation_index']
        if index not in terminal:
            start=time.perf_counter_ns()
            journal.append('attempt_terminal',attempt=plain(row))
            terminal.add(index)
            row['write_seconds']=(time.perf_counter_ns()-start)/1e9
        return row['write_seconds']

    def write_timing(row,measured=None,parent=None):
        index=row['evaluation_index']
        if index in timed:
            return
        if measured is None:
            journal.append('attempt_timing',evaluation_index=index,write_seconds=row['write_seconds'])
        else:
            journal.append('measurement',evaluation_index=parent['evaluation_index'],
                           timed_evaluation_index=index,write_seconds=row['write_seconds'],measurement=plain(measured))
        timed.add(index)

    def repair(kind,reason):
        if not any(r['kind']==kind for r in result['repair_decisions']):
            result['repair_decisions'].append(dict(kind=kind,scope='all_methods_and_users',reason=reason,
                status='required',pre_freeze=True,applied=False))
        result['status']='repair_required'

    cap_users=set()
    try:
        for ordinal,uid in enumerate(users,1):
            user_start=time.perf_counter_ns()
            user_attempt_start=len(result['attempts'])
            ledger.checkpoint()
            episode=episodes[uid]
            labels=None if timing_only else load_development_outcomes(store,uid)
            input_seconds=(time.perf_counter_ns()-user_start)/1e9
            user_measured=[]
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
                    row['diagnostics']=plain(model.diagnostics)
                    if model.diagnostics['status']!='ok':
                        raise ValueError('Failed fit; see persisted model diagnostics')
                    if 'P' in model.state:
                        row['prototype_identity']=canonical_sha256(dict(
                            positive_ids=list(episode.history.positive_ids),vectors=model.state['P'].tolist()))
                    if model.diagnostics.get('prototype',{}).get('cap_hit'):
                        cap_users.add(uid)
                    start=time.perf_counter_ns()
                    scores=score_user(model,episode.candidate_ids,cache,chunk_size=chunk)
                    metrics=None if timing_only else evaluate_user(scores,labels,episode.candidate_ids)
                    score_seconds=(time.perf_counter_ns()-start)/1e9
                    row.update(status='completed',fit_seconds=fit_seconds,score_seconds=score_seconds,
                               metrics=metrics,scores=scores.tolist(),candidate_ids=list(episode.candidate_ids))
                    write_seconds=finish_attempt(row)
                    overhead_seconds=max(0.,(time.perf_counter_ns()-boundary_start)/1e9-fit_seconds-score_seconds-write_seconds)
                    measured=dict(user_id=uid,family=entry.family,
                        configuration_id=entry.id,parameters=plain(entry.parameters),K=entry.parameters.get('K',4),
                        support_count=len(episode.history.positive_ids)+len(episode.history.negative_ids),
                        positive_count=len(episode.history.positive_ids),negative_count=len(episode.history.negative_ids),
                        candidate_count=len(episode.candidate_ids),fit_seconds=fit_seconds,score_seconds=score_seconds,
                        write_seconds=write_seconds,overhead_seconds=overhead_seconds+input_seconds/len(entries),
                        diagnostics=plain(model.diagnostics),
                        chunk_size=chunk,score_measurements={str(chunk):dict(score_seconds=score_seconds,
                            write_seconds=write_seconds,overhead_seconds=overhead_seconds+input_seconds/len(entries),
                            numerical_accepted=True)})
                    result['user_measurements'].append(measured)
                    measured_indices[id(measured)]=row['evaluation_index']
                    user_measured.append(measured)
                    write_timing(row,measured,row)
                    if numerical_checks:
                        for alternate in config.numerics['score_chunk_sizes_check']:
                            if alternate==chunk:
                                continue
                            repeat_start=time.perf_counter_ns()
                            check=start_attempt(uid,entry,'score_repeat',alternate)
                            repeat_overhead=(time.perf_counter_ns()-repeat_start)/1e9
                            try:
                                start=time.perf_counter_ns()
                                repeated=score_user(model,episode.candidate_ids,cache,chunk_size=alternate)
                                repeated_metrics=evaluate_user(repeated,labels,episode.candidate_ids)
                                repeated_score_seconds=(time.perf_counter_ns()-start)/1e9
                                diagnostic_start=time.perf_counter_ns()
                                tolerance=config.numerics['score_tolerance_factor']*(1+entry.parameters.get('lambda',0))
                                variation=numerical_variation(scores,repeated,episode.candidate_ids,eta=tolerance)
                                accepted=variation['accepted']
                                check.update(status='completed' if accepted else 'failed',
                                    score_seconds=repeated_score_seconds,
                                    diagnostic_seconds=(time.perf_counter_ns()-diagnostic_start)/1e9,
                                    **variation,metrics=repeated_metrics,scores=repeated.tolist(),
                                    candidate_ids=list(episode.candidate_ids),
                                    metric_difference={k:repeated_metrics[k]-metrics[k] for k in metrics})
                                if not accepted:
                                    result['chunk_invariance']=False
                                    repair('numerical_investigation','Chunk score variation or pair ordering violates the declared contract')
                            except BudgetExceeded:
                                check.update(status='interrupted',diagnostics=plain(model.diagnostics))
                                raise
                            except JournalWriteError:
                                raise
                            except OSError as error:
                                check.update(status='interrupted',exception=str(error),diagnostics=plain(model.diagnostics))
                                raise
                            except Exception as error:
                                check.update(status='failed',exception=str(error),diagnostics=plain(model.diagnostics))
                                result['chunk_invariance']=False
                                repair('numerical_investigation','Repeated scoring failed')
                            except BaseException:
                                check.update(status='interrupted',diagnostics=plain(model.diagnostics))
                                raise
                            finally:
                                if not journal.failed:
                                    finish_attempt(check)
                                    measured['score_measurements'][str(alternate)]=dict(
                                        score_seconds=check.get('score_seconds',0.),write_seconds=check['write_seconds'],
                                        overhead_seconds=max(measured['overhead_seconds'],repeat_overhead),
                                        numerical_accepted=check['status']=='completed')
                                    write_timing(check,measured,row)
                except BudgetExceeded:
                    if row['evaluation_index'] not in terminal:
                        row.update(status='interrupted',diagnostics={} if model is None else plain(model.diagnostics))
                    raise
                except JournalWriteError:
                    raise
                except OSError as error:
                    if row['evaluation_index'] not in terminal:
                        row.update(status='interrupted',exception=str(error),
                                   diagnostics={} if model is None else plain(model.diagnostics))
                    raise
                except Exception as error:
                    if row['evaluation_index'] not in terminal:
                        row.update(status='failed',exception=str(error),elapsed_seconds=(time.perf_counter_ns()-start)/1e9,
                                   diagnostics={} if model is None else plain(model.diagnostics))
                    failures=row['diagnostics'].get('failures',[])
                    kind='solver_iterations' if entry.family in ('LinearLogistic','RBFSVM') and any(
                        f in failures for f in ('iterations','warning','status')) else 'implementation_investigation'
                    repair(kind,'Failed calibration fit/scoring; inspect complete attempt diagnostics')
                except BaseException:
                    if row['evaluation_index'] not in terminal:
                        row.update(status='interrupted',diagnostics={} if model is None else plain(model.diagnostics))
                    raise
                finally:
                    if not journal.failed:
                        finish_attempt(row)
                        write_timing(row)
            resource=ledger.checkpoint()
            result['resource_trace'].append({'user_id':uid,**resource})
            # Powers-of-two checkpoints rewrite fewer than twice the final row
            # volume; the final aggregate is emitted once after all raw events.
            if ordinal & (ordinal-1)==0:
                persist()
            # Capture supplemental timing publications and numerical diagnostics
            # that occur after each entry's first overhead boundary. Attribute
            # only elapsed work not already represented by fit/score/write or
            # per-entry overhead. The closing event itself is outside this bound.
            accounted=sum(r.get('fit_seconds',0.)+r.get('score_seconds',0.)+r.get('write_seconds',0.)
                          for r in result['attempts'][user_attempt_start:])
            attributed=sum(m['overhead_seconds'] for m in user_measured)
            overhead=max(0.,(time.perf_counter_ns()-user_start)/1e9-accounted-attributed)/len(entries)
            for measured in user_measured:
                measured['overhead_seconds']+=overhead
                for timing in measured['score_measurements'].values():
                    timing['overhead_seconds']+=overhead
            journal.append('user_completed',user_id=uid,resource=plain(result['resource_trace'][-1]),
                measurements=[dict(evaluation_index=measured_indices[id(m)],measurement=plain(m)) for m in user_measured],state=state())
        result['prototype_cap_user_fraction']=len(cap_users)/len(users) if users else 0.
        if result['prototype_cap_user_fraction']>config.prototype['cap_hit_inspection_fraction']:
            repair('prototype_trace_inspection','More than 5% of calibration users hit prototype cap; inspect objective traces')
        result['resources']=ledger.checkpoint()
        journal.append('state',state=state())
        journal.append('finished',measurements_sha256=canonical_sha256(plain(result)))
        finalized=True
        recovered=replay_workload_journal(journal_path,expected_input_sha256=input_digest,
                                         expected_config=config_record(config))
        if not recovered['complete'] or recovered['measurements']!=plain(result):
            raise ValueError('Durable workload replay differs from final materialization')
        persist(final=True)
        return result
    except BaseException:
        result['status']='interrupted'
        # Preserve available partial evidence without another fit/inference. A
        # storage failure may leave a torn tail, so never append behind it.
        if not journal.failed and not finalized:
            try:
                journal.append('state',state=state())
            except (OSError,ValueError):
                pass
        try:
            persist()
        except (OSError,ValueError):
            pass
        raise
