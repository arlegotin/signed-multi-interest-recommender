"""Locked confirmation, durable exposure, and explicit exploratory corrections."""
import fcntl
import json
import os
from pathlib import Path
import time
import uuid

import numpy as np
from threadpoolctl import threadpool_limits

from . import data, protocol_lock
from .budget import BudgetExceeded, LedgerBusy
from .metrics import evaluate_user, paired_analysis
from .models import UserModel, fit_user, score_user
from .protocol import ProtocolMismatch, VerifiedProtocol, canonical_sha256
from .stage_evidence import plain, verify_record


def _events_path(lock):
    return Path(lock).with_name('confirmation.events.jsonl')


def read_events(lock):
    """Strict durable hash chain; a torn or changed exposure record fails closed."""
    path=_events_path(lock)
    rows=[]
    if path.exists():
        for line in path.read_text().splitlines():
            row=json.loads(line)
            verify_record(row)
            if row['previous_sha256']!=(rows[-1]['output_sha256'] if rows else None):
                raise ProtocolMismatch('Confirmation exposure chain changed')
            rows.append(row)
    marker=Path(lock).with_name('confirmation.exposure.json')
    if marker.exists():
        anchor=json.loads(marker.read_text())
        if not any(e['output_sha256']==anchor['first_exposure_sha256'] and e['kind']=='outcome_access' for e in rows):
            raise ProtocolMismatch('Persistent exposure history was removed')
    return rows


def _event(lock,kind,*,_journal=None,**fields):
    rows=read_events(lock) if _journal is None else _journal
    digest=rows[-1]['lock_sha256'] if _journal is not None and rows else protocol_lock.read_lock(lock)[0]
    row=dict(schema_version=1,kind=kind,lock_sha256=digest,
             previous_sha256=rows[-1]['output_sha256'] if rows else None,**plain(fields))
    row['output_sha256']=canonical_sha256(row)
    with _events_path(lock).open('a') as handle:
        handle.write(protocol_lock.canonical_bytes(row).decode()+'\n')
        handle.flush(); os.fsync(handle.fileno())
    if kind=='outcome_access':
        marker=Path(lock).with_name('confirmation.exposure.json')
        if not marker.exists():
            protocol_lock.exclusive_json(marker,dict(schema_version=1,lock_sha256=digest,first_exposure_sha256=row['output_sha256']))
    if _journal is not None: _journal.append(row)
    return row


def _exposed(events):
    # Intent is persisted before loading; interruption is conservatively exposure.
    return any(e['kind']=='outcome_access' for e in events)


def _invalidate(lock,reason):
    if _exposed(read_events(lock)):
        _event(lock,'invalidated',reason=reason,interpretation='invalid_confirmatory')


def _verify_correction(lock,live,authorization):
    digest,payload=protocol_lock.read_lock(lock)
    identities=live['identities']
    if identities!=authorization['identities']:
        raise ProtocolMismatch('Live correction identities changed after authorization')
    # Only implementation/current correctness evidence may change. Scientific
    # settings, IDs, data/model/cache, selections and schemas remain frozen.
    implementation={'source','correctness_sha256','metric_sha256'}
    if {k:v for k,v in identities.items() if k not in implementation}!={k:v for k,v in payload['identities'].items() if k not in implementation}:
        raise ProtocolMismatch('Correction changes frozen scientific inputs')
    if not identities['source']['clean'] or identities['source']==payload['identities']['source']:
        raise ProtocolMismatch('Correction requires clean changed implementation')
    protocol_lock.verify_lock_accounting(payload,digest,live)
    return VerifiedProtocol(digest,payload)


def _record_implementation_correction(lock_path,ledger,reason):
    """Invalidate first, then authorize a same-user exploratory implementation fix.

    This metadata operation does not execute the correction or consume a repair
    cycle. confirm owns one complete repair admission for the corrected rerun.
    """
    if not isinstance(reason,str) or not reason.strip():
        raise ValueError('Correction requires a concrete reason')
    if not _exposed(read_events(lock_path)):
        raise ProtocolMismatch('Post-exposure correction requires exposure evidence')
    with ledger.work('G4',1,0):
        _invalidate(lock_path,reason)
        live=protocol_lock.live_inputs(ledger.config,ledger)
        authorization=dict(identities=live['identities'],reason=reason,
                           interpretation='corrected_exploratory',correction_id=uuid.uuid4().hex)
        _verify_correction(lock_path,live,authorization)
        return _event(lock_path,'correction_authorized',**authorization)


def _save_geometry(lock,permit,run_id,row,episode,model,cache):
    path=Path(lock).with_name(f"confirmation-state-{run_id}-{row['user_id']}-{row['slot']}.npy")
    with path.open('xb') as handle:
        np.save(handle,np.ascontiguousarray(model.state['P'],dtype=np.float64),allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    state=dict(schema_version=1,lock_sha256=permit.lock_hash,run_id=run_id,
               user_id=row['user_id'],slot=row['slot'],fit_family=row['fit_family'],parameters=row['parameters'],
               support_ids=list(model.support_ids),positive_ids=list(episode.history.positive_ids),
               negative_ids=list(episode.history.negative_ids),positive_vectors=protocol_lock.evidence.file_reference(path),
               prototype_identity=row['prototype_identity'],cache_sha256=cache.manifest['output_sha256'],
               diagnostics=plain(model.diagnostics))
    state['output_sha256']=canonical_sha256(state)
    return state


def load_frozen_geometry(state,cache,lock_hash):
    """Restore original float64 geometry from saved P and immutable N cache rows.

    No fitting, outcome access, or estimator deserialization. The caller owns
    verified lock admission and diagnostic accounting; cast the returned fixed
    arrays for float32 checks without refitting. State is only saved for frozen
    numerical_user_ids, and its hash is bound by the immutable confirmation row.
    """
    verify_record(state)
    if state['lock_sha256']!=lock_hash or state['cache_sha256']!=cache.manifest['output_sha256']:
        raise ProtocolMismatch('Diagnostic geometry lock/cache identity mismatch')
    protocol_lock.evidence.verify_reference(state['positive_vectors'])
    P=np.load(state['positive_vectors']['path'],allow_pickle=False)
    if P.dtype!=np.float64 or P.ndim!=2 or not np.isfinite(P).all():
        raise ValueError('Invalid saved float64 prototype geometry')
    if canonical_sha256(dict(positive_ids=state['positive_ids'],vectors=P.tolist()))!=state['prototype_identity']:
        raise ProtocolMismatch('Saved actual prototype identity changed')
    N=np.ascontiguousarray(cache.rows(state['negative_ids']),dtype=np.float64)
    return UserModel(state['fit_family'],dict(state['parameters']),tuple(state['support_ids']),
                     dict(P=P,N=N),dict(state['diagnostics']))


def _summary(lock,permit,run_id,interpretation,attempts,attempt_ids,stop_reason):
    selected=permit.payload['development']; users=selected['confirmation_user_ids']
    auc={s['name']:{} for s in selected['comparison_schedule']}
    for row in attempts:
        if row['status']=='completed': auc[row['slot']][row['user_id']]=row['metrics']['auc']
    config=permit.payload['config']
    comparisons={name:paired_analysis(users,auc['SignedMax'],values,delta=config['delta'],
                 seed=config['bootstrap_seed'],resamples=config['bootstrap_resamples'],batch_size=config['bootstrap_batch_size'])
                 for name,values in auc.items() if name!='SignedMax'}
    path=Path(lock).with_name(f'confirmation-{run_id}.json')
    result=dict(schema_version=1,lock_sha256=permit.lock_hash,run_id=run_id,
                interpretation=interpretation,status='completed' if all(len(v)==len(users) for v in auc.values()) else 'partial',
                scheduled_user_ids=users,comparison_schedule=selected['comparison_schedule'],
                attempts=attempts,attempt_ids=attempt_ids,actual_evaluations=len(attempts),comparisons=comparisons,
                primary_comparator=selected['B_star']['family'],primary=comparisons[selected['B_star']['family']],
                comparison_scope='complete_suite' if selected['complete_suite_eligible'] else 'coverage_limited',
                complete_suite_eligible=selected['complete_suite_eligible'],endpoint_coverage=selected['endpoint_coverage'],
                boundary_winners=selected['boundary_winners'],stop_reason=stop_reason,
                artifact_path=str(path),exposure_journal=str(_events_path(lock)))
    result['output_sha256']=canonical_sha256(result)
    protocol_lock.exclusive_json(path,result)
    _event(lock,'run_finished',run_id=run_id,artifact=protocol_lock.evidence.file_reference(path),interpretation=interpretation)
    return result


def _confirm(lock_path,ledger):
    """Verify first, then load each admitted user's labels once for all slots.

    A finished partial run is terminal: repeated invocations return its saved
    evidence. Interrupted work retains starts and stops the affected family;
    no failed attempt is silently retried. Corrections use a new run identity.
    """
    lock_path=Path(lock_path)
    # Verification failures never expose arrays, even on resume.
    try:
        with ledger.work('G4',1,0):
            events=read_events(lock_path)
            authorizations=[e for e in events if e['kind']=='correction_authorized']
            authorization=authorizations[-1] if authorizations else None
            live,resolved,store,cache,_,_=protocol_lock._context(ledger.config,ledger)
            permit=(_verify_correction(lock_path,live,authorization) if authorization else
                    protocol_lock.verify_protocol(lock_path,live))
    except (ValueError,KeyError,OSError) as error:
        _invalidate(lock_path,str(error))
        raise ProtocolMismatch(str(error)) from error
    interpretation=('corrected_exploratory' if authorization else
                    'invalid_confirmatory' if any(e['kind']=='invalidated' for e in events) else 'confirmatory')
    run_id=authorization['correction_id'] if authorization else permit.lock_hash
    events=read_events(lock_path)
    finished=[e for e in events if e['kind']=='run_finished' and e['run_id']==run_id]
    if finished:
        ref=finished[-1]['artifact']; protocol_lock.evidence.verify_reference(ref)
        result=json.loads(Path(ref['path']).read_text()); verify_record(result)
        accounting=ledger.snapshot()['attempts']
        failed=[accounting[aid]['status'] for aid in result['attempt_ids'] if accounting[aid]['status']!='completed']
        if result['interpretation']!=interpretation or failed:
            result.update(interpretation=interpretation,original_output_sha256=result.pop('output_sha256'))
            if failed:
                result.update(status='budget_exceeded' if 'budget_exceeded' in failed else 'partial',
                              accounting_status=failed,stop_reason='Execution accounting did not complete successfully')
            result['output_sha256']=canonical_sha256(result)
        return result
    selected=permit.payload['development']; users=selected['confirmation_user_ids']
    rows={}
    for event in events:
        if event.get('run_id')==run_id and event['kind']=='evaluation':
            row=event['attempt']; rows[(row['user_id'],row['slot'])]=row
    attempts=list(rows.values())
    stopped={row['fit_family'] for row in attempts if row['status']!='completed'}
    episodes={e.history.user_id:e for e in store.confirmation}
    remaining=sum((uid,s['name']) not in rows and s['fit_family'] not in stopped
                  for uid in users for s in selected['comparison_schedule'])
    stage='repair' if authorization else 'G4'
    # Interrupted repairs may not acquire another repair cycle to continue.
    prior=[e for e in events if e['kind']=='run_started' and e['run_id']==run_id]
    if prior:
        with ledger.work('G4',1,0):
            return _summary(lock_path,permit,run_id,interpretation,attempts,
                            [e['ledger_attempt_id'] for e in prior],'Interrupted execution; retained attempts, no automatic retry')
    attempt_ids=[]; stop_reason=None
    journal=read_events(lock_path)
    try:
        with ledger.work(stage,selected['selected_forecast']['seconds']['G4'],remaining) as aid:
            attempt_ids.append(aid)
            _event(lock_path,'run_started',_journal=journal,run_id=run_id,ledger_attempt_id=aid,interpretation=interpretation)
            with threadpool_limits(limits=resolved.timing['blas_threads']):
                for uid in users:
                    ledger.checkpoint()
                    slots=[s for s in selected['comparison_schedule'] if (uid,s['name']) not in rows and s['fit_family'] not in stopped]
                    if not slots: continue
                    _event(lock_path,'outcome_access',_journal=journal,run_id=run_id,user_id=uid,interpretation=interpretation)
                    labels=data.load_confirmation_outcomes(store,uid,permit)
                    episode=episodes[uid]
                    for slot in slots:
                        if slot['fit_family'] in stopped: continue
                        ledger.checkpoint()
                        start=time.perf_counter()
                        row=dict(user_id=uid,slot=slot['name'],configuration_id=slot['id'],fit_family=slot['fit_family'],
                                 parameters=slot['parameters'],status='started',candidate_ids=list(episode.candidate_ids),
                                 scores=None,metrics=None,diagnostics={},exception=None,elapsed_seconds=0.)
                        attempts.append(row)
                        _event(lock_path,'evaluation',_journal=journal,run_id=run_id,attempt=row)
                        try:
                            model=fit_user(slot['fit_family'],episode.history,cache,slot['parameters'],config=resolved)
                            row['diagnostics']=plain(model.diagnostics)
                            if model.diagnostics['status']!='ok': raise ValueError('Failed fit; see diagnostics')
                            scores=score_user(model,episode.candidate_ids,cache,chunk_size=resolved.numerics['score_chunk_size'])
                            row.update(scores=scores.tolist(),metrics=evaluate_user(scores,labels,episode.candidate_ids),status='completed')
                            if 'P' in model.state:
                                row['prototype_identity']=canonical_sha256(dict(positive_ids=list(episode.history.positive_ids),vectors=model.state['P'].tolist()))
                                if uid in selected['selected_forecast']['numerical_user_ids']:
                                    row['fitted_state']=_save_geometry(lock_path,permit,run_id,row,episode,model,cache)
                        except BudgetExceeded: raise
                        except Exception as error:
                            row.update(status='failed',exception=f'{type(error).__name__}: {error}')
                            stopped.add(slot['fit_family'])
                        finally:
                            row['elapsed_seconds']=time.perf_counter()-start
                            _event(lock_path,'evaluation',_journal=journal,run_id=run_id,attempt=row)
            result=_summary(lock_path,permit,run_id,interpretation,attempts,attempt_ids,stop_reason)
            ledger.finish(aid,'completed',{}, {'confirmation':protocol_lock.evidence.file_reference(result['artifact_path'])})
            return result
    except (BudgetExceeded,OSError,ValueError) as error:
        stop_reason=f'{type(error).__name__}: {error}'
        # Persist inexpensive raw partial evidence even if analysis admission fails.
        partial=dict(schema_version=1,lock_sha256=permit.lock_hash,run_id=run_id,
                     interpretation=interpretation,status='analysis_pending',attempts=attempts,
                     scheduled_user_ids=users,attempt_ids=attempt_ids,stop_reason=stop_reason)
        path=lock_path.with_name(f'confirmation-partial-{run_id}-{uuid.uuid4().hex}.json')
        partial['artifact_path']=str(path)
        partial['output_sha256']=canonical_sha256(partial)
        protocol_lock.exclusive_json(path,partial)
        return partial


def _serialized(lock_path,operation,*args):
    path=Path(lock_path).with_name('confirmation.writer.lock')
    with path.open('a') as handle:
        try:
            fcntl.flock(handle,fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise LedgerBusy('Confirmation evidence already has an active writer') from error
        try:
            return operation(lock_path,*args)
        finally:
            fcntl.flock(handle,fcntl.LOCK_UN)


def confirm(lock_path,ledger):
    return _serialized(lock_path,_confirm,ledger)


def record_implementation_correction(lock_path,ledger,reason):
    return _serialized(lock_path,_record_implementation_correction,ledger,reason)
