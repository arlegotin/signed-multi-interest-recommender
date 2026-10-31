"""Ordered acquisition, audit, calibration and development-only smoke gates."""
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import time
import uuid

from threadpoolctl import threadpool_limits, threadpool_info

from .budget import BudgetExceeded, LedgerBusy, atomic_write_json, results_first
from .calibration import forecast_scale, select_calibration_users
from .data import (SOURCES, Movie, episode_store_digest, _acquire_archive as acquire_archive, load_dataset,
                   make_episodes, load_episode_store)
from .embed import (resolve_encoder, expected_embedding_manifest, load_encoder,
                    encode_probe, check_repeatability, encode_items, load_embedding_cache)
from .protocol import load_config, canonical_sha256
from .stage_evidence import (config_record, file_reference, read_stage, save_stage,
                             verify_record, verify_correctness, environment_pins,
                             write_environment_lock, verify_environment_lock, verify_reference)
from .workload import measure_users, calibration_entries


class StageBlocked(ValueError):
    pass


def calibration_measurements(config, record):
    """Read the verified calibration role, including versioned adopted receipts."""
    reference=record['artifacts']['calibration']
    verify_reference(reference)
    result=json.loads(Path(reference['path']).read_text())
    verify_record(result)
    return result


def _official(manifest):
    name=manifest.get('dataset')
    if (name not in SOURCES or manifest.get('status')!='validated'
            or manifest.get('archive_sha256')!=SOURCES[name].sha256):
        raise StageBlocked('Validated official dataset acquisition provenance is required')
    if name=='ml-100k' and manifest.get('access_fallback') is not True:
        raise StageBlocked('100K requires the recorded access-only fallback')


def _read(config,name):
    return json.loads((config.paths['results']/name).read_text())


def _prerequisite(config,name,ledger,inputs=None):
    record=read_stage(config,name,inputs)
    attempts=ledger.snapshot()['attempts']
    if not record['attempt_ids'] or any(attempts[a]['status']!='completed' for a in record['attempt_ids']):
        raise StageBlocked(f'{name} has no completed ledger evidence')
    return record


def _acquisition_inputs(config):
    environment=_read(config,'environment.json')
    verify_record(environment)
    if environment.get('target_compatible') is not True or environment.get('blockers'):
        raise StageBlocked('G0 requires verified compatible environment evidence')
    from .run import _PACKAGES
    pins=environment_pins()
    packages=environment.get('packages',{})
    if any(packages.get(name) is None or packages[name]!=pins.get(name) for name in _PACKAGES):
        raise StageBlocked('Environment package provenance is unresolved or changed; inspect again')
    identity=config_record(config)
    # These validated global policies do not acquire different data/model files.
    # G1/calibration identities still bind every effective policy value.
    identity['pre_freeze_repairs']=[]
    for group,key,value in [('prototype','max_updates',50),('linear_logistic','max_iter',2000),
                            ('rbf_svm','max_iter',100000),('encoder','batch_size',32),
                            ('numerics','score_chunk_size',512),('timing','blas_threads',1)]:
        identity[group][key]=value
    identity['bootstrap_batch_size']=200
    return {'config':identity,'environment':file_reference(config.paths['results']/'environment.json')}


def _acquired(config,ledger):
    record=_prerequisite(config,'acquire',ledger,_acquisition_inputs(config))
    resolved=load_config(config.paths['results']/'resolved_config.json')
    verify_environment_lock(Path('requirements.lock.txt'))
    resolved=replace(resolved,prototype=config.prototype,linear_logistic=config.linear_logistic,
                     rbf_svm=config.rbf_svm,pre_freeze_repairs=config.pre_freeze_repairs,
                     encoder=config.encoder,numerics=config.numerics,timing=config.timing,
                     bootstrap_batch_size=config.bootstrap_batch_size)
    manifest=_read(config,'data_manifest.json')
    _official(manifest)
    encoder=_read(config,'encoder_acquisition.json')
    metadata={row['item_id']:Movie(row['item_id'],row['title'],tuple(row['genres']))
              for row in _read(config,'metadata.json')['movies']}
    expected=expected_embedding_manifest(metadata,resolved,encoder_provenance=encoder,dataset_manifest=manifest)
    return record,resolved,manifest,encoder,metadata,expected


def _acquire(config,ledger,attempts):
    with ledger.work('G0',0,0) as check:
        attempts.append(check)
        inputs=_acquisition_inputs(config)
        path=config.paths['results']/'stages'/'acquire.json'
        if path.exists():
            return _acquired(config,ledger)[0]
    encoder_path=config.paths['results']/'encoder_acquisition.json'
    with ledger.work('setup',60,0) as aid:
        attempts.append(aid)
        acquire_archive(config)
        # An interrupted G0 must retain its pinned revision rather than resolve
        # a mutable upstream revision for the second time.
        encoder=_read(config,'encoder_acquisition.json') if encoder_path.exists() else resolve_encoder(config)
        atomic_write_json(encoder_path,encoder)
    with ledger.work('G0',10,0) as aid:
        attempts.append(aid)
        start=time.perf_counter_ns()
        dataset=load_dataset(config)
        parse_seconds=(time.perf_counter_ns()-start)/1e9
        _official(dataset.manifest)
        environment_hash=inputs['environment']['sha256']
        if config.environment_hash not in (None,environment_hash):
            raise StageBlocked('Configured environment identity differs from acquisition evidence')
        resolved=replace(config,model_revision=encoder['model_revision'],environment_hash=environment_hash)
        atomic_write_json(config.paths['results']/'resolved_config.json',config_record(resolved))
        resolved=load_config(config.paths['results']/'resolved_config.json')
        expected_embedding_manifest(dataset.metadata,resolved,encoder_provenance=encoder,dataset_manifest=dataset.manifest)
        atomic_write_json(config.paths['results']/'data_manifest.json',dataset.manifest)
        metadata_record={
            'schema_version':1,'input_sha256':canonical_sha256(dataset.manifest),
            'movies':[asdict(dataset.metadata[i]) for i in sorted(dataset.metadata)]}
        metadata_record['output_sha256']=canonical_sha256(metadata_record)
        atomic_write_json(config.paths['results']/'metadata.json',metadata_record)
        files={name:config.paths['results']/name for name in
               ('encoder_acquisition.json','data_manifest.json','resolved_config.json','metadata.json')}
        files['archive']=config.paths['data']/'raw'/f"{dataset.manifest['dataset']}.zip"
        write_environment_lock(Path('requirements.lock.txt'))
        files['environment_lock']=Path('requirements.lock.txt')
        result=save_stage(config,'acquire',inputs,files,attempts,measurements={'parse_validate_seconds':parse_seconds},
                          environment_pins_status='provisional_until_current_G1')
    return result


def _audit(config,ledger,attempts):
    with ledger.work('G0',0,0) as check:
        attempts.append(check)
        acquired,resolved,manifest,*_= _acquired(config,ledger)
        inputs={'acquire':acquired['output_sha256']}
        if (config.paths['results']/'stages'/'audit.json').exists():
            result=_prerequisite(config,'audit',ledger,inputs)
            load_episode_store(result['artifacts']['audit']['path'],result['artifacts']['episodes']['path'])
            return result
    with ledger.work('G0',10,0) as aid:
        attempts.append(aid)
        start=time.perf_counter_ns()
        dataset=load_dataset(resolved)
        if dataset.manifest!=manifest:
            raise StageBlocked('Dataset changed since acquisition')
        store=make_episodes(dataset,resolved)
        elapsed=(time.perf_counter_ns()-start)/1e9
        result=save_stage(config,'audit',inputs,{'audit':store.paths['audit'],'episodes':store.paths['episodes']},
                          attempts,measurements={'parse_validate_episode_seconds':elapsed})
    return result


def _calibration_context(config,ledger):
    acquired,resolved,manifest,encoder,metadata,expected=_acquired(config,ledger)
    audit=_prerequisite(config,'audit',ledger,{'acquire':acquired['output_sha256']})
    correctness=verify_correctness(config,ledger)
    store=load_episode_store(audit['artifacts']['audit']['path'],audit['artifacts']['episodes']['path'])
    if store.audit['dataset']!=manifest['dataset']:
        raise StageBlocked('Audit and acquisition dataset differ')
    inputs={'acquire':acquired['output_sha256'],'audit':audit['output_sha256'],
            'correctness':correctness['output_sha256']}
    atomic_write_json(config.paths['results']/'resolved_policy.json',config_record(resolved))
    resolved=load_config(config.paths['results']/'resolved_policy.json')
    return inputs,resolved,manifest,encoder,metadata,expected,store


def _calibrate(config,ledger,attempts):
    with ledger.work('G0',0,0) as check:
        attempts.append(check)
        inputs,resolved,manifest,provenance,metadata,expected,store=_calibration_context(config,ledger)
        cache_path=resolved.paths['cache']/'embeddings'
        if (config.paths['results']/'stages'/'calibrate.json').exists():
            result=_prerequisite(config,'calibrate',ledger,inputs)
            load_embedding_cache(cache_path,expected)
            return result
        if (config.paths['results']/'calibration.json').exists():
            raise StageBlocked('Saved calibration publication exists; use resume-calibration and preserve its history')
    measurements={'schema_version':1,'input_sha256':canonical_sha256(inputs), 'timings':{},'catalog_items':len(metadata)}
    def timed(name,call):
        start=time.perf_counter_ns()
        result=call()
        measurements['timings'][name]=(time.perf_counter_ns()-start)/1e9
        return result
    probe_ids=sorted(metadata,key=lambda i:hashlib.sha256(f'signed-study:item:{i}'.encode()).hexdigest())[:128]
    if len(probe_ids)!=128:
        raise StageBlocked('Calibration needs the full catalog and 128 probe items')
    repeat_ids=probe_ids[:32]
    users=select_calibration_users(list(store.development))
    measurements.update(probe_item_ids=probe_ids,repeat_item_ids=repeat_ids,calibration_user_ids=users)
    kwargs=dict(encoder_provenance=provenance,dataset_manifest=manifest,checkpoint=ledger.checkpoint)
    with ledger.work('G2',15,0) as aid:
        attempts.append(aid)
        encoder=timed('cold_model_load_seconds',lambda:load_encoder(resolved,provenance))
        with threadpool_limits(limits=resolved.timing['blas_threads']):
            probe=timed('probe_seconds',lambda:encode_probe(metadata,resolved,item_ids=probe_ids,encoder=encoder,**kwargs))
            repeat=timed('repeat_seconds',lambda:encode_probe(metadata,resolved,item_ids=repeat_ids,encoder=encoder,**kwargs))
            measurements['encoder_probe_policy']=dict(getattr(encoder,'thread_policy',{}),threadpools=threadpool_info())
        measurements['repeatability']=check_repeatability(probe.raw_rows(repeat_ids),repeat.raw_rows(repeat_ids))
        measurements['probe_resources']=ledger.checkpoint()
        probe_record={**measurements,'output_sha256':canonical_sha256(measurements)}
        atomic_write_json(config.paths['results']/'calibration_probe.json',probe_record)
        if not measurements['repeatability']['accepted']:
            raise StageBlocked('Raw encoder repeatability failed; investigate before full-cache admission')
    projected=measurements['timings']['probe_seconds']*len(metadata)/128
    # This is a distinct admission: full encoding cannot hide inside the probe.
    with ledger.work('G2',projected,0) as aid:
        attempts.append(aid)
        with threadpool_limits(limits=resolved.timing['blas_threads']):
            timed('catalog_encode_seconds',lambda:encode_items(metadata,resolved,encoder=encoder,**kwargs))
            measurements['encoder_catalog_policy']=dict(getattr(encoder,'thread_policy',{}),threadpools=threadpool_info())
        cache=timed('cache_load_verify_seconds',lambda:load_embedding_cache(cache_path,expected))
        measurements['cache_resources']=ledger.checkpoint()
    del encoder
    chunks=set(resolved.numerics['score_chunk_sizes_check']) | {resolved.numerics['score_chunk_size']}
    reservation=len(users)*len(calibration_entries(resolved))*len(chunks)
    workload_path=config.paths['results']/f'calibration_attempts-{uuid.uuid4().hex}.json'
    with ledger.work('G2',30,reservation) as aid:
        attempts.append(aid)
        with threadpool_limits(limits=resolved.timing['blas_threads']):
            measured=measure_users(store,cache,users,resolved,ledger,numerical_checks=True,output_path=workload_path)
            measurements['threadpools']=threadpool_info()
        measurements.update(measured)
        measurements['reserved_evaluations']=reservation
        measurements['policy']=config_record(resolved)
        measurements['forecast']=forecast_scale(measurements,store,ledger) if measured['status']=='completed' else None
        measurements['resources']=ledger.checkpoint()
        measurements['output_sha256']=canonical_sha256(measurements)
        atomic_write_json(config.paths['results']/'calibration.json',measurements)
        if measured['status']!='completed':
            ledger.finish(aid,'blocked',{}, {'calibration':file_reference(config.paths['results']/'calibration.json')})
            return dict(status='repair_required',artifacts={'calibration':file_reference(config.paths['results']/'calibration.json'),
                        'attempts':file_reference(workload_path)},attempt_ids=attempts,
                        stop_reason='Unresolved calibration validity/convergence evidence',repair_decisions=measured['repair_decisions'])
        forecast=measurements['forecast']
        if forecast['status']!='feasible':
            # Preserve the exact measured stop evidence at its owning terminal
            # admission. Reporting must not trust a later self-checksummed file.
            ledger.finish(aid,'failed',{'exception':forecast['stop_reason']},
                          {'calibration':file_reference(config.paths['results']/'calibration.json'),
                           'attempts':file_reference(workload_path)})
            raise StageBlocked(forecast['stop_reason'])
        policy=forecast['policy']
        resolved=replace(resolved,encoder={**resolved.encoder,'batch_size':policy['embedding_batch_size']},
                         numerics={**resolved.numerics,'score_chunk_size':policy['score_chunk_size']})
        atomic_write_json(config.paths['results']/'calibration_config.json',config_record(resolved))
        files={'calibration':config.paths['results']/'calibration.json','attempts':workload_path,
               'policy':config.paths['results']/'calibration_config.json',
               **{f'cache_{name}':cache_path/name for name in ('manifest.json','vectors.npy','item_ids.npy')}}
        if workload_path.with_suffix('.jsonl').exists():
            files['attempt_journal']=workload_path.with_suffix('.jsonl')
        result=save_stage(config,'calibrate',inputs,files,attempts,forecast=forecast)
    return result


def _saved_calibration(config,ledger,inputs,resolved,store,metadata):
    """Qualify the exact failed forecast publication, never a rewritten summary."""
    from . import stage_evidence as evidence
    path=config.paths['results']/'calibration.json'
    reference=file_reference(path)
    measured=json.loads(path.read_text())
    verify_record(measured)
    reason='Smallest permitted scale cannot fit remaining allocations and reserve'
    state=ledger.snapshot()
    parents=[a for a in state['attempts'].values() if a['stage']=='G2' and a['status']=='failed'
             and a.get('measurements',{}).get('exception')==reason
             and a.get('outputs',{}).get('calibration')==reference]
    if (len(parents)!=1 or measured.get('status')!='completed'
            or (measured.get('forecast') or {}).get('status')!='infeasible'
            or measured['forecast'].get('stop_reason')!=reason):
        raise StageBlocked('Saved calibration lacks the original failed forecast publication')
    parent=parents[0]
    failed=[]
    for candidate in (config.paths['results']/'stages').glob('calibrate.failed-*.json'):
        record=json.loads(candidate.read_text()); verify_record(record)
        if record.get('attempt_ids',[])[-1:]==[parent['attempt_id']] and record.get('stop_reason')==reason:
            failed.append((candidate,record))
    if len(failed)!=1:
        raise StageBlocked('Saved calibration requires its original failed invocation')
    for aid in failed[0][1]['attempt_ids']:
        if aid!=parent['attempt_id'] and state['attempts'][aid]['status']!='completed':
            raise StageBlocked('Saved calibration has unsuccessful prerequisite work')
    proofs=[]
    for candidate in config.paths['results'].glob('correctness*.json'):
        proof=json.loads(candidate.read_text()); verify_record(proof)
        previous={**inputs,'correctness':proof['output_sha256']}
        if measured['input_sha256']==canonical_sha256(previous):
            proofs.append((candidate,proof))
    if not proofs:
        raise StageBlocked('Original calibration G1 evidence is missing')
    proof_path,proof=proofs[0]
    if (proof.get('status')!='completed' or proof.get('returncode')!=0
            or not proof.get('command_attempt_ids')
            or proof['input_sha256']!=canonical_sha256(proof['inputs'])
            or any(state['attempts'][a]['stage']!='G1' or state['attempts'][a]['status']!='completed'
                   for a in proof['attempt_ids'])
            or any(state['attempts'][a].get('outputs',{}).get('returncode')!=0
                   or state['attempts'][a]['outputs'].get('command')!=proof['command']
                   for a in proof['command_attempt_ids'])):
        raise StageBlocked('Original calibration G1 proof is unsuccessful or unbound')
    current=evidence.correctness_inputs(config)
    if any(proof['inputs'].get(key)!=current[key] for key in ('config','environment','environment_lock')):
        raise StageBlocked('Original scientific settings or environment changed')
    scientific=('data.py','embed.py','models.py','metrics.py','protocol.py','workload.py')
    if any(not proof['inputs']['code'].get('src/signed_study/'+name)
           or proof['inputs']['code']['src/signed_study/'+name]!=current['code'].get('src/signed_study/'+name)
           for name in scientific):
        raise StageBlocked('Measured scientific implementation changed; remeasurement is required')
    if measured.get('policy')!=config_record(resolved):
        raise StageBlocked('Saved calibration scientific policy changed')
    users=select_calibration_users(list(store.development))
    probes=sorted(metadata,key=lambda i:hashlib.sha256(f'signed-study:item:{i}'.encode()).hexdigest())[:128]
    if (measured.get('calibration_user_ids')!=users or measured.get('probe_item_ids')!=probes
            or measured.get('repeat_item_ids')!=probes[:32] or measured.get('catalog_items')!=len(metadata)
            or measured.get('repeatability',{}).get('accepted') is not True
            or measured.get('chunk_invariance') is not True or measured.get('repair_decisions')
            or measured.get('memory_bound') is not False
            or measured.get('prototype_cap_user_fraction',1)>resolved.prototype['cap_hit_inspection_fraction']):
        raise StageBlocked('Saved calibration has unresolved numerical, sampling or memory evidence')
    workload_reference=parent['outputs']['attempts']; verify_reference(workload_reference)
    workload_path=Path(workload_reference['path'])
    workload=json.loads(workload_path.read_text())
    entries=calibration_entries(resolved)
    original_inputs=dict(episodes=episode_store_digest(store),user_ids=users,entries=[e.id for e in entries],
                         numerical_checks=True,timing_only=False)
    digest=canonical_sha256(original_inputs)
    work=workload['measurements']
    if (workload.get('schema_version')!=1 or workload.get('input_sha256')!=digest
            or workload.get('output_sha256')!=canonical_sha256(work)
            or any(measured.get(k)!=work.get(k) for k in ('status','attempts','actual_evaluations',
                'user_measurements','repair_decisions','chunk_invariance','prototype_cap_user_fraction','memory_bound'))):
        raise StageBlocked('Saved calibration workload does not match its original publication')
    chunks=set(resolved.numerics['score_chunk_sizes_check'])|{resolved.numerics['score_chunk_size']}
    expected={(uid,e.id,chunk) for uid in users for e in entries for chunk in chunks}
    rows=work['attempts']
    if (work['actual_evaluations']!=len(expected) or len(rows)!=len(expected)
            or measured.get('reserved_evaluations')!=len(expected) or parent['evaluations']!=len(expected)
            or {r['evaluation_index'] for r in rows}!=set(range(len(rows)))
            or {(r['user_id'],r['configuration_id'],r['chunk_size']) for r in rows}!=expected
            or any(r['status']!='completed' or r.get('exception')
                   or r['kind']!=('fit_and_score' if r['chunk_size']==resolved.numerics['score_chunk_size'] else 'score_repeat')
                   or (r['kind']=='score_repeat' and r.get('accepted') is not True)
                   or (r['kind']=='fit_and_score' and r.get('diagnostics',{}).get('status')!='ok') for r in rows)):
        raise StageBlocked('Saved calibration evaluations are incomplete or numerically rejected')
    pairs={(u,e.id) for u in users for e in entries}
    timings=measured['user_measurements']
    if (len(timings)!=len(pairs) or {(r['user_id'],r['configuration_id']) for r in timings}!=pairs
            or any(r.get('diagnostics',{}).get('status')!='ok'
                   or set(r['score_measurements'])!={str(c) for c in chunks}
                   or any(v.get('numerical_accepted') is not True for v in r['score_measurements'].values()) for r in timings)):
        raise StageBlocked('Saved calibration numerical timing coverage is incomplete')
    journal=workload_path.with_suffix('.jsonl'); completed={}; started=set()
    for line in journal.read_text().splitlines():
        event=json.loads(line); verify_record(event)
        if event.get('input_sha256')!=digest: raise StageBlocked('Saved calibration journal inputs changed')
        row=event['attempt']; index=row['evaluation_index']
        if row['status']=='started':
            if index in started: raise StageBlocked('Duplicate saved calibration journal start')
            started.add(index)
        elif row['status']=='completed' and index in started:
            completed[index]=row
        else: raise StageBlocked('Saved calibration journal has incomplete or failed work')
    # Repeated-score write duration is measured after its final durable journal.
    def journal_value(row):
        return {k:v for k,v in row.items() if not (k=='write_seconds' and row['kind']=='score_repeat')}
    if (started!=set(range(len(rows))) or set(completed)!=started
            or any(journal_value(completed[r['evaluation_index']])!=journal_value(r) for r in rows)):
        raise StageBlocked('Saved calibration journal differs from completed workload')
    provenance=dict(parent_attempt_id=parent['attempt_id'],parent_status=parent['status'],
                    parent_publication=reference,workload=workload_reference,journal=file_reference(journal),
                    failed_invocation=file_reference(failed[0][0]),original_correctness=file_reference(proof_path),
                    original_input_sha256=measured['input_sha256'],original_forecast=measured['forecast'],
                    reused_evaluations=len(rows),new_evaluations=0)
    return measured,provenance,workload_path,journal


def _resume_calibration(config,ledger,attempts):
    if not results_first(ledger.snapshot()):
        raise StageBlocked('Saved calibration adoption requires explicit results-first policy')
    with ledger.work('G2',0,0) as aid:
        attempts.append(aid)
        inputs,resolved,_,_,metadata,expected,store=_calibration_context(config,ledger)
        if (config.paths['results']/'stages'/'calibrate.json').exists():
            return _prerequisite(config,'calibrate',ledger,inputs)
        cache_path=resolved.paths['cache']/'embeddings'
        load_embedding_cache(cache_path,expected)
        measured,provenance,workload,journal=_saved_calibration(config,ledger,inputs,resolved,store,metadata)
        forecast=forecast_scale(measured,store,ledger)
        if forecast['status']!='feasible': raise StageBlocked(forecast['stop_reason'])
        policy=forecast['policy']
        resolved=replace(resolved,encoder={**resolved.encoder,'batch_size':policy['embedding_batch_size']},
                         numerics={**resolved.numerics,'score_chunk_size':policy['score_chunk_size']})
        adopted={**measured,'input_sha256':canonical_sha256(inputs),'forecast':forecast,
                 'reused_measurement_provenance':provenance,'execution_policy':ledger.snapshot()['execution_policy']}
        adopted.pop('output_sha256')
        adopted['output_sha256']=canonical_sha256(adopted)
        path=config.paths['results']/f'calibration-resumed-{aid}.json'
        atomic_write_json(path,adopted)
        policy_path=config.paths['results']/'calibration_config.json'
        atomic_write_json(policy_path,config_record(resolved))
        files={'calibration':path,'policy':policy_path,'attempts':workload,'attempt_journal':journal,
               **{f'cache_{name}':cache_path/name for name in ('manifest.json','vectors.npy','item_ids.npy')},
               'original_calibration':Path(provenance['parent_publication']['path']),
               'original_correctness':Path(provenance['original_correctness']['path'])}
        return save_stage(config,'calibrate',inputs,files,attempts,forecast=forecast,
                          reused_measurement_provenance=provenance)


def _smoke(config,ledger,attempts):
    with ledger.work('G0',0,0) as check:
        attempts.append(check)
        inputs,resolved,_,_,_,expected,store=_calibration_context(config,ledger)
        calibration=_prerequisite(config,'calibrate',ledger,inputs)
        cache=load_embedding_cache(resolved.paths['cache']/'embeddings',expected)
        resolved=load_config(config.paths['results']/'calibration_config.json')
        inputs={'calibrate':calibration['output_sha256'],'development_users':8}
        if (config.paths['results']/'stages'/'smoke.json').exists():
            return _prerequisite(config,'smoke',ledger,inputs)
    users=calibration['forecast']['planned_user_ids']['development'][:8]
    if len(users)!=8:
        raise StageBlocked('Smoke requires eight planned development users')
    path=config.paths['results']/f'smoke_attempts-{uuid.uuid4().hex}.json'
    with ledger.work('G3',calibration['forecast']['smoke_seconds'],8*78) as aid:
        attempts.append(aid)
        with threadpool_limits(limits=resolved.timing['blas_threads']):
            measurements=measure_users(store,cache,users,resolved,ledger,numerical_checks=False,output_path=path)
        if measurements['status']!='completed':
            raise StageBlocked('Development smoke has failed fits/scoring; inspect attempt diagnostics')
        artifacts={'attempts':path}
        if path.with_suffix('.jsonl').exists():
            artifacts['attempt_journal']=path.with_suffix('.jsonl')
        result=save_stage(config,'smoke',inputs,artifacts,attempts,development_user_ids=users,
                          actual_evaluations=measurements['actual_evaluations'])
    return result


def _develop(config,ledger,attempts):
    from .development import execute_development
    with ledger.work('G3',0,0) as check:
        attempts.append(check)
        inputs,_,_,_,_,expected,store=_calibration_context(config,ledger)
        calibration=_prerequisite(config,'calibrate',ledger,inputs)
        smoke=_prerequisite(config,'smoke',ledger,{'calibrate':calibration['output_sha256'],'development_users':8})
        resolved=load_config(config.paths['results']/'calibration_config.json')
        cache=load_embedding_cache(resolved.paths['cache']/'embeddings',expected)
        inputs={'calibrate':calibration['output_sha256'],'smoke':smoke['output_sha256']}
        if (config.paths['results']/'stages'/'develop.json').exists():
            return _prerequisite(config,'develop',ledger,inputs)
        measurements=calibration_measurements(config,calibration)
    result=execute_development(resolved,ledger,store,cache,measurements,attempt_ids=attempts)
    files={'development':config.paths['results']/result['artifact_name']}
    for i,batch in enumerate(result['batches']):
        files[f'batch_{i}']=Path(batch['path'])
        files[f'journal_{i}']=Path(batch['journal_path'])
    if result['status']!='completed':
        return result
    with ledger.work('G3',0,0) as aid:
        attempts.append(aid)
        return save_stage(config,'develop',inputs,files,attempts,
                          complete_suite_eligible=result['complete_suite_eligible'],
                          confirmation_outcomes_accessed=False)


def run_stage(stage,config,ledger):
    """Run one ordered gate; incomplete prerequisites are explicit failures."""
    dispatch={'acquire':_acquire,'audit':_audit,'calibrate':_calibrate,
              'resume-calibration':_resume_calibration,'smoke':_smoke,'develop':_develop}
    if stage not in dispatch:
        raise ValueError(f'Unknown implemented stage: {stage}')
    attempts=[]
    try:
        return dispatch[stage](config,ledger,attempts)
    except (OSError,ValueError,KeyError,BudgetExceeded,LedgerBusy) as error:
        status='budget_exceeded' if isinstance(error,BudgetExceeded) else 'blocked'
        result=dict(schema_version=1,stage=stage,status=status,artifacts={},attempt_ids=attempts,stop_reason=str(error))
        # Preserve every failed invocation; never replace completed evidence.
        result['output_sha256']=canonical_sha256(result)
        atomic_write_json(config.paths['results']/'stages'/f'{stage}.failed-{uuid.uuid4().hex}.json',result)
        return result
