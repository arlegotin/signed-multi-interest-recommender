"""Ordered acquisition, audit, calibration and development-only smoke gates."""
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import time
import uuid

from threadpoolctl import threadpool_limits, threadpool_info

from .budget import BudgetExceeded, LedgerBusy, atomic_write_json
from .calibration import forecast_scale, select_calibration_users
from .data import (SOURCES, Movie, _acquire_archive as acquire_archive, load_dataset,
                   make_episodes, load_episode_store)
from .embed import (resolve_encoder, expected_embedding_manifest, load_encoder,
                    encode_probe, check_repeatability, encode_items, load_embedding_cache)
from .protocol import load_config, canonical_sha256
from .stage_evidence import (config_record, file_reference, read_stage, save_stage,
                             verify_record, verify_correctness, environment_pins,
                             write_environment_lock, verify_environment_lock)
from .workload import measure_users, calibration_entries


class StageBlocked(ValueError):
    pass


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
        measurements=_read(config,'calibration.json')
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
    dispatch={'acquire':_acquire,'audit':_audit,'calibrate':_calibrate,'smoke':_smoke,'develop':_develop}
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
