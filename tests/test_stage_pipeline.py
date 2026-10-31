"""Real persisted stage prerequisites/cache/firewall, injected external workloads."""
import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest

from signed_study.budget import Ledger, atomic_write_json
from signed_study.data import Dataset, EpisodeStore, Movie, episode_store_digest
from signed_study.protocol import canonical_sha256
from test_calibration import cohort


@pytest.fixture
def pipeline(embedding_inputs, monkeypatch, tmp_path):
    from signed_study import run
    assert hasattr(run, 'run_stage'), 'ordered stage dispatcher is missing'
    from signed_study import stages
    from signed_study import stage_evidence as evidence
    config = replace(embedding_inputs['config'], model_revision=None)
    clock = [0.]
    resources = dict(installed_ram_bytes=32*1024**3, available_memory_bytes=16*1024**3,
                     rss_bytes=100*1024**2, project_disk_bytes=1000, free_disk_bytes=20*1024**3)
    ledger = Ledger(config.paths['results']/'budget_ledger.jsonl', config,
                    clock=lambda: clock[0], wall_clock=lambda: 1000000+clock[0],
                    sampler=lambda: dict(resources), project_root=tmp_path)
    metadata = {i: Movie(i, f'Movie {i}', ('Drama',)) for i in range(256)}
    manifest = {**embedding_inputs['dataset_manifest'], 'status': 'validated', 'access_fallback': False}
    manifest['archive_sha256'] = stages.SOURCES['ml-1m'].sha256
    raw=config.paths['data']/'raw'/'ml-1m.zip'
    raw.parent.mkdir(parents=True,exist_ok=True)
    raw.write_bytes(b'injected acquisition bytes; parsing boundary supplies synthetic records')
    dataset = Dataset((), metadata, manifest)
    store = cohort()
    root = config.paths['data']/'derived'/'fixture'
    paths = {'audit': config.paths['results']/'cohort_audit.json', 'episodes': root/'episodes.json',
             **{role+'_outcomes': root/'outcomes'/role for role in ('development','confirmation')}}
    audit = {'schema_version': 1, 'dataset': 'ml-1m', 'status': 'validated', 'input_sha256': 'a'*64}
    store = EpisodeStore(store.development, store.confirmation, audit, paths)
    audit['episodes_sha256'] = episode_store_digest(store)
    records = {'schema_version': 1, **{role: [asdict(e) for e in sorted(getattr(store,role),key=lambda e:e.history.user_id)]
                                     for role in ('development','confirmation')}}
    atomic_write_json(paths['episodes'], {**records,'input_sha256': 'a'*64,'output_sha256':audit['episodes_sha256']})
    outcomes = {'development': {}, 'confirmation': {}}
    for e in store.development:
        path = paths['development_outcomes']/f'{e.history.user_id}.npy'
        path.parent.mkdir(parents=True,exist_ok=True)
        np.save(path,np.arange(len(e.candidate_ids),dtype=np.int8)%2)
        outcomes['development'][str(e.history.user_id)] = hashlib.sha256(path.read_bytes()).hexdigest()
    audit['outputs'] = {'episodes_file_sha256':hashlib.sha256(paths['episodes'].read_bytes()).hexdigest(),
                        'outcome_sha256':outcomes}
    audit['output_sha256'] = canonical_sha256(audit['outputs'])
    atomic_write_json(paths['audit'],audit)
    import importlib.metadata
    environment = {'schema_version':1,'target_compatible':True,'blockers':[],
                   'packages':{n:importlib.metadata.version(n) for n in run._PACKAGES}}
    environment['output_sha256'] = canonical_sha256(environment)
    atomic_write_json(config.paths['results']/'environment.json',environment)
    monkeypatch.setattr(stages, 'acquire_archive', lambda c: None)
    monkeypatch.setattr(stages, 'load_dataset', lambda c: dataset)
    monkeypatch.setattr(stages, 'resolve_encoder', lambda c: embedding_inputs['encoder_provenance'])
    monkeypatch.setattr(stages, 'make_episodes', lambda d,c: store)
    events = []

    class Encoder:
        def token_lengths(self,texts): return [8]*len(texts)
        def encode(self,texts,*,batch_size):
            clock[0] += .01
            rows = np.zeros((len(texts),384),dtype=np.float32)
            rows[:,0] = 1.
            return rows
    monkeypatch.setattr(stages,'load_encoder',lambda *a: Encoder())
    real_probe, real_encode = stages.encode_probe, stages.encode_items
    def probe(*a,**kwargs):
        events.append('probe_128' if len(kwargs['item_ids'])==128 else 'repeat_32')
        return real_probe(*a,**kwargs)
    def encode(*a,**kwargs):
        events.append('encode_catalog')
        return real_encode(*a,**kwargs)
    monkeypatch.setattr(stages,'encode_probe',probe)
    monkeypatch.setattr(stages,'encode_items',encode)
    original_admit = ledger.admit
    def admit(stage,projected_seconds,evaluations):
        result = original_admit(stage,projected_seconds,evaluations)
        if stage=='G2' and (events and events[-1]=='repeat_32'):
            events.append('admit_full_cache')
        return result
    monkeypatch.setattr(ledger,'admit',admit)
    from test_calibration import measured
    def measured_users(store,cache,users,config,ledger,*,numerical_checks,output_path):
        assert cache.item_ids.tolist() == sorted(metadata)
        assert len(users)==32
        events.append('fit_32_users')
        clock[0]+=.1
        value=measured()
        value.update(status='completed',attempts=[],actual_evaluations=0,repair_decisions=[])
        atomic_write_json(output_path,{'schema_version':1,'input_sha256':canonical_sha256({'users':users}),
                                       'output_sha256':canonical_sha256(value),'measurements':value})
        return value
    monkeypatch.setattr(stages,'measure_users',measured_users)
    # Fake only subprocess execution; real G1 command, proof, and ledger validation remain.
    monkeypatch.setattr(evidence,'correctness_command',lambda: [str(Path(__import__('sys').executable)), '-c','pass'])
    yield config,ledger,clock,resources,events,stages,evidence,store
    ledger.close()


def prepare(pipeline):
    config,ledger,_,_,_,stages,evidence,_=pipeline
    assert stages.run_stage('acquire',config,ledger)['status']=='completed'
    assert stages.run_stage('audit',config,ledger)['status']=='completed'
    assert evidence.record_correctness(config,ledger)['status']=='completed'


def test_calibration_orders_probe_cache_and_user_fits(pipeline):
    config,ledger,_,_,events,stages,_,_=pipeline
    prepare(pipeline)
    result=stages.run_stage('calibrate',config,ledger)
    assert result['status']=='completed',result
    assert events==['probe_128','repeat_32','admit_full_cache','encode_catalog','fit_32_users']
    assert result['forecast']['safety_multiplier']==1.5
    assert result['forecast']['scale']==[418,1776]
    before=ledger.snapshot()['attempted_evaluations']
    again=stages.run_stage('calibrate',config,ledger)
    assert again['status']=='completed',again
    assert len(events)==5
    assert ledger.snapshot()['attempted_evaluations']==before


def test_infeasible_calibration_publishes_failed_measurements(pipeline,monkeypatch):
    from signed_study.stage_evidence import file_reference
    config,ledger,_,_,_,stages,_,_=pipeline
    prepare(pipeline)
    original=stages.measure_users
    def measured(*a,**kw):
        result=original(*a,**kw)
        result.update(actual_evaluations=1,attempts=[dict(evaluation_index=0,status='completed')])
        return result
    monkeypatch.setattr(stages,'measure_users',measured)
    monkeypatch.setattr(stages,'forecast_scale',lambda *a:dict(status='infeasible',tested_scales=[[256,1000]],stop_reason='smallest scale exceeds caps'))
    result=stages.run_stage('calibrate',config,ledger)
    assert result['status']=='blocked' and result['stop_reason']=='smallest scale exceeds caps'
    measurements=file_reference(config.paths['results']/'calibration.json')
    final=ledger.snapshot()['attempts'][result['attempt_ids'][-1]]
    assert final['stage']=='G2' and final['status']=='failed'
    assert final['outputs']['calibration']==measurements
    assert final['measurements']['exception']=='smallest scale exceeds caps'
    from signed_study.report import render_report
    assert render_report(config.paths['results'])['outcome']=='practical infeasibility'


@pytest.mark.parametrize('fault',['g1','bad_probe','cache','memory','reserve','provenance'])
def test_calibration_refuses_invalid_prerequisites(pipeline,monkeypatch,fault):
    config,ledger,clock,resources,events,stages,_,_=pipeline
    prepare(pipeline)
    if fault=='g1': (config.paths['results']/'correctness.json').unlink()
    elif fault=='bad_probe':
        original=stages.encode_probe
        def bad(*a,**kw):
            result=original(*a,**kw)
            if len(kw['item_ids'])==32:
                from signed_study.embed import EmbeddingProbe
                return EmbeddingProbe(result.item_ids,result.raw_vectors*2,result.manifest)
            return result
        monkeypatch.setattr(stages,'encode_probe',bad)
    elif fault=='cache':
        monkeypatch.setattr(stages,'encode_items',lambda *a,**kw:None)
    elif fault=='memory': resources['available_memory_bytes']=1
    elif fault=='reserve':
        # Charge G6 legally, leaving less than the protected reserve.
        with ledger.work('G6',0,0): clock[0]+=1799
        # The remaining capacity is exhausted through existing persistent records.
        with ledger.path.open('a') as f:
            f.write(json.dumps({'event':'overhead','stage':'G4','elapsed_seconds':10800})+'\n')
    else: (config.paths['results']/'metadata.json').write_text('{}')
    result=stages.run_stage('calibrate',config,ledger)
    assert result['status'] in ('blocked','failed','budget_exceeded','repair_required'),result
    assert 'fit_32_users' not in events
    assert result['stop_reason']


def test_smoke_requires_calibration_and_only_eight_development_users(pipeline,monkeypatch):
    config,ledger,_,_,_,stages,_,store=pipeline
    prepare(pipeline)
    assert stages.run_stage('smoke',config,ledger)['status']=='blocked'
    assert stages.run_stage('calibrate',config,ledger)['status']=='completed'
    seen=[]
    def smoke(store,cache,users,config,ledger,**kwargs):
        seen.extend(users)
        assert all(uid in {e.history.user_id for e in store.development} for uid in users)
        value={'status':'completed','actual_evaluations':0,'attempts':[]}
        atomic_write_json(kwargs['output_path'],value)
        return value
    monkeypatch.setattr(stages,'measure_users',smoke)
    assert stages.run_stage('smoke',config,ledger)['status']=='completed'
    assert len(seen)==8
    # Resume validates the complete cache before accepting old smoke evidence.
    (config.paths['cache']/'embeddings'/'vectors.npy').unlink()
    assert stages.run_stage('smoke',config,ledger)['status']=='blocked'


def test_correctness_proof_requires_current_inputs_and_success(pipeline,monkeypatch):
    config,ledger,_,_,_,stages,evidence,_=pipeline
    prepare(pipeline)
    monkeypatch.setattr(evidence,'code_identity',lambda: {'changed.py':'b'*64})
    assert stages.run_stage('calibrate',config,ledger)['status']=='blocked'


def test_cli_dispatch_and_exact_smoke_count(pipeline,monkeypatch,capsys):
    from signed_study import __main__ as cli
    from signed_study import budget,protocol
    config,ledger,_,_,_,_,_,_=pipeline
    monkeypatch.setattr(protocol,'load_config',lambda p:config)
    monkeypatch.setattr(budget,'Ledger',lambda *a,**kw:ledger)
    assert cli.main(['acquire'])==0
    assert json.loads(capsys.readouterr().out)['status']=='completed'
    with pytest.raises(SystemExit) as error:
        cli.main(['smoke','--development-users','9'])
    assert error.value.code==2


def test_acquisition_pins_environment_and_reuses_immutable_model(pipeline,monkeypatch):
    config,ledger,_,_,_,stages,_,_=pipeline
    result=stages.run_stage('acquire',config,ledger)
    assert result['status']=='completed',result
    assert Path('requirements.lock.txt').is_file()
    pins=Path('requirements.lock.txt').read_text()
    assert f'numpy=={np.__version__}' in pins
    monkeypatch.setattr(stages,'resolve_encoder',lambda c:pytest.fail('must reuse pinned model'))
    assert stages.run_stage('acquire',config,ledger)['status']=='completed'
    Path('requirements.lock.txt').write_text('changed==1\n')
    assert stages.run_stage('audit',config,ledger)['status']=='blocked'


def test_recorded_global_policy_reuses_acquisition_and_applies_to_every_fit(pipeline,monkeypatch):
    config,ledger,_,_,_,stages,evidence,_=pipeline
    prepare(pipeline)
    repaired=replace(config,prototype={**config.prototype,'max_updates':100},
                     pre_freeze_repairs=({'kind':'prototype_iterations','stage':'G2','reason':'trace inspection'},))
    evidence.record_correctness(repaired,ledger)
    original=stages.measure_users
    def check(*a,**kw):
        assert a[3].prototype['max_updates']==100
        assert a[3].pre_freeze_repairs[0]['kind']=='prototype_iterations'
        return original(*a,**kw)
    monkeypatch.setattr(stages,'measure_users',check)
    monkeypatch.setattr(stages,'resolve_encoder',lambda c:pytest.fail('global repair must retain model revision'))
    result=stages.run_stage('calibrate',repaired,ledger)
    assert result['status']=='completed',result


def test_failed_correctness_never_opens_g2(pipeline,monkeypatch):
    config,ledger,_,_,events,stages,evidence,_=pipeline
    prepare(pipeline)
    monkeypatch.setattr(evidence,'correctness_command',lambda:[__import__('sys').executable,'-c','raise SystemExit(1)'])
    assert evidence.record_correctness(config,ledger)['status']=='failed'
    assert stages.run_stage('calibrate',config,ledger)['status']=='blocked'
    assert events==[]


@pytest.mark.parametrize('damage',['missing','corrupt'])
def test_audit_refuses_missing_acquired_archive_without_reacquiring(pipeline,monkeypatch,damage):
    config,ledger,_,_,_,stages,_,_=pipeline
    assert stages.run_stage('acquire',config,ledger)['status']=='completed'
    raw=config.paths['data']/'raw'/'ml-1m.zip'
    if damage=='missing': raw.unlink()
    else: raw.write_bytes(b'corrupted archive')
    monkeypatch.setattr(stages,'load_dataset',lambda c:pytest.fail('must stop before any possible acquisition'))
    result=stages.run_stage('audit',config,ledger)
    assert result['status']=='blocked',result
