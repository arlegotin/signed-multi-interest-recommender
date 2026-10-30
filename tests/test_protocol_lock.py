"""Synthetic end-to-end freeze and confirmation; no acquired research data."""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from test_stage_pipeline import pipeline, prepare


@pytest.fixture
def frozen_study(pipeline, monkeypatch):
    from signed_study import run, stages, workload
    from signed_study.budget import atomic_write_json
    from signed_study.protocol import canonical_sha256
    assert hasattr(run, 'freeze_protocol'), 'immutable protocol freeze is missing'
    from signed_study import protocol_lock
    config, ledger, _, _, _, _, _, store = pipeline
    for episode in store.confirmation[:2]:
        uid = episode.history.user_id
        path = store.paths['confirmation_outcomes']/f'{uid}.npy'
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, np.arange(len(episode.candidate_ids), dtype=np.int8)%2)
        store.audit['outputs']['outcome_sha256']['confirmation'][str(uid)] = hashlib.sha256(path.read_bytes()).hexdigest()
    store.audit['output_sha256'] = canonical_sha256(store.audit['outputs'])
    atomic_write_json(store.paths['audit'], store.audit)
    prepare(pipeline)
    original = stages.forecast_scale
    def small(*args):
        value = original(*args)
        value['planned_user_ids']['development'] = value['planned_user_ids']['development'][:8]
        value['planned_user_ids']['confirmation'] = [e.history.user_id for e in store.confirmation[:2]]
        return value
    monkeypatch.setattr(stages, 'forecast_scale', small)
    assert stages.run_stage('calibrate', config, ledger)['status'] == 'completed'
    monkeypatch.setattr(stages, 'measure_users', workload.measure_users)
    assert stages.run_stage('smoke', config, ledger)['status'] == 'completed'
    development = run.develop(config, ledger)
    assert development['status'] == 'completed'
    monkeypatch.setattr(protocol_lock, 'source_identity', lambda: {'commit':'a'*40, 'tree':'b'*40, 'clean':True, 'content':{'module.py':'c'*64}})
    lock = run.freeze_protocol(config, development, ledger)
    return config, ledger, store, lock


def test_freeze_rejects_mutation_before_outcome_access(frozen_study, monkeypatch):
    from signed_study import protocol_lock, protocol, data
    config, ledger, _, lock = frozen_study
    outcome_reads=[]
    monkeypatch.setattr(data, 'load_confirmation_outcomes', lambda *a: outcome_reads.append(a))
    live = protocol_lock.live_inputs(config, ledger)
    permit = protocol.verify_protocol(lock, live)
    assert permit.lock_hash == hashlib.sha256(lock.read_bytes()).hexdigest()
    assert lock.with_suffix('.sha256').read_text().strip() == permit.lock_hash
    for key in live['identities']:
        changed=copy.deepcopy(live)
        changed['identities'][key] = 'changed'
        with pytest.raises(protocol.ProtocolMismatch):
            protocol.verify_protocol(lock, changed)
    assert outcome_reads == []


def test_freeze_exclusive_canonical_and_ledger_prefix(frozen_study):
    from signed_study import run, protocol_lock, protocol
    config, ledger, _, lock = frozen_study
    before=lock.read_bytes()
    with pytest.raises(FileExistsError):
        run.freeze_protocol(config, {}, ledger)
    assert lock.read_bytes()==before
    with ledger.work('G4',0,0): pass
    live=protocol_lock.live_inputs(config,ledger)
    assert protocol.verify_protocol(lock,live).lock_hash
    live['ledger_events'][0]['event']='changed'
    with pytest.raises(protocol.ProtocolMismatch): protocol.verify_protocol(lock,live)
    lock.write_bytes(before+b'\n')
    with pytest.raises(protocol.ProtocolMismatch): protocol.verify_protocol(lock,protocol_lock.live_inputs(config,ledger))


def test_confirmation_all_slots_scores_failures_and_resume(frozen_study, monkeypatch):
    from signed_study import run, confirmation, data
    config, ledger, store, lock=frozen_study
    original_fit=confirmation.fit_user
    def fail(method,*a,**kw):
        if method=='RBFSVM': raise RuntimeError('synthetic systemic defect')
        return original_fit(method,*a,**kw)
    monkeypatch.setattr(confirmation,'fit_user',fail)
    original_load=data.load_confirmation_outcomes
    reads=[]
    monkeypatch.setattr(data,'load_confirmation_outcomes',lambda s,u,p:(reads.append(u),original_load(s,u,p))[1])
    result=run.confirm(lock,ledger)
    users=[e.history.user_id for e in store.confirmation[:2]]
    assert result['status']=='partial'
    assert result['scheduled_user_ids']==users
    assert reads==users
    assert len(result['comparisons'])==12
    assert result['comparisons']['RBFSVM']['n_missing']==2
    assert result['comparisons']['RBFSVM']['user_ids']==users
    failed=[a for a in result['attempts'] if a['slot']=='RBFSVM']
    assert len(failed)==1 and failed[0]['status']=='failed'
    for uid in users:
        rows={a['slot']:a for a in result['attempts'] if a['user_id']==uid}
        assert rows['SignedMax']['scores']==rows['matched_PositiveMax']['scores']
    before=ledger.snapshot()['attempted_evaluations']
    assert run.confirm(lock,ledger)['output_sha256']==result['output_sha256']
    assert ledger.snapshot()['attempted_evaluations']==before
    assert reads==users


def test_post_exposure_correction_is_persistently_exploratory(frozen_study,monkeypatch):
    from signed_study import run, confirmation, protocol_lock, protocol
    config,ledger,_,lock=frozen_study
    original=run.confirm(lock,ledger)
    original_bytes=Path(original['artifact_path']).read_bytes()
    source=protocol_lock.source_identity()
    changed={**source,'commit':'d'*40,'tree':'e'*40,'content':{'module.py':'f'*64}}
    monkeypatch.setattr(protocol_lock,'source_identity',lambda:changed)
    with pytest.raises(protocol.ProtocolMismatch): run.confirm(lock,ledger)
    events=confirmation.read_events(lock)
    assert any(e['kind']=='invalidated' for e in events)
    from signed_study import stage_evidence
    old_identity=stage_evidence.code_identity()
    monkeypatch.setattr(stage_evidence,'code_identity',lambda:{**old_identity,'fixed.py':'f'*64})
    with pytest.raises(protocol.ProtocolMismatch):
        confirmation.record_implementation_correction(lock,ledger,'Fix pending G1 proof')
    assert stage_evidence.record_correctness(config,ledger)['status']=='completed'
    confirmation.record_implementation_correction(lock,ledger,'Fix canonical computation defect; no retuning')
    before=ledger.snapshot()['repair_cycles']
    result=run.confirm(lock,ledger)
    assert result['interpretation']=='corrected_exploratory'
    assert ledger.snapshot()['repair_cycles']==before+1
    assert Path(original['artifact_path']).read_bytes()==original_bytes
    assert result['lock_sha256']==original['lock_sha256']
    assert result['scheduled_user_ids']==original['scheduled_user_ids']
    assert run.confirm(lock,ledger)['output_sha256']==result['output_sha256']
    assert ledger.snapshot()['repair_cycles']==before+1


def test_rejected_confirmation_never_opens_outcomes(frozen_study,monkeypatch):
    from signed_study import run, data, protocol_lock, protocol
    _,ledger,_,lock=frozen_study
    monkeypatch.setattr(data,'load_confirmation_outcomes',lambda *a:pytest.fail('outcomes opened'))
    monkeypatch.setattr(protocol_lock,'source_identity',lambda:dict(clean=False))
    with pytest.raises(protocol.ProtocolMismatch): run.confirm(lock,ledger)


def test_lock_rejects_missing_coverage_bad_controls_and_infeasible_work(frozen_study):
    from signed_study import protocol_lock, protocol
    config,ledger,store,lock=frozen_study
    selected=protocol_lock.read_lock(lock)[1]['development']
    for change in ('family','control','parameters','forecast','ids','prototypes'):
        bad=copy.deepcopy(selected)
        if change=='family': bad['comparison_schedule'].pop()
        elif change=='control': bad['matched_controls']['matched_MeanNegative']['parameters']['lambda']=999
        elif change=='parameters': bad['comparison_schedule'][0]['fit_family']='RBFSVM'
        elif change=='forecast': bad['selected_forecast']['seconds']['G4']=10000
        elif change=='ids': bad['confirmation_user_ids']=[True]
        else: bad['comparison_schedule'][0]['development_prototype_identities']={}
        with pytest.raises(protocol.ProtocolMismatch):
            protocol_lock._validate_schedule(bad,config,store,ledger)


def test_live_artifact_mutations_fail_before_outcome_arrays(frozen_study,monkeypatch):
    from signed_study import run, data, protocol
    config,ledger,store,lock=frozen_study
    monkeypatch.setattr(data,'load_confirmation_outcomes',lambda *a:pytest.fail('outcome array exposed'))
    paths=[config.paths['results']/'environment.json',config.paths['results']/'calibration_config.json',
           config.paths['cache']/'embeddings'/'vectors.npy',store.paths['episodes'],
           store.paths['confirmation_outcomes']/f'{store.confirmation[0].history.user_id}.npy']
    for path in paths:
        before=path.read_bytes()
        try:
            path.write_bytes(b'changed')
            with pytest.raises(protocol.ProtocolMismatch): run.confirm(lock,ledger)
        finally: path.write_bytes(before)


def test_confirmation_scores_unchanged_by_permuted_labels(frozen_study,monkeypatch):
    from signed_study import run,confirmation,data
    _,ledger,_,lock=frozen_study
    real=data.load_confirmation_outcomes
    original_evaluate=confirmation.evaluate_user
    expected=[]
    def evaluate(scores,labels,ids):
        expected.append((np.asarray(scores).tobytes(),np.asarray(labels).copy(),list(ids)))
        return original_evaluate(scores,labels,ids)
    monkeypatch.setattr(confirmation,'evaluate_user',evaluate)
    monkeypatch.setattr(data,'load_confirmation_outcomes',lambda s,u,p:real(s,u,p)[::-1])
    result=run.confirm(lock,ledger)
    assert result['status']=='completed'
    live,resolved,store,cache,_,_=__import__('signed_study.protocol_lock',fromlist=['_context'])._context(ledger.config,ledger)
    for row,(score_bytes,labels,ids) in zip(result['attempts'],expected):
        assert np.asarray(row['scores'],dtype=np.float64).tobytes()==score_bytes
        # Independently refit from history only, using unpermuted labels nowhere.
        episode=next(e for e in store.confirmation if e.history.user_id==row['user_id'])
        model=confirmation.fit_user(row['fit_family'],episode.history,cache,row['parameters'],config=resolved)
        assert confirmation.score_user(model,ids,cache,chunk_size=resolved.numerics['score_chunk_size']).tobytes()==score_bytes


def test_freeze_refuses_dirty_source_even_with_valid_saved_development(frozen_study,monkeypatch):
    from signed_study import run,protocol_lock,protocol
    config,ledger,_,lock=frozen_study
    saved=protocol_lock.read_lock(lock)[1]['development']
    lock.unlink(); lock.with_suffix('.sha256').unlink()
    monkeypatch.setattr(protocol_lock,'source_identity',lambda:dict(clean=False))
    with pytest.raises(protocol.ProtocolMismatch): run.freeze_protocol(config,saved,ledger)


def test_exposure_history_cannot_be_removed_and_restarted(frozen_study):
    from signed_study import run,confirmation,protocol
    _,ledger,_,lock=frozen_study
    run.confirm(lock,ledger)
    confirmation._events_path(lock).unlink()
    with pytest.raises(protocol.ProtocolMismatch): run.confirm(lock,ledger)


def test_lock_carries_selections_without_copying_raw_development_scores(frozen_study):
    from signed_study.protocol_lock import read_lock
    *_,lock=frozen_study
    payload=read_lock(lock)[1]
    assert 'attempts' not in payload['development']
    assert 'pilot_attempts' not in payload['development']
    assert len(payload['development']['configurations'])>=78
    assert payload['identities']['development_sha256']


def test_diagnostic_schedule_and_evaluation_forecast_are_exact(frozen_study):
    from signed_study import protocol_lock,protocol
    config,ledger,store,lock=frozen_study
    selected=protocol_lock.read_lock(lock)[1]['development']
    for key,value in [('evaluations',{'G4':26,'G5':0}),('numerical_user_ids',[999999]),('timing_user_ids',[999999])]:
        bad=copy.deepcopy(selected)
        bad['selected_forecast'][key]=value
        with pytest.raises(protocol.ProtocolMismatch): protocol_lock._validate_schedule(bad,config,store,ledger)


def test_interruption_keeps_started_attempt_and_whole_denominator(frozen_study,monkeypatch):
    from signed_study import run,confirmation
    _,ledger,_,lock=frozen_study
    def interrupted(*a,**kw): raise KeyboardInterrupt('synthetic interruption')
    monkeypatch.setattr(confirmation,'fit_user',interrupted)
    with pytest.raises(KeyboardInterrupt): run.confirm(lock,ledger)
    before=ledger.snapshot()['attempted_evaluations']
    result=run.confirm(lock,ledger)
    assert result['status']=='partial'
    assert len(result['attempts'])==1
    assert result['attempts'][0]['status']=='started'
    assert result['primary']['n_missing']==2
    assert result['primary']['lower_bounds']==[-1.,-1.]
    assert result['primary']['upper_bounds']==[1.,1.]
    assert ledger.snapshot()['attempted_evaluations']==before


def test_changed_lock_and_sidecar_disagree_with_permanent_freeze_evidence(frozen_study):
    from signed_study import protocol_lock,protocol
    config,ledger,_,lock=frozen_study
    digest,payload=protocol_lock.read_lock(lock)
    payload['config']['delta']=.005
    raw=protocol_lock.canonical_bytes(payload)
    lock.write_bytes(raw)
    lock.with_suffix('.sha256').write_text(hashlib.sha256(raw).hexdigest()+'\n')
    with pytest.raises(protocol.ProtocolMismatch):
        protocol.verify_protocol(lock,protocol_lock.live_inputs(config,ledger))


def test_diagnostics_restore_exact_fitted_geometry_without_refitting(frozen_study,monkeypatch):
    from signed_study import run,confirmation,protocol_lock
    _,ledger,_,lock=frozen_study
    result=run.confirm(lock,ledger)
    assert hasattr(confirmation,'load_frozen_geometry'), 'original fitted diagnostic state is missing'
    _,_,store,cache,_,_=protocol_lock._context(ledger.config,ledger)
    def forbidden(*a,**kw): pytest.fail('diagnostic reconstruction refit')
    monkeypatch.setattr(confirmation,'fit_user',forbidden)
    for row in result['attempts']:
        if row['fit_family'] in ('LinearLogistic','RBFSVM'): continue
        state=row['fitted_state']
        assert state['lock_sha256']==result['lock_sha256']
        model=confirmation.load_frozen_geometry(state,cache,result['lock_sha256'])
        assert model.state['P'].dtype==np.float64
        scores=confirmation.score_user(model,row['candidate_ids'],cache)
        assert scores.tobytes()==np.asarray(row['scores'],dtype=np.float64).tobytes()
    state=next(r['fitted_state'] for r in result['attempts'] if 'fitted_state' in r)
    path=Path(state['positive_vectors']['path']); path.write_bytes(b'changed')
    with pytest.raises(ValueError): confirmation.load_frozen_geometry(state,cache,result['lock_sha256'])


def test_invalidated_interrupted_run_cannot_regain_confirmatory_status(frozen_study,monkeypatch):
    from signed_study import run,confirmation
    _,ledger,_,lock=frozen_study
    def interrupted(*a,**kw): raise KeyboardInterrupt('synthetic interruption')
    monkeypatch.setattr(confirmation,'fit_user',interrupted)
    with pytest.raises(KeyboardInterrupt): run.confirm(lock,ledger)
    confirmation._invalidate(lock,'Implementation defect discovered after exposure')
    assert run.confirm(lock,ledger)['interpretation']=='invalid_confirmatory'


def test_completed_arrays_do_not_hide_failed_final_resource_check(frozen_study,monkeypatch):
    from signed_study import run
    from signed_study.budget import BudgetExceeded
    _,ledger,_,lock=frozen_study
    finish=ledger.finish
    def final_limit(aid,status,measurements,outputs):
        if 'confirmation' in outputs:
            finish(aid,'budget_exceeded',measurements,outputs)
            raise BudgetExceeded('synthetic final resource check')
        return finish(aid,status,measurements,outputs)
    monkeypatch.setattr(ledger,'finish',final_limit)
    assert run.confirm(lock,ledger)['status']=='analysis_pending'
    monkeypatch.setattr(ledger,'finish',finish)
    assert run.confirm(lock,ledger)['status']=='budget_exceeded'
