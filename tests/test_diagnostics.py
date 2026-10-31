"""Bounded synthetic diagnostics; no study data or model downloads."""
from dataclasses import replace

import numpy as np
import pytest

from test_protocol_lock import frozen_study
from test_stage_pipeline import pipeline


def test_stratum_boundaries_and_contamination_parity():
    from signed_study import metrics, run
    from signed_study.data import History
    assert hasattr(metrics, 'history_statistics'), 'shared history diagnostics missing'
    assert hasattr(metrics, 'assign_stratum'), 'frozen stratum assignment missing'
    assert hasattr(run, 'contaminate_history'), 'bounded contamination missing'
    assert metrics.assign_stratum(.2,.3,{'H':.2,'O':.3})=='high_H_high_O'
    assert metrics.assign_stratum(.19,.29,{'H':.2,'O':.3})=='low_H_low_O'
    history=History(7,50,'confirmation',tuple(range(25)),tuple(range(25,45)))
    original_ids=set(history.positive_ids+history.negative_ids)
    for seed in (1101,1102,1103):
        corrupted=run.contaminate_history(history,seed)
        assert corrupted==run.contaminate_history(history,seed)
        assert set(corrupted.positive_ids)|set(corrupted.negative_ids)==original_ids
        assert len(corrupted.positive_ids)==25 and len(corrupted.negative_ids)==20
        assert len(set(history.positive_ids)-set(corrupted.positive_ids))==2
        assert len(set(history.negative_ids)-set(corrupted.negative_ids))==2
        assert (corrupted.user_id,corrupted.cutoff,corrupted.role)==(7,50,'confirmation')
    class Cache:
        def rows(self, ids): return np.array([[1.,0.] if i<25 else [0.,1.] for i in ids])
    assert metrics.history_statistics(history,Cache())=={'H':0.,'O':0.,'negative_mean_norm':1.,'negative_count':20}


@pytest.mark.parametrize('method,parameters', [('SignedMax',{'lambda':.25}),('MeanNegative',{'lambda':.25}),
    ('PrototypeRatio',{}),('ExemplarRatio',{}),('AllExemplarMargin',{'lambda':.25})])
def test_float32_uses_fixed_cast_geometry_and_direct_float32_distances(method,parameters,monkeypatch):
    from signed_study import diagnostics
    from signed_study.models import UserModel
    P=np.array([[1.,0.],[0.,1.]])
    N=np.array([[-1.,0.],[.6,.8]])
    X=np.array([[1.,0.],[.8,.6],[0.,-1.]])
    model=UserModel(method,parameters,(),{'P':P.copy(),'N':N.copy()},{'status':'ok'})
    def forbidden(*a,**kw): pytest.fail('float32 diagnostic must not use canonical scorer or refit')
    monkeypatch.setattr(diagnostics,'fit_user',forbidden)
    monkeypatch.setattr(diagnostics,'score_user',forbidden)
    values=diagnostics.float32_geometry_scores(model,X,chunk_size=1)
    p,n,x=(a.astype(np.float32) for a in (P,N,X))
    if 'Ratio' in method:
        dp=np.min(np.sqrt(np.sum((x[:,None,:]-p)**2,axis=2,dtype=np.float32)),axis=1)
        dn=np.min(np.sqrt(np.sum((x[:,None,:]-n)**2,axis=2,dtype=np.float32)),axis=1)
        expected=dn/(dp+dn)
    else:
        negative=x@n.mean(axis=0) if method=='MeanNegative' else np.max(x@n.T,axis=1)
        expected=np.max(x@p.T,axis=1)-np.float32(.25)*negative
    assert values.dtype==np.float32
    np.testing.assert_array_equal(values,expected)
    np.testing.assert_array_equal(model.state['P'],P)


def test_timing_full_catalog_rotation_and_accounting(local_config):
    from signed_study import diagnostics
    from signed_study.data import History, Episode
    from signed_study.embed import EmbeddingCache
    ids=np.arange(17); vectors=np.zeros((17,384));vectors[:,0]=1
    cache=EmbeddingCache(ids,vectors,{})
    episodes={u:Episode(History(u,1,'development',(0,1),(2,3)),(4,5)) for u in range(32)}
    schedule=[dict(name=n,fit_family='SignedMax',parameters={'K':1,'lambda':0},id=n) for n in ('a','b','c')]
    calls=[]
    class Ledger:
        def checkpoint(self): return {'rss_bytes':123}
    rows=diagnostics.timing_measurements(episodes,list(range(32)),schedule,cache,local_config,Ledger(),calls.append)
    assert len(rows)==32*3*6
    assert sum(not r['warmup'] for r in rows)==32*3*5
    assert [r['slot'] for r in rows[:9]]==['a','b','c','b','c','a','c','a','b']
    assert all(r['candidate_ids']==ids.tolist() for r in rows)
    assert len(calls)==2*len(rows) and all(r['status']=='completed' for r in rows)
    assert all(r['rank_seconds']>=r['score_seconds']>=0 and r['fit_seconds']>=0 for r in rows)


def test_diagnose_saved_result_without_confirmation_resume_or_extra_label_access(frozen_study,monkeypatch):
    from signed_study import run,diagnostics,confirmation,data,protocol_lock
    _,ledger,store,lock=frozen_study
    saved=run.confirm(lock,ledger)
    before=ledger.snapshot()['attempted_evaluations']
    frozen=protocol_lock.read_lock(lock)[1]['development']['selected_forecast']
    assert frozen['contamination']['user_ids']==saved['scheduled_user_ids'][:200]
    assert frozen['contamination']['seeds']==[1101,1102,1103]
    monkeypatch.setattr(confirmation,'confirm',lambda *a:pytest.fail('diagnostics called confirm'))
    original_load=data.load_confirmation_outcomes
    reads=[]
    monkeypatch.setattr(data,'load_confirmation_outcomes',lambda s,u,p:(reads.append(u),original_load(s,u,p))[1])
    monkeypatch.setattr(diagnostics,'contamination_applicable',lambda *a:False)
    context=protocol_lock._context
    def charged_context(*args,**kwargs):
        assert ledger._active is not None, 'diagnostic live context must be admitted'
        assert ledger.snapshot()['attempts'][ledger._active]['stage']=='G5'
        return context(*args,**kwargs)
    monkeypatch.setattr(protocol_lock,'_context',charged_context)
    result=run.diagnose(lock,ledger)
    assert reads==frozen['numerical_user_ids']
    assert result['contamination']['status']=='skipped'
    assert 'not applicable' in result['contamination']['reason']
    assert result['interpretation']=='secondary_unadjusted'
    assert len(result['strata']['H_O'])==4 and len(result['strata']['negative_count'])==3
    assert any(r['status']=='unavailable' for r in result['strata']['H_O'].values())
    assert all(r['imprecise'] for r in result['strata']['H_O'].values() if r['n_users'])
    assert all(r['dtype']=='float32' for r in result['numerical'] if r['status']=='completed')
    assert ledger.snapshot()['attempted_evaluations']-before==frozen['evaluations']['G5']
    after=ledger.snapshot()['attempted_evaluations']
    second=run.diagnose(lock,ledger)
    assert second['numerical']==result['numerical'] and second['timing']==result['timing']
    assert second['read_verification_attempt_id'] in second['attempt_ids']
    assert ledger.snapshot()['attempts'][second['read_verification_attempt_id']]['evaluations']==0
    assert second['original_output_sha256']==result['output_sha256']
    assert ledger.snapshot()['attempted_evaluations']==after


def test_contamination_reserve_and_applicability_gate(local_config):
    from signed_study import diagnostics
    caps=local_config.resources
    state={'experimental_seconds':caps['experimental_seconds']-caps['report_reserve_seconds']-1199,
           'stage_seconds':{'G5':0},'stage_limits':{'G5':1200},'attempted_evaluations':0}
    plan={'seconds':1.,'evaluations':12}
    assert '20 minutes' in diagnostics.contamination_skip_reason(True,state,plan,local_config)
    assert 'not applicable' in diagnostics.contamination_skip_reason(False,state,plan,local_config)


def test_contamination_uses_identical_cached_swaps_frozen_parameters_and_original_labels(local_config):
    from signed_study import diagnostics
    from signed_study.data import History,Episode
    from signed_study.embed import EmbeddingCache
    from signed_study.models import model_configuration
    rng=np.random.default_rng(17)
    vectors=rng.normal(size=(36,384));vectors/=np.linalg.norm(vectors,axis=1)[:,None]
    cache=EmbeddingCache(np.arange(36),vectors,{})
    history=History(1,25,'confirmation',tuple(range(15)),tuple(range(15,25)))
    episode=Episode(history,tuple(range(25,36)))
    labels=np.arange(11,dtype=np.int8)%2
    original=labels.copy(); calls=[]
    params={'SignedMax':{'K':2,'lambda':.25},'MeanNegative':{'K':2,'lambda':.5},'LinearLogistic':{'C':1},'RBFSVM':{'C':1,'g':1}}
    schedule=[dict(name=n,fit_family=n,parameters=p,id=model_configuration(n,p).id) for n,p in params.items()]
    plan=dict(user_ids=[1],seeds=[1101,1102,1103],schedule=schedule)
    class Ledger:
        def checkpoint(self): pass
    from unittest.mock import patch
    with patch.object(diagnostics.data,'load_confirmation_outcomes',return_value=labels):
        result=diagnostics.contamination_measurements(plan,{1:episode},cache,local_config,None,None,Ledger(),calls.append)
    assert len(result['attempts'])==12 and len(calls)==24
    assert all(r['status']=='completed' and r['parameters']==params[r['slot']] for r in result['attempts'])
    for seed in plan['seeds']:
        histories={(tuple(r['positive_ids']),tuple(r['negative_ids'])) for r in result['attempts'] if r['seed']==seed}
        assert len(histories)==1
    np.testing.assert_array_equal(labels,original)
    assert episode.history==history


def test_applicability_uses_material_auc_and_precise_subgroups_only(local_config):
    from signed_study import diagnostics
    saved={'primary':{'low':-.01},'comparisons':{'MeanNegative':{'low':-.01},'matched_MeanNegative':{'low':-.01}}}
    high={'n_users':99,'comparisons':{'matched_MeanNegative':{'low':.02}}}
    strata={'H_O':{'high_H_high_O':high}}
    plan=dict(applicability=diagnostics.APPLICABILITY,minimum_subgroup_users=100,numerical_auc_threshold=.01)
    numerical=[dict(slot='SignedMax',status='completed',absolute_auc_change=.001,rank_positions_changed=12,max_absolute_score_difference=.0001)]
    assert not diagnostics.contamination_applicable(saved,strata,numerical,plan,local_config)
    numerical[0]['absolute_auc_change']=.01
    assert diagnostics.contamination_applicable(saved,strata,numerical,plan,local_config)
    numerical[0]['absolute_auc_change']=0
    high['n_users']=100
    assert diagnostics.contamination_applicable(saved,strata,numerical,plan,local_config)
    high['n_users']=0
    saved['comparisons']['MeanNegative']['low']=.001
    assert diagnostics.contamination_applicable(saved,strata,numerical,plan,local_config)


def test_missing_original_geometry_stays_unavailable_without_refit(local_config,monkeypatch):
    from signed_study import diagnostics
    monkeypatch.setattr(diagnostics,'fit_user',lambda *a,**kw:pytest.fail('missing geometry was refitted'))
    class Ledger:
        def checkpoint(self): pass
    calls=[]
    rows=diagnostics.numerical_measurements([9],[{'geometric':True,'name':'SignedMax','id':'frozen-id'}],[],None,'lock',local_config,Ledger(),calls.append,{})
    assert rows[0]['status']=='unavailable' and rows[0]['refitted'] is False
    assert len(calls)==2


def test_saved_confirmation_reader_never_resumes_or_admits(frozen_study,monkeypatch):
    from signed_study import confirmation,run,protocol
    _,ledger,_,lock=frozen_study
    before=ledger.snapshot()['events']
    with pytest.raises(protocol.ProtocolMismatch,match='No finished saved'):
        confirmation.load_saved_confirmation(lock,ledger)
    assert ledger.snapshot()['events']==before
    saved=run.confirm(lock,ledger)
    before=ledger.snapshot()['events']
    monkeypatch.setattr(ledger,'admit',lambda *a,**kw:pytest.fail('read-only saved-result access admitted work'))
    monkeypatch.setattr(confirmation,'confirm',lambda *a,**kw:pytest.fail('saved-result reader called confirm'))
    assert confirmation.load_saved_confirmation(lock,ledger)==saved
    assert ledger.snapshot()['events']==before


def test_diagnostic_final_resource_failure_qualifies_saved_artifact(frozen_study,monkeypatch):
    from signed_study import run,diagnostics
    _,ledger,_,lock=frozen_study
    run.confirm(lock,ledger)
    monkeypatch.setattr(diagnostics,'contamination_applicable',lambda *a:False)
    finish=ledger.finish
    def fail_final(aid,status,measurements,outputs):
        if 'diagnostics' in outputs:
            sampler=ledger.sampler
            monkeypatch.setattr(ledger,'sampler',lambda:{**sampler(),'rss_bytes':9*1024**3})
        return finish(aid,status,measurements,outputs)
    monkeypatch.setattr(ledger,'finish',fail_final)
    result=run.diagnose(lock,ledger)
    assert result['status']=='partial' and result['accounting_status']==['budget_exceeded']
    assert ledger.snapshot()['attempts'][result['attempt_ids'][-1]]['outputs']['diagnostics']['path']==result['artifact_path']
    # Restore resource availability for a new charged verification read.
    monkeypatch.setattr(ledger,'sampler',lambda:dict(installed_ram_bytes=32*1024**3,available_memory_bytes=16*1024**3,rss_bytes=100*1024**2,project_disk_bytes=1000,free_disk_bytes=20*1024**3))
    second=run.diagnose(lock,ledger)
    assert second['status']==result['status'] and second['accounting_status']==result['accounting_status']
    assert second['original_output_sha256']==result['original_output_sha256']


def test_saved_diagnostics_bind_ledger_and_corrected_confirmation_identity(frozen_study,monkeypatch):
    import json
    from pathlib import Path
    from signed_study import run,diagnostics,confirmation,protocol_lock,protocol,stage_evidence
    config,ledger,_,lock=frozen_study
    original=run.confirm(lock,ledger)
    monkeypatch.setattr(diagnostics,'contamination_applicable',lambda *a:False)
    diagnostic=run.diagnose(lock,ledger)
    path=Path(diagnostic['artifact_path']); original_bytes=path.read_bytes()
    changed=json.loads(original_bytes)
    changed['status']='forged_but_self_checksummed'
    changed.pop('output_sha256')
    changed['output_sha256']=protocol.canonical_sha256(changed)
    path.write_bytes(protocol_lock.canonical_bytes(changed))
    with pytest.raises(protocol.ProtocolMismatch,match='diagnostic|Diagnostic'):
        run.diagnose(lock,ledger)
    path.write_bytes(original_bytes)
    source=protocol_lock.source_identity()
    monkeypatch.setattr(protocol_lock,'source_identity',lambda:{**source,'commit':'d'*40,'tree':'e'*40,'content':{'fixed.py':'f'*64}})
    old=stage_evidence.code_identity()
    monkeypatch.setattr(stage_evidence,'code_identity',lambda:{**old,'fixed.py':'f'*64})
    assert stage_evidence.record_correctness(config,ledger)['status']=='completed'
    confirmation.record_implementation_correction(lock,ledger,'Synthetic implementation correction; frozen scientific inputs')
    corrected=run.confirm(lock,ledger)
    assert corrected['run_id']!=original['run_id']
    target=path.with_name(f"diagnostics-{corrected['run_id']}.json")
    target.write_bytes(original_bytes)
    with pytest.raises(protocol.ProtocolMismatch,match='diagnostic|Diagnostic'):
        run.diagnose(lock,ledger)
    # The original artifact remains externally ledger-bound and valid. Its
    # scientific inputs still cannot be attributed to the current corrected run.
    target.unlink()
    digest,payload=protocol_lock.read_lock(lock)
    expected=diagnostics._diagnostic_inputs(digest,payload,corrected,confirmation.read_events(lock))
    with pytest.raises(protocol.ProtocolMismatch,match='identity|inputs|Identity'):
        diagnostics._load_saved_diagnostics(path,ledger,expected)


def test_saved_diagnostics_allow_same_run_confirmation_qualification_changes(frozen_study,monkeypatch):
    from signed_study import run,diagnostics,confirmation
    _,ledger,_,lock=frozen_study
    original=run.confirm(lock,ledger)
    monkeypatch.setattr(diagnostics,'contamination_applicable',lambda *a:False)
    first=run.diagnose(lock,ledger)
    confirmation._invalidate(lock,'Synthetic late interpretation qualification')
    qualified=confirmation.load_saved_confirmation(lock,ledger)
    assert qualified['output_sha256']!=original['output_sha256']
    second=run.diagnose(lock,ledger)
    assert second['confirmation_interpretation']=='invalid_confirmatory'
    assert second['numerical']==first['numerical'] and second['timing']==first['timing']
    assert second['confirmation_run_id']==original['run_id']
    assert second['confirmation_artifact']==first['confirmation_artifact']
