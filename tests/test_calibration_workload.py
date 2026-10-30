"""Measured fitting boundaries retain failures and enforce development access."""
import json
import numpy as np
import pytest

from signed_study.models import ModelConfig, UserModel
from test_stage_pipeline import pipeline, prepare


def test_user_measurements_load_labels_once_and_persist_failed_fits(pipeline,monkeypatch):
    from signed_study import workload
    config,ledger,_,_,_,_,_,store=pipeline
    calls=[]
    original=workload.load_development_outcomes
    monkeypatch.setattr(workload,'load_development_outcomes',lambda s,u:(calls.append(u),original(s,u))[1])
    entries=[ModelConfig('SignedMax',{'K':4,'lambda':8},'good'),
             ModelConfig('RBFSVM',{'C':100,'g':16},'bad')]
    monkeypatch.setattr(workload,'calibration_entries',lambda c:entries)
    def fit(method,history,cache,parameters,*,config):
        return UserModel(method,parameters,(),{},dict(status='failed' if method=='RBFSVM' else 'ok',
                         failures=['iterations'] if method=='RBFSVM' else [],policy={'max_iter':config.rbf_svm['max_iter']}))
    monkeypatch.setattr(workload,'fit_user',fit)
    monkeypatch.setattr(workload,'score_user',lambda model,ids,cache,**kw:np.arange(len(ids),dtype=np.float64))
    uid=store.development[0].history.user_id
    path=config.paths['results']/'work.json'
    with ledger.work('G2',1,6):
        result=workload.measure_users(store,None,[uid],config,ledger,numerical_checks=True,output_path=path)
    assert calls==[uid]
    assert result['status']=='repair_required'
    assert result['actual_evaluations']==4  # fit+canonical score, two chunk repeats, failed fit
    saved=json.loads(path.read_text())['measurements']
    assert len(saved['attempts'])==4
    assert saved['attempts'][-1]['status']=='failed'
    assert saved['attempts'][-1]['diagnostics']['failures']==['iterations']
    assert saved['repair_decisions'][0]['kind']=='solver_iterations'
    assert saved['repair_decisions'][0]['scope']=='all_methods_and_users'
    assert saved['user_measurements'][0]['support_count']==15
    with ledger.work('G2',1,1):
        with pytest.raises(PermissionError,match='development'):
            workload.measure_users(store,None,[10000],config,ledger,numerical_checks=True,output_path=path)


def test_numerical_chunk_failure_blocks_calibration(pipeline,monkeypatch):
    from signed_study import workload
    config,ledger,_,_,_,_,_,store=pipeline
    monkeypatch.setattr(workload,'calibration_entries',lambda c:[ModelConfig('SignedMax',{'K':4,'lambda':8},'one')])
    monkeypatch.setattr(workload,'fit_user',lambda *a,**kw:UserModel('SignedMax',{'K':4,'lambda':8},(),{},
                                                                    {'status':'ok','failures':[]}))
    monkeypatch.setattr(workload,'score_user',lambda m,ids,c,*,chunk_size:np.full(len(ids),float(chunk_size)))
    with ledger.work('G2',1,3):
        result=workload.measure_users(store,None,[1],config,ledger,numerical_checks=True,
                                      output_path=config.paths['results']/'numeric.json')
    assert result['status']=='repair_required'
    assert result['chunk_invariance'] is False
    assert result['repair_decisions'][0]['kind']=='numerical_investigation'


def test_budget_interrupt_is_not_swallowed_as_a_fit_failure(pipeline,monkeypatch):
    from signed_study import workload
    from signed_study.budget import BudgetExceeded
    config,ledger,_,_,_,_,_,store=pipeline
    def interrupted(*a,**kw): raise BudgetExceeded('stop now')
    monkeypatch.setattr(workload,'fit_user',interrupted)
    with ledger.work('G2',1,33):
        with pytest.raises(BudgetExceeded,match='stop now'):
            workload.measure_users(store,None,[1],config,ledger,numerical_checks=True,
                                  output_path=config.paths['results']/'interrupted.json')


def test_real_fits_and_metrics_use_saved_cache_and_uniform_policy(pipeline,monkeypatch):
    from signed_study import workload
    from signed_study.embed import load_embedding_cache
    config,ledger,_,_,_,stages,_,store=pipeline
    prepare(pipeline)
    assert stages.run_stage('calibrate',config,ledger)['status']=='completed'
    _,resolved,_,_,_,expected=stages._acquired(config,ledger)
    cache=load_embedding_cache(config.paths['cache']/'embeddings',expected)
    monkeypatch.setattr(workload,'initial_configurations',lambda c:[
        ModelConfig('SignedMax',{'K':4,'lambda':8},'signed'),
        ModelConfig('LinearLogistic',{'C':100},'linear')])
    with ledger.work('G3',1,2):
        measured=workload.measure_users(store,cache,[1],resolved,ledger,numerical_checks=False,
                                       output_path=config.paths['results']/'real_fits.json')
    assert measured['status']=='completed',measured
    assert measured['actual_evaluations']==2
    assert [a['metrics']['auc'] for a in measured['attempts']]==[.5,.5]
    assert all(a['diagnostics']['policy']['source']=='resolved_config' for a in measured['attempts'])
    assert len(measured['resource_trace'])==1


@pytest.mark.parametrize('case,accepted,near,distant',[
    ('eta_boundary',True,0,0), ('above_eta',False,0,0),
    ('distant_reversal',False,0,1), ('near_reversal',True,1,0),
    ('two_eta_tie',True,1,0)])
def test_declared_eta_and_pairwise_order_evidence(pipeline,monkeypatch,case,accepted,near,distant):
    from signed_study import workload
    config,ledger,_,_,_,_,_,store=pipeline
    monkeypatch.setattr(workload,'calibration_entries',lambda c:[ModelConfig('SignedMax',{'K':4,'lambda':8},'one')])
    monkeypatch.setattr(workload,'fit_user',lambda *a,**kw:UserModel('SignedMax',{'K':4,'lambda':8},(),{},
                                                                    {'status':'ok','failures':[]}))
    eta=9e-10
    reference=np.arange(10,dtype=np.float64)
    repeat=reference.copy()
    if case=='eta_boundary': repeat[0]=eta
    elif case=='above_eta': repeat[0]=np.nextafter(eta,np.inf)
    elif case=='distant_reversal': reference[:2]=[0,2.2e-9]; repeat[:2]=[1.2e-9,1e-9]
    elif case=='near_reversal': reference[:2]=[0,1e-10]; repeat[:2]=[2e-10,0]
    else: reference[:2]=[0,1.8e-9]; repeat[:2]=[9e-10,9e-10]
    monkeypatch.setattr(workload,'score_user',lambda *a,chunk_size:reference.copy() if chunk_size==512 else repeat.copy())
    with ledger.work('G2',1,3):
        result=workload.measure_users(store,None,[1],config,ledger,numerical_checks=True,
                                      output_path=config.paths['results']/f'{case}.json')
    assert result['chunk_invariance'] is accepted
    checks=[a for a in result['attempts'] if a['kind']=='score_repeat']
    for check in checks:
        assert check['tolerance']==eta
        assert check['pairwise_order_accepted'] is (distant==0)
        assert check['distant_order_changes']==distant
        assert check['near_tie_order_changes']==near
        assert len(check['near_tie_examples'])==near
        if near:
            assert check['near_tie_examples'][0]['item_ids']==[200,201]


def test_dense_near_tie_diagnostic_has_bounded_evidence():
    from signed_study import workload
    assert hasattr(workload,'numerical_variation'), 'bounded ordering diagnostic is missing'
    n=2048
    reference=np.arange(n,dtype=np.float64)*1e-14
    result=workload.numerical_variation(reference,reference[::-1],np.arange(n),eta=9e-10)
    assert result['accepted'] is True
    assert result['near_tie_order_changes']==2096128
    assert len(result['near_tie_examples'])==32
    assert result['near_tie_examples_truncated'] is True
    assert result['pairwise_order_accepted'] is True


def test_workload_records_measured_scoring_cost_for_each_chunk(pipeline,monkeypatch):
    from signed_study import workload
    config,ledger,clock,_,_,_,_,store=pipeline
    monkeypatch.setattr(workload,'calibration_entries',lambda c:[ModelConfig('SignedMax',{'K':4,'lambda':8},'one')])
    monkeypatch.setattr(workload.time,'perf_counter_ns',lambda:int(clock[0]*1e9))
    def fit(*a,**kw):
        clock[0]+=.01
        return UserModel('SignedMax',{'K':4,'lambda':8},(),{},dict(status='ok',failures=[]))
    def score(model,ids,cache,*,chunk_size):
        clock[0]+={512:.02,1:.4,7:.2}[chunk_size]
        return np.arange(len(ids),dtype=np.float64)
    monkeypatch.setattr(workload,'fit_user',fit)
    monkeypatch.setattr(workload,'score_user',score)
    with ledger.work('G2',1,3):
        result=workload.measure_users(store,None,[1],config,ledger,numerical_checks=True,
                                      output_path=config.paths['results']/'chunk_times.json')
    assert result['status']=='completed'
    row=result['user_measurements'][0]
    assert row['chunk_size']==512
    assert set(row['score_measurements'])=={'1','7','512'}
    for chunk,seconds in [('512',.02),('1',.4),('7',.2)]:
        assert row['score_measurements'][chunk]['score_seconds']==pytest.approx(seconds)
        assert row['score_measurements'][chunk]['numerical_accepted'] is True
