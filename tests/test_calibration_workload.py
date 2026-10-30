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
