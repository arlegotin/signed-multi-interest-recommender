"""Fixed-cohort tuning with actual synthetic model fits and durable evidence."""
import json

import numpy as np
import pytest

from test_stage_pipeline import pipeline, prepare


def context(pipeline):
    from signed_study.embed import load_embedding_cache
    config,ledger,_,_,_,stages,_,store = pipeline
    prepare(pipeline)
    assert stages.run_stage('calibrate',config,ledger)['status'] == 'completed'
    _,resolved,_,_,_,expected = stages._acquired(config,ledger)
    cache = load_embedding_cache(config.paths['cache']/'embeddings', expected)
    calibration = json.loads((config.paths['results']/'calibration.json').read_text())
    return config,ledger,store,cache,calibration


def test_development_real_fits_complete_selection_and_firewall(pipeline,monkeypatch):
    from signed_study import run, development, data, workload
    assert hasattr(development,'execute_development'), 'development execution missing'
    config,ledger,store,cache,calibration = context(pipeline)
    users = [e.history.user_id for e in store.development[:2]]
    calibration['forecast']['planned_user_ids']['development'] = users
    calls=[]
    original=workload.load_development_outcomes
    monkeypatch.setattr(workload,'load_development_outcomes',lambda s,u:(calls.append(u),original(s,u))[1])
    monkeypatch.setattr(data,'load_confirmation_outcomes',lambda *a:pytest.fail('confirmation exposed'))
    result=development.execute_development(config,ledger,store,cache,calibration)
    assert result['status']=='completed',result
    assert result['actual_evaluations'] == 2*(78+11+11)  # initial, outcome-free pilot, expansion
    assert calls == users+users
    assert len(result['families']) == 10
    assert result['B_star']['family'] == 'AllExemplarMargin'
    assert result['S_star']['parameters'] == {'K':1,'lambda':0}
    assert result['endpoint_coverage']['admitted'] is True
    assert result['endpoint_coverage']['expansion_batches'] == 1
    assert result['confirmation_outcomes_accessed'] is False
    assert result['planned_precision']['delta'] == .01
    assert result['history_thresholds']['H'] == pytest.approx(0.)
    assert result['history_thresholds']['O'] == pytest.approx(1.)
    assert len(result['comparison_schedule']) == 13
    assert result['selected_forecast']['status']=='feasible'
    assert result['selected_forecast']['comparison_entries']==13
    assert all('parameters' in r and 'prototype_identity' in r for r in result['matched_controls'].values())
    assert result['matched_controls']['matched_MeanNegative']['development_prototype_identities']==result['S_star']['development_prototype_identities']
    assert set(result['S_star']['development_prototype_identities'])==set(users)
    saved=json.loads((config.paths['results']/result['artifact_name']).read_text())
    assert saved['output_sha256']
    assert all(a['diagnostics']['policy']['source']=='resolved_config' for a in saved['attempts'])


def test_partial_configuration_cannot_win_and_missing_family_stops(local_config):
    from signed_study import development
    from signed_study.models import initial_configurations
    assert hasattr(development,'aggregate_configurations'), 'cohort completeness aggregation missing'
    entries=initial_configurations(local_config)
    attempts=[]
    for e in entries:
        for uid in (1,2):
            from signed_study.metrics import evaluate_user
            attempts.append(dict(configuration_id=e.id,user_id=uid,status='completed',
                                 metrics=evaluate_user([0.,0.],[0,1],[1,2])))
    for a in attempts:
        if a['configuration_id']==entries[0].id:
            a['metrics']['auc']=1.
            if a['user_id']==2: a['status']='failed'
        if a['configuration_id'].startswith('RBFSVM:'): a['status']='failed'
    rows=development.aggregate_configurations(entries,attempts,[1,2])
    assert rows[0]['complete'] is False and rows[0]['failed_share']==.5
    choices,missing=development.choose_families(rows)
    assert missing==['RBFSVM']
    assert choices['MeanNegative']['configuration'] != entries[0]


def test_declared_aliases_still_charge_and_journal(pipeline):
    from signed_study import workload
    from signed_study.models import initial_configurations
    config,ledger,store,cache,_=context(pipeline)
    entries=[e for e in initial_configurations(config) if e.family in ('SignedMax','MeanNegative')
             and e.parameters=={'K':1,'lambda':0}]
    path=config.paths['results']/'aliases.json'
    with ledger.work('G3',1,2):
        value=workload.measure_users(store,cache,[1],config,ledger,numerical_checks=False,
                                    output_path=path,declared_entries=entries)
    assert value['actual_evaluations']==2
    assert len(value['attempts'])==2
    assert value['attempts'][0]['scores']==value['attempts'][1]['scores']
    assert path.with_suffix('.jsonl').exists()


def test_complete_expansion_denial_keeps_all_pilot_entries_ineligible(pipeline,monkeypatch):
    from signed_study import development
    from signed_study.budget import BudgetExceeded
    config,ledger,store,cache,calibration=context(pipeline)
    users=[e.history.user_id for e in store.development[:2]]
    calibration['forecast']['planned_user_ids']['development']=users
    original=ledger.admit
    calls=[]
    def admit(stage,projected_seconds,evaluations):
        if evaluations: calls.append(evaluations)
        if evaluations and calls==[156,22,22]:
            raise BudgetExceeded('complete expansion does not fit')
        return original(stage,projected_seconds,evaluations)
    monkeypatch.setattr(ledger,'admit',admit)
    result=development.execute_development(config,ledger,store,cache,calibration)
    assert calls==[156,22,22]
    assert result['status']=='completed'
    assert len(result['configurations'])==78
    assert len(result['endpoint_coverage']['unavailable'])==11
    assert result['endpoint_coverage']['admitted'] is False
    assert result['complete_suite_eligible'] is False
    assert len(result['pilot_attempts'])==22
    assert all(a['metrics'] is None for a in result['pilot_attempts'])
    assert not {a['configuration_id'] for a in result['pilot_attempts']} & {
        e['id'] for e in result['configurations']}


def test_failed_required_family_stops_real_execution(pipeline,monkeypatch):
    from signed_study import development,workload
    from signed_study.models import UserModel
    config,ledger,store,cache,calibration=context(pipeline)
    calibration['forecast']['planned_user_ids']['development']=[1,2]
    original=workload.fit_user
    def fit(method,*a,**kw):
        if method=='RBFSVM':
            return UserModel(method,kw.get('parameters',{}),(),{},
                             dict(status='failed',failures=['iterations'],warnings=[]))
        return original(method,*a,**kw)
    monkeypatch.setattr(workload,'fit_user',fit)
    result=development.execute_development(config,ledger,store,cache,calibration)
    assert result['status']=='blocked'
    assert result['missing_required_families']==['RBFSVM']
    assert result['actual_evaluations']==156
    assert result['endpoint_coverage']['requested']==[]
    assert all(r['failed_share']==1. for r in result['configuration_results'] if r['configuration']['family']=='RBFSVM')
    assert 'B_star' not in result


def test_public_develop_refuses_missing_prerequisites(pipeline):
    from signed_study.run import develop
    config,ledger,*_=pipeline
    result=develop(config,ledger)
    assert result['status']=='blocked'
    assert result['stop_reason']


def test_real_variable_scores_choose_macro_auc_winners(pipeline):
    from signed_study import development
    config,ledger,store,_,calibration=context(pipeline)
    calibration['forecast']['planned_user_ids']={'development':[1,2],'confirmation':[10000,10001]}
    rng=np.random.default_rng(901)
    vectors=rng.normal(size=(256,384))
    vectors/=np.linalg.norm(vectors,axis=1,keepdims=True)
    class Cache:
        def rows(self,ids): return vectors[list(ids)]
    result=development.execute_development(config,ledger,store,Cache(),calibration)
    assert result['status']=='completed',result
    rows=result['configuration_results']
    assert len({r['macro_auc'] for r in rows if r['complete']})>3
    for family,selection in result['families'].items():
        if family in development.ALIASES: continue
        eligible=[r for r in rows if r['configuration']['family']==family and r['complete']]
        maximum=max(r['macro_auc'] for r in eligible)
        tied=[r for r in eligible if maximum-r['macro_auc']<=1e-6]
        expected=min(tied,key=lambda r:tuple(r['configuration']['parameters'].get(k,0)
                     for k in ('K','lambda','C','g'))+(r['configuration']['id'],))
        assert selection['id']==expected['configuration']['id']
    signed=result['S_star']['parameters']
    controls=result['matched_controls']
    assert controls['matched_MeanNegative']['parameters']==signed
    assert controls['matched_PositiveMax']['parameters']=={'K':signed['K'],'lambda':0}
    assert controls['matched_K1_SignedMax']['parameters']=={'K':1,'lambda':signed['lambda']}


def test_endpoint_profile_failure_cannot_create_partial_tuning(pipeline,monkeypatch):
    from signed_study import development,workload
    from signed_study.models import UserModel
    config,ledger,store,cache,calibration=context(pipeline)
    calibration['forecast']['planned_user_ids']['development']=[1,2]
    original=workload.fit_user
    def fit(method,history,embeddings,parameters,**kw):
        if method=='RBFSVM' and parameters['C']==.001:
            return UserModel(method,parameters,(),{},dict(status='failed',failures=['iterations'],warnings=[]))
        return original(method,history,embeddings,parameters,**kw)
    monkeypatch.setattr(workload,'fit_user',fit)
    result=development.execute_development(config,ledger,store,cache,calibration)
    assert result['status']=='completed'
    assert len(result['configurations'])==78
    assert len(result['endpoint_coverage']['unavailable'])==11
    assert result['endpoint_coverage']['expansion_batches']==0
    assert any(a['status']=='failed' for a in result['pilot_attempts'])
    assert all(a.get('metrics') is None for a in result['pilot_attempts'])


def test_develop_stage_verifies_smoke_resume_and_saved_artifacts(pipeline,monkeypatch):
    from signed_study import workload
    from signed_study.run import develop
    config,ledger,_,_,_,stages,_,_=pipeline
    prepare(pipeline)
    original_forecast=stages.forecast_scale
    def small(*args):
        value=original_forecast(*args)
        value['planned_user_ids']['development']=value['planned_user_ids']['development'][:8]
        return value
    monkeypatch.setattr(stages,'forecast_scale',small)
    assert stages.run_stage('calibrate',config,ledger)['status']=='completed'
    assert develop(config,ledger)['status']=='blocked'  # no smoke evidence
    monkeypatch.setattr(stages,'measure_users',workload.measure_users)
    assert stages.run_stage('smoke',config,ledger)['status']=='completed'
    result=develop(config,ledger)
    assert result['status']=='completed',result
    before=ledger.snapshot()['attempted_evaluations']
    assert develop(config,ledger)['output_sha256']==result['output_sha256']
    assert ledger.snapshot()['attempted_evaluations']==before
    from pathlib import Path
    Path(result['artifacts']['development']['path']).write_text('{}')
    assert develop(config,ledger)['status']=='blocked'


def test_develop_interruption_retains_live_attempt_references(pipeline,monkeypatch):
    from signed_study import workload
    from signed_study.budget import BudgetExceeded
    from signed_study.run import develop
    config,ledger,_,_,_,stages,_,_=pipeline
    prepare(pipeline)
    assert stages.run_stage('calibrate',config,ledger)['status']=='completed'
    # Smoke prerequisite uses real fits on its prescribed eight-user cohort.
    monkeypatch.setattr(stages,'measure_users',workload.measure_users)
    assert stages.run_stage('smoke',config,ledger)['status']=='completed'
    before=set(ledger.snapshot()['attempts'])
    def interrupted(*args,**kwargs): raise BudgetExceeded('test interruption')
    monkeypatch.setattr(workload,'fit_user',interrupted)
    result=develop(config,ledger)
    assert result['status']=='budget_exceeded'
    actual={aid for aid in ledger.snapshot()['attempts'] if aid not in before}
    assert set(result['attempt_ids'])==actual
    assert any(ledger.snapshot()['attempts'][aid]['status']!='completed' for aid in actual)
