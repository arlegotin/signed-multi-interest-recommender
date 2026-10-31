"""Regressions for the native watchdog failure and whole-branch review."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import json
import signal
import sys
import threading
import time

import numpy as np
import pytest

from test_stages_and_provenance import budget_case


@pytest.fixture
def unblocked_alarm():
    previous=signal.pthread_sigmask(signal.SIG_UNBLOCK,{signal.SIGALRM})
    try: yield
    finally: signal.pthread_sigmask(signal.SIG_SETMASK,previous)


def test_slow_timer_sampler_never_reenters_and_restores_ownership(local_config, tmp_path, unblocked_alarm):
    from signed_study.budget import Ledger, BudgetExceeded
    resources=dict(installed_ram_bytes=32*2**30,available_memory_bytes=16*2**30,
                   rss_bytes=1,project_disk_bytes=1000,free_disk_bytes=20*2**30)
    state=dict(depth=0,slow=False,max_depth=0)
    def sample():
        state['depth']+=1;state['max_depth']=max(state['max_depth'],state['depth'])
        try:
            assert state['depth']==1, 'resource sampler reentered'
            if state['slow']: time.sleep(.14)
            return dict(resources)
        finally: state['depth']-=1
    stop=threading.Event();worker=threading.Thread(target=lambda:stop.wait(3));worker.start()
    original_handler=signal.getsignal(signal.SIGALRM)
    original_timer=signal.getitimer(signal.ITIMER_REAL)
    prior=lambda *_: None
    signal.signal(signal.SIGALRM,prior)
    signal.setitimer(signal.ITIMER_REAL,20,30)
    ledger=Ledger(tmp_path/'slow.jsonl',local_config,sampler=sample,monitor=True)
    try:
        with ledger.work('G1',1,0):
            state['slow']=True
            time.sleep(.38)
            state['slow']=False
        assert state['max_depth']==1
        with pytest.raises(BudgetExceeded,match='RSS'):
            with ledger.work('G1',1,0):
                resources['rss_bytes']=9*2**30
                time.sleep(.3)
        assert signal.getsignal(signal.SIGALRM)==prior
        delay,interval=signal.getitimer(signal.ITIMER_REAL)
        assert 17<delay<20 and interval==30
        assert not ledger.snapshot()['unfinished_attempts']
        assert list(ledger.snapshot()['attempts'].values())[-1]['status']=='budget_exceeded'
        resources['rss_bytes']=1
        with ledger.work('G1',0,0): pass
    finally:
        state['slow']=False;ledger.close();stop.set();worker.join(timeout=1)
        signal.signal(signal.SIGALRM,original_handler)
        signal.setitimer(signal.ITIMER_REAL,*original_timer)


@pytest.mark.parametrize('method,parameters,target',[
    ('SignedMax',{'K':1,'lambda':1},'spherical_prototypes'),
    ('LinearLogistic',{'C':1},'LogisticRegression'),
    ('RBFSVM',{'C':1,'g':1},'SVC')])
def test_resource_stop_inside_model_fit_propagates(budget_case, monkeypatch, method,parameters,target):
    from signed_study import models
    from signed_study.budget import BudgetExceeded
    from signed_study.data import History
    ledger,_,_=budget_case
    def stopped(*a,**kw): raise BudgetExceeded('transient native RSS')
    monkeypatch.setattr(models,target,stopped if target=='spherical_prototypes' else lambda **kw:SimpleNamespace(fit=stopped))
    history=History(1,0,'development',(1,2),(3,4))
    cache=SimpleNamespace(rows=lambda ids:np.array([[1.,0.],[0.,1.],[-1.,0.],[0.,-1.]]))
    with pytest.raises(BudgetExceeded,match='transient native RSS'):
        with ledger.work('G3',0,1): models.fit_user(method,history,cache,parameters)
    assert list(ledger.snapshot()['attempts'].values())[-1]['status']=='budget_exceeded'


def test_repair_command_one_cycle_time_once_and_caps(budget_case):
    from signed_study.budget import BudgetExceeded
    ledger,clock,_=budget_case
    ledger.clock=time.monotonic  # positive actual command cost, no time-zero identity
    assert ledger.run_command([sys.executable,'-c','pass'],stage='repair',projected_seconds=1)==0
    state=ledger.snapshot();attempt=list(state['attempts'].values())[-1]
    assert state['repair_cycles']==1 and state['attempted_evaluations']==0
    assert state['experimental_seconds']==state['stage_seconds']['repair']>0
    assert attempt['outputs']['process_group_cleanup']['verified_no_live_members']
    assert ledger.run_command([sys.executable,'-c','raise SystemExit(3)'],stage='repair',projected_seconds=1)==3
    with pytest.raises(BudgetExceeded,match='cycles'): ledger.run_command([sys.executable,'-c','pass'],stage='repair',projected_seconds=1)
    for stage in ('G2','G3','G4','G5','G6'):
        with pytest.raises(ValueError): ledger.run_command([sys.executable,'-c','pass'],stage=stage)


def test_interrupted_encoder_download_reuses_first_resolution(embedding_inputs):
    from signed_study.embed import resolve_encoder
    config=replace(embedding_inputs['config'],model_revision=None)
    revisions=[];downloads=[]
    files=embedding_inputs['encoder_provenance']['files']
    class API:
        def model_info(self,repo_id,revision,**kw):
            revisions.append(revision)
            return SimpleNamespace(sha=revision or ('a'*40 if len(revisions)==1 else 'b'*40),
                siblings=[SimpleNamespace(rfilename=f) for f in files],card_data={'license':'apache-2.0'})
    def download(**kw):
        downloads.append(kw['revision'])
        if len(downloads)==1: raise OSError('interrupted download')
        return embedding_inputs['encoder_provenance']['model_path']
    with pytest.raises(OSError,match='interrupted'): resolve_encoder(config,api=API(),download=download)
    assert not (config.paths['results']/'encoder_acquisition.json').exists()
    result=resolve_encoder(config,api=API(),download=download)
    assert downloads==['a'*40,'a'*40] and revisions==[None,'a'*40]
    assert result['model_revision']=='a'*40
    with pytest.raises(ValueError,match='intent'):
        resolve_encoder(replace(config,model_revision='b'*40),api=API(),download=download)


def test_fixed_text_normalizes_whitespace():
    from signed_study.embed import format_movie
    assert format_movie('  Toy\t Story  (1995)\n',['Comedy','Animation'])=='Movie: Toy Story (1995). Genres: Animation, Comedy.'
    with pytest.raises(ValueError): format_movie(' \t\n',[])


@pytest.mark.parametrize('low,high,missing,suite,expected',[
    (.02,.04,1,True,'supported contribution'),(-.04,-.02,1,True,'informative negative result'),
    (.02,.04,0,False,'supported contribution'),(-.01,.02,1,False,'inconclusive evidence')])
def test_primary_whole_cohort_decision_independent_of_coverage(low,high,missing,suite,expected):
    from signed_study.report import _outcome
    saved=dict(interpretation='confirmatory',status='partial',complete_suite_eligible=suite,
               primary=dict(low=low,high=high,n_missing=missing))
    assert _outcome(saved,False,False,.01)==expected


@pytest.mark.parametrize('missing',[0,1])
def test_public_paired_diagnostics_allowlist(missing):
    from signed_study.report import _aggregate
    value=dict(low=.02,high=.04,standard_error=.005 if not missing else None,
               complete_case=dict(label='complete_case_diagnostic',n_users=2,mean_difference=.03,standard_error=.005,user_ids=[991]),
               n_users=2+missing,n_complete=2,n_missing=missing,user_ids=[991],scores=[1])
    result=_aggregate(value)
    assert result['standard_error']==value['standard_error']
    assert result['complete_case']==dict(label='complete_case_diagnostic',n_users=2,mean_difference=.03,standard_error=.005)
    assert 'user_ids' not in json.dumps(result) and 'scores' not in result


def test_timer_defers_disk_walk_and_checkpoint_reentry(local_config,tmp_path,unblocked_alarm):
    from signed_study.budget import Ledger, ResourceSampler
    class Sampler(ResourceSampler):
        forbid_disk=False
        def refresh_disk(self):
            assert not self.forbid_disk, 'timer entered expensive disk walk'
            super().refresh_disk()
    sampler=Sampler(tmp_path)
    ledger=Ledger(tmp_path/'ticks.jsonl',local_config,sampler=sampler,monitor=True)
    stop=threading.Event();worker=threading.Thread(target=lambda:stop.wait(2));worker.start()
    try:
        with ledger.work('G1',1,0):
            sampler._disk_at=-float('inf');sampler.forbid_disk=True
            try: time.sleep(.25)
            finally: sampler.forbid_disk=False
        assert not ledger.snapshot()['unfinished_attempts']
    finally: ledger.close();stop.set();worker.join(timeout=1)


@pytest.mark.parametrize('review_id', ['R1', 'N1'])
def test_report_accepts_only_published_accounted_repair_log(budget_case, review_id):
    from signed_study import report
    from signed_study.budget import atomic_write_json
    from signed_study.protocol import canonical_sha256
    from signed_study.stage_evidence import file_reference
    ledger,clock,_=budget_case
    start=len(ledger.snapshot()['events'])
    with ledger.work('G1',0,0) as diagnostic: clock.now+=2
    with ledger.work('repair',0,0) as repair: clock.now+=3
    end=len(ledger.snapshot()['events'])
    path=ledger.path.parent/'repair-log.json'
    trace=ledger.path.parent/'failure.log';trace.write_text('retained failure trace')
    record=dict(schema_version=1,review_ids=[review_id],source_base='a'*40,source_sha256='b'*64,
                failure_attempt_ids=[],failure_trace=file_reference(trace),
                validation_attempt_ids=[diagnostic,repair],repair_attempt_id=repair,repair_cycle=1,
                ledger_event_start=start,ledger_event_end=end,attributed_seconds=5.)
    record['output_sha256']=canonical_sha256(record);atomic_write_json(path,record)
    problems=[]
    assert report._repair_summaries(ledger.snapshot(),problems)==[]
    with ledger.work('G0',0,0) as publication:
        ledger.finish(publication,'completed',{}, {'repair_log':file_reference(path)})
    summaries=report._repair_summaries(ledger.snapshot(),problems)
    assert len(summaries)==1 and summaries[0]['attributed_seconds']==5.
    assert summaries[0]['repair_cycle']==1 and not problems
    # Even a newly published, self-hashed account cannot lie about actual cost.
    record['attributed_seconds']=1.;record.pop('output_sha256')
    record['output_sha256']=canonical_sha256(record);atomic_write_json(path,record)
    with ledger.work('G0',0,0) as publication:
        ledger.finish(publication,'completed',{}, {'repair_log':file_reference(path)})
    problems=[]
    assert report._repair_summaries(ledger.snapshot(),problems)==[]
    assert problems


def test_repair_validation_preserves_time_ceiling(budget_case):
    from signed_study.budget import BudgetExceeded
    ledger,clock,_=budget_case
    with ledger.work('repair',0,0): clock.now+=1199
    with pytest.raises(BudgetExceeded,match='stage time'):
        ledger.run_command([sys.executable,'-c','pass'],stage='repair',projected_seconds=1)
    assert ledger.snapshot()['repair_cycles']==1


@pytest.mark.parametrize('low,high,label',[(-.04,-.02,'harm'),(-.005,.005,'small_difference')])
def test_primary_decision_exposes_harm_and_small_difference(low,high,label):
    from signed_study.report import _primary_decision
    saved=dict(interpretation='confirmatory',status='partial',primary=dict(low=low,high=high,n_missing=1))
    assert label in _primary_decision(saved,False,.01)
    assert label not in _primary_decision(saved,True,.01)
