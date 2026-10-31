"""Saved-schema fixtures exercise reports without executing a synthetic study."""
import hashlib
import json
from pathlib import Path

import pytest

from signed_study.protocol import canonical_sha256


def save(path, record):
    from signed_study.protocol_lock import canonical_bytes
    record = dict(record, schema_version=1)
    record.pop('output_sha256', None)
    record['output_sha256'] = canonical_sha256(record)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(record))
    return record


@pytest.fixture
def saved_evidence(local_config):
    from signed_study.budget import Ledger
    from signed_study.protocol_lock import canonical_bytes
    from signed_study.stage_evidence import config_record, file_reference
    from signed_study.confirmation import _event
    tick = [0.]
    sample = dict(rss_bytes=1, installed_ram_bytes=64*2**30, available_memory_bytes=8*2**30,
                  project_disk_bytes=1, free_disk_bytes=20*2**30)
    ledger = Ledger(local_config.paths['results']/'budget_ledger.jsonl', local_config,
                    clock=lambda: tick[0], sampler=lambda: sample, monitor=False)
    root = local_config.paths['results']; lock = root/'protocol.lock.json'
    with ledger.work('G1', 0, 0) as failed:
        tick[0] += 3
        ledger.finish(failed, 'failed', {}, {})
    with ledger.work('G3', 0, 0) as freeze:
        payload = dict(schema_version=1, invocation_config=config_record(local_config),
                       config=config_record(local_config), freeze_attempt_id=freeze,
                       ledger_snapshot=ledger.snapshot(), ledger_path=str(ledger.path),
                       identities=dict(source={'commit':'a'*40}, cache_sha256='cache'),
                       data_access={'episodes_sha256':'episodes'}, deviations=[],
                       development=dict(B_star={'family':'MeanNegative'}, complete_suite_eligible=True,
                           endpoint_coverage={}, boundary_winners=[],
                           comparison_schedule=[dict(name='SignedMax',parameters={'K':2,'lambda':1}),
                                                dict(name='MeanNegative',parameters={'K':2,'lambda':1}),
                                                dict(name='RBFSVM',parameters={'C':1})]))
        lock.write_bytes(canonical_bytes(payload))
        digest = hashlib.sha256(lock.read_bytes()).hexdigest()
        lock.with_suffix('.sha256').write_text(digest+'\n')
        ledger.finish(freeze, 'completed', {}, {'protocol_lock':file_reference(lock),
                                               'sidecar':file_reference(lock.with_suffix('.sha256'))})
    with ledger.work('G4', 0, 6) as execution:
        _event(lock, 'run_started', run_id=digest, ledger_attempt_id=execution)
        comparison = dict(low=.02, high=.04, sample_mean_low=.03, sample_mean_high=.03,
                          n_users=2,n_complete=2,n_missing=0,lower_bounds=[.02,.04],upper_bounds=[.02,.04])
        path = root/f'confirmation-{digest}.json'
        result = save(path, dict(lock_sha256=digest,run_id=digest,interpretation='confirmatory',
            status='completed',scheduled_user_ids=[1,2],attempt_ids=[execution],analysis_attempt_ids=[],
            actual_evaluations=6,primary_comparator='MeanNegative',primary=comparison,
            comparisons={'MeanNegative':comparison,'RBFSVM':dict(comparison,low=-.06,high=-.02,
                sample_mean_low=-.04,sample_mean_high=-.04,lower_bounds=[-.06,-.02],upper_bounds=[-.06,-.02])},
            comparison_schedule=payload['development']['comparison_schedule'],complete_suite_eligible=True,
            endpoint_coverage={}, boundary_winners=[],artifact_path=str(path),stop_reason=None,
            attempts=[dict(user_id=u,slot=slot,status='completed',metrics={'auc':auc},
                           candidate_ids=[999],scores=[.123],fitted_state={'vectors':'PRIVATE'})
                      for u,slot,auc in [(1,'SignedMax',.8),(2,'SignedMax',.8),
                                        (1,'MeanNegative',.78),(2,'MeanNegative',.76),
                                        (1,'RBFSVM',.86),(2,'RBFSVM',.82)]]))
        _event(lock,'run_finished',run_id=digest,artifact=file_reference(path),interpretation='confirmatory')
        ledger.finish(execution, 'completed', {}, {'confirmation':file_reference(path)})
    yield root, ledger, lock, result
    ledger.close()


def test_report_preserves_scope_without_complete_results(saved_evidence, monkeypatch):
    from signed_study import report, confirmation, embed, data, protocol_lock
    root, ledger, lock, saved = saved_evidence
    def forbidden(*a, **kw): pytest.fail('report performed live work or opened labels/model')
    for owner, name in ((confirmation,'fit_user'),(embed,'encode_items'),(data,'load_confirmation_outcomes'),
                        (protocol_lock,'live_inputs'),(protocol_lock,'_context')):
        monkeypatch.setattr(owner,name,forbidden)
    first = report.render_report(root)
    text = Path(first['report']).read_text()
    assert 'supported contribution' in text
    assert 'B*: MeanNegative' in text and 'RBFSVM' in text
    assert 'practical case' in text and 'absent relative to RBFSVM' in text
    assert 'complete-suite' in text and 'not established' in text
    assert len([line for line in text.splitlines() if line.startswith('## ')]) == 10
    table = Path(first['tables'][0]).read_bytes()
    figure_bytes=[Path(p).read_bytes() for p in first['figures']]
    assert b'0.02' in table and b'0.04' in table
    assert report.render_report(root)['outcome'] == 'supported contribution'
    assert Path(first['tables'][0]).read_bytes() == table
    assert [Path(p).read_bytes() for p in first['figures']]==figure_bytes
    metrics=next(Path(p).read_text() for p in first['tables'] if p.endswith('metrics.csv'))
    assert 'SignedMax,2,2,0.8' in metrics
    exports = ''.join(Path(p).read_text() for p in [first['report'],first['inventory'],*first['tables']])
    assert all(secret not in exports for secret in ('candidate_ids','PRIVATE','0.123','user_ids'))
    assert '3.0' in Path(first['tables'][1]).read_text()  # failed G1 work retained
    assert ledger.snapshot()['attempted_evaluations'] == 6


@pytest.mark.parametrize('low,high,interpretation,status,expected', [
    (.02,.04,'confirmatory','completed','supported contribution'),
    (-.04,0.,'confirmatory','completed','informative negative result'),
    (-.01,.02,'confirmatory','completed','inconclusive evidence'),
    (.02,.04,'invalid_confirmatory','completed','invalid experiment'),
    (.02,.04,'corrected_exploratory','completed','inconclusive evidence'),
    (.02,.04,'confirmatory','budget_exceeded','practical infeasibility'),
])
def test_report_outcome_qualification(saved_evidence, low, high, interpretation, status, expected):
    from signed_study import report, confirmation
    root, ledger, lock, saved = saved_evidence
    # Save different scientific summaries while retaining real external references.
    path = Path(saved['artifact_path'])
    saved['primary'].update(low=low,high=high)
    saved['status']=status
    if interpretation == 'invalid_confirmatory':
        confirmation._event(lock,'invalidated',reason='synthetic baseline defect')
    elif interpretation == 'corrected_exploratory':
        confirmation._event(lock,'invalidated',reason='implementation defect')
        confirmation._event(lock,'correction_authorized',correction_id='corrected',reason='fixed')
        saved.update(run_id='corrected',interpretation=interpretation)
        path=root/'confirmation-corrected.json'; saved['artifact_path']=str(path)
    # A new fixture publication has its own terminal accounting anchor.
    with ledger.work('G4',0,0) as publication:
        saved['attempt_ids'].append(publication)
        save(path,saved)
        from signed_study.stage_evidence import file_reference
        ledger.finish(publication,'budget_exceeded' if status=='budget_exceeded' else 'completed',{},
                      {'confirmation':file_reference(path)})
    # Re-publish fixture bytes by replacing the original final-event/reference;
    # no production stage, fitting or label access occurs in this fixture.
    from signed_study.stage_evidence import file_reference
    events=confirmation.read_events(lock)
    events=[e for e in events if e['kind']!='run_finished']
    confirmation._events_path(lock).unlink()
    for e in events:
        fields={k:v for k,v in e.items() if k not in ('schema_version','output_sha256','previous_sha256','lock_sha256','kind')}
        confirmation._event(lock,e['kind'],**fields)
    confirmation._event(lock,'run_finished',run_id=saved['run_id'],artifact=file_reference(path),interpretation=interpretation)
    rendered=report.render_report(root)
    assert rendered['outcome']==expected
    table=Path(rendered['tables'][0]).read_bytes()
    assert report.render_report(root)['outcome']==expected
    assert Path(rendered['tables'][0]).read_bytes()==table
    if interpretation=='corrected_exploratory': assert 'corrected exploratory' in Path(rendered['report']).read_text()


@pytest.mark.parametrize('blocker', ['encoder missing', 'non-target machine'])
def test_report_early_stops(local_config, blocker):
    from signed_study import report
    root=local_config.paths['results']; root.mkdir(exist_ok=True)
    save(root/'environment.json',dict(target_compatible=False,blockers=[blocker],hardware={'chip':'unavailable'}))
    result=report.render_report(root)
    text=Path(result['report']).read_text()
    assert result['outcome']=='inconclusive evidence' and 'not executed' in text and blocker in text
    assert not result['figures'] and len(result['tables'])==2  # cost/actual-work tables only


@pytest.mark.parametrize('status,stage,relative,other_path,wrong_digest,verified', [
    ('completed','G0',True,False,False,True),
    ('completed','G0','dotted',False,False,True),
    ('completed','G0','parent',False,False,True),
    ('blocked','G0',True,False,False,True),
    ('failed','G0',False,False,False,False),
    ('interrupted','G0',False,False,False,False),
    ('completed','G1',False,False,False,False),
    ('completed','G0',False,True,False,False),
    ('completed','G0',False,False,True,False),
])
def test_inspection_publication_uses_exact_artifact_and_terminal_owner(
        local_config,status,stage,relative,other_path,wrong_digest,verified):
    from signed_study import report
    from signed_study.budget import Ledger
    from signed_study.stage_evidence import file_reference
    root=local_config.paths['results'].resolve();root.mkdir(exist_ok=True)
    path=root/'environment.json'
    save(path,dict(target_compatible=status!='blocked',blockers=['non-target machine'] if status=='blocked' else [],hardware={'chip':'fixture'}))
    publication=path
    if other_path:
        publication=root/'same-content.json';publication.write_bytes(path.read_bytes())
    reference=file_reference(publication)
    if relative:
        reference['path']=str(publication.relative_to(Path.cwd()))
        if relative=='dotted':reference['path']='./'+reference['path']
        if relative=='parent':reference['path']='results/../'+reference['path']
    if wrong_digest: reference['sha256']='0'*64
    sample=dict(rss_bytes=1,installed_ram_bytes=64*2**30,available_memory_bytes=8*2**30,project_disk_bytes=1,free_disk_bytes=20*2**30)
    ledger=Ledger(root/'budget_ledger.jsonl',local_config,clock=lambda:0.,sampler=lambda:sample,monitor=False)
    with ledger.work(stage,0,0) as owner:
        ledger.finish(owner,status,{}, {'environment':reference})
    ledger.close()
    rendered=report.render_report(root)
    text=Path(rendered['report']).read_text()
    assert ('Hardware (saved inspection; ledger publication verified)' in text)==verified
    if status=='blocked':
        assert 'non-target machine' in text and rendered['outcome']=='inconclusive evidence'


def test_historical_reader_is_nonmutating_and_checks_anchor(saved_evidence, monkeypatch):
    from signed_study import confirmation, protocol_lock
    from signed_study.protocol import ProtocolMismatch
    _,ledger,lock,saved=saved_evidence
    before=ledger.path.read_bytes()
    monkeypatch.setattr(protocol_lock,'live_inputs',lambda *a:pytest.fail('docs HEAD/model should be irrelevant'))
    result=confirmation.load_historical_confirmation(lock,ledger)
    assert result['run_id']==saved['run_id'] and ledger.path.read_bytes()==before
    with pytest.raises(BaseException,match='docs HEAD/model'):
        confirmation.load_saved_confirmation(lock,ledger)
    lock.with_suffix('.sha256').write_text('changed')
    with pytest.raises(ProtocolMismatch): confirmation.load_historical_confirmation(lock,ledger)


def test_report_cli_works_for_early_stop(local_config,capsys):
    from signed_study.__main__ import main
    assert main(['report','--results',str(local_config.paths['results'])])==0
    assert Path(json.loads(capsys.readouterr().out)['report']).exists()


def test_historical_reader_rejects_rewritten_summary_and_checks_all_analysis_ids(saved_evidence):
    from signed_study import confirmation
    from signed_study.protocol import ProtocolMismatch
    root,ledger,lock,saved=saved_evidence
    path=Path(saved['artifact_path'])
    saved['primary']['low']=.9
    save(path,saved)
    # Self-checksum plus a rewritten run_finished journal is not publication.
    from signed_study.stage_evidence import file_reference
    rows=confirmation.read_events(lock)
    confirmation._events_path(lock).unlink()
    for e in rows:
        fields={k:v for k,v in e.items() if k not in ('schema_version','output_sha256','previous_sha256','lock_sha256','kind')}
        if e['kind']=='run_finished': fields['artifact']=file_reference(path)
        confirmation._event(lock,e['kind'],**fields)
    with pytest.raises(ProtocolMismatch,match='publication'):
        confirmation.load_historical_confirmation(lock,ledger)


def test_report_missing_baseline_and_analysis_pending_remain_inconclusive(saved_evidence):
    from signed_study import report,confirmation
    root,ledger,lock,saved=saved_evidence
    from signed_study.stage_evidence import file_reference
    path=Path(saved['artifact_path'])
    saved.update(complete_suite_eligible=False,status='partial')
    for row in saved['attempts']:
        if row['slot']=='RBFSVM': row.update(status='failed',exception='synthetic baseline defect',metrics=None)
    saved['comparisons']['RBFSVM'].update(n_missing=2,low=-1.,high=1.)
    with ledger.work('G4',0,0) as publication:
        saved['attempt_ids'].append(publication)
        save(path,saved)
        ledger.finish(publication,'completed',{}, {'confirmation':file_reference(path)})
    rows=confirmation.read_events(lock);confirmation._events_path(lock).unlink()
    for e in rows:
        fields={k:v for k,v in e.items() if k not in ('schema_version','output_sha256','previous_sha256','lock_sha256','kind')}
        if e['kind']=='run_finished': fields['artifact']=file_reference(path)
        confirmation._event(lock,e['kind'],**fields)
    initial=report.render_report(root)
    assert initial['outcome']=='supported contribution'
    assert 'Complete-suite conclusion: not established' in Path(initial['report']).read_text()
    initial_text=Path(initial['report']).read_text()
    # An interrupted analysis admission outside summary.attempt_ids must qualify
    # the historical result, not disappear from reporting.
    with ledger.work('G4',0,0) as analysis:
        confirmation._event(lock,'analysis_started',run_id=saved['run_id'],ledger_attempt_id=analysis)
        ledger.finish(analysis,'failed',{}, {})
    current=confirmation.load_historical_confirmation(lock,ledger)
    assert current['status']=='partial' and current['accounting_status']==['failed']
    # Removing finished evidence leaves a useful pending report without inference.
    rows=confirmation.read_events(lock);confirmation._events_path(lock).unlink()
    for e in rows:
        if e['kind']=='run_finished': continue
        fields={k:v for k,v in e.items() if k not in ('schema_version','output_sha256','previous_sha256','lock_sha256','kind')}
        confirmation._event(lock,e['kind'],**fields)
    result=report.render_report(root)
    assert result['outcome']=='inconclusive evidence' and len(result['tables'])==2
    assert 'analysis pending' in Path(result['report']).read_text()
    assert not (root/'comparisons.csv').exists()
    assert not (root/'paired-intervals.png').exists()
    names=[r['name'] for r in json.loads(Path(result['inventory']).read_text())['artifacts']]
    assert ('synthetic baseline defect' in initial_text,'confirmation.events.jsonl' in names)==(True,True)


def test_historical_report_diagnostics_keep_external_run_binding(saved_evidence):
    from signed_study import report,diagnostics,confirmation,protocol_lock
    from signed_study.stage_evidence import file_reference
    root,ledger,lock,saved=saved_evidence
    digest,payload=protocol_lock.read_lock(lock)
    inputs=diagnostics._diagnostic_inputs(digest,payload,saved,confirmation.read_events(lock))
    journal=root/'diagnostic-attempts.jsonl';journal.write_text('')
    path=root/f'diagnostics-{saved["run_id"]}.json'
    with ledger.work('G5',0,0) as publication:
        diag=save(path,dict(lock_sha256=digest,confirmation_run_id=saved['run_id'],
            confirmation_artifact=inputs['confirmation_artifact'],inputs=inputs,input_sha256=canonical_sha256(inputs),
            confirmation_status='completed',confirmation_interpretation='confirmatory',
            status='completed',attempt_ids=[publication],artifact_path=str(path),attempt_journal=file_reference(journal),
            interpretation='secondary_unadjusted',strata={'H_O':{'high_H_high_O':dict(n_users=2,status='available',imprecise=True,comparisons={'MeanNegative':saved['primary']})}},
            numerical=[],contamination={'status':'skipped','reason':'not applicable'},timing={}))
        ledger.finish(publication,'completed',{}, {'diagnostics':file_reference(path)})
    result=report.render_report(root)
    assert 'not applicable' in Path(result['report']).read_text()
    diag['strata']['H_O']['high_H_high_O']['n_users']=200
    save(path,diag)
    result=report.render_report(root)
    assert any('publication' in problem for problem in result['limitations'])
    assert '"n_users": 200' not in Path(result['report']).read_text()


def test_cli_diagnose_dispatches_saved_lock(saved_evidence,monkeypatch,capsys):
    from signed_study.__main__ import main
    from signed_study import run
    _,_,lock,_=saved_evidence
    monkeypatch.setattr(run,'diagnose',lambda path,ledger:dict(status='completed',lock=str(path)))
    assert main(['diagnose','--lock',str(lock)])==0
    assert json.loads(capsys.readouterr().out)['lock']==str(lock)


def test_failed_stage_reason_survives_early_stop(saved_evidence):
    from signed_study import report
    root,ledger,lock,saved=saved_evidence
    lock.unlink();lock.with_suffix('.sha256').unlink()
    save(root/'stages'/'calibrate.failed-fixture.json',dict(stage='calibrate',status='blocked',
        artifacts={},attempt_ids=saved['attempt_ids'],stop_reason='Smallest permitted scale cannot fit remaining allocations and reserve'))
    result=report.render_report(root)
    assert 'Smallest permitted scale' in Path(result['report']).read_text()


def test_historical_report_survives_changed_head_and_missing_installed_model(saved_evidence,monkeypatch):
    from signed_study import report,protocol_lock,embed
    root,ledger,lock,_=saved_evidence
    first=report.render_report(root)
    table=Path(first['tables'][0]).read_bytes()
    # A docs-only HEAD differs from the frozen source while saved scientific
    # bytes remain intact. The installed encoder is deliberately absent.
    monkeypatch.setattr(protocol_lock,'source_identity',lambda:dict(commit='docs-only-head',clean=True))
    monkeypatch.setattr(embed,'load_encoder',lambda *a,**k:pytest.fail('historical model loading'))
    assert not (Path('.cache')/'models').exists()
    second=report.render_report(root)
    assert second['outcome']==first['outcome'] and Path(second['tables'][0]).read_bytes()==table


@pytest.mark.parametrize('published',[False,True])
def test_saved_measured_scaling_stop_is_practical_infeasibility(saved_evidence,published):
    from signed_study import report
    root,ledger,lock,_=saved_evidence
    lock.unlink();lock.with_suffix('.sha256').unlink()
    reason='Smallest permitted scale cannot fit remaining allocations and reserve'
    save(root/'calibration.json',dict(status='completed',actual_evaluations=1,timings={'catalog_encode_seconds':4.},
        forecast=dict(status='infeasible',tested_scales=[[418,1776],[256,1000]],
                      support_model='upper observed size-bin envelope plus measured nonnegative slope',stop_reason=reason)))
    with ledger.work('G2',0,1) as aid:
        from signed_study.stage_evidence import file_reference
        ledger.finish(aid,'failed',{'exception':reason},
                      {'calibration':file_reference(root/'calibration.json')} if published else {})
    save(root/'stages'/'calibrate.failed-fixture.json',dict(stage='calibrate',status='blocked',artifacts={},attempt_ids=[aid],stop_reason=reason))
    result=report.render_report(root)
    assert result['outcome']==('practical infeasibility' if published else 'inconclusive evidence')
    assert '[256, 1000]' in Path(result['report']).read_text()
    if published:
        record=json.loads((root/'calibration.json').read_text())
        record['forecast']['tested_scales']=[[1,1]]
        save(root/'calibration.json',record)
        changed=report.render_report(root)
        assert changed['outcome']=='inconclusive evidence'
        assert any('publication' in p for p in changed['limitations'])


def test_actual_work_accounts_all_stages_once_and_preserves_unknown(saved_evidence,monkeypatch):
    import csv
    from signed_study import report,confirmation,protocol_lock,diagnostics
    from signed_study.stage_evidence import file_reference
    root,ledger,lock,saved=saved_evidence
    monkeypatch.setattr(report,'_figures',lambda *a:[])
    def rows(statuses):
        return [dict(evaluation_index=i,user_id=i,slot='SignedMax',configuration_id='s',status=s) for i,s in enumerate(statuses)]
    def stage(name,stage_name,record,filename):
        path=root/filename
        with ledger.work(stage_name,0,len(record['attempts'])+len(record.get('pilot_attempts',[]))) as aid:
            save(path,record)
            refs={name if name!='develop' else 'development':file_reference(path)}
            if name=='calibrate':
                cache=root/'old-cache.npy';cache.write_bytes(b'old cache')
                refs['cache_vectors']=file_reference(cache)
            save(root/'stages'/f'{name}.json',dict(stage=name,status='completed',attempt_ids=[aid],
                artifacts=refs,stop_reason=None))
            if name=='calibrate': cache.unlink()  # historical report needs no installed cache
            ledger.finish(aid,'completed',{}, {name:file_reference(path)})
    stage('calibrate','G2',dict(status='completed',actual_evaluations=3,attempts=rows(['completed','completed','failed'])), 'calibration.json')
    stage('smoke','G3',dict(actual_evaluations=2,attempts=rows(['completed','failed'])),'smoke.json')
    main_rows=rows(['completed','completed','started'])
    main_rows[1]['evaluation_index']=0  # counters restart in each saved batch
    stage('develop','G3',dict(actual_evaluations=5,attempts=main_rows,
        pilot_attempts=rows(['completed','failed'])),'development.json')
    # An evaluation reservation without retained evidence must not become zero.
    with ledger.work('G3',0,7): pass
    digest,payload=protocol_lock.read_lock(lock)
    inputs=diagnostics._diagnostic_inputs(digest,payload,saved,confirmation.read_events(lock))
    timing=[dict(kind='timing',user_id=1,slot='SignedMax',repeat=i,warmup=i==0,status=s) for i,s in enumerate(['completed','completed','failed'])]
    numerical=[dict(kind='numerical',user_id=u,slot='SignedMax',status=s) for u,s in [(1,'completed'),(2,'unavailable')]]
    contamination=[dict(kind='contamination',user_id=1,slot='SignedMax',seed=1101,status='completed')]
    path=root/f'diagnostics-{digest}.json';journal=path.with_suffix('.jsonl')
    with ledger.work('G5',0,6) as aid:
        journal.write_text('')
        for r in [dict(timing[0],status='started'),timing[0],*numerical,*contamination]:
            r=dict(r,schema_version=1,lock_sha256=digest);r['output_sha256']=canonical_sha256(r)
            with journal.open('a') as f:f.write(json.dumps(r)+'\n')
        save(path,dict(lock_sha256=digest,confirmation_run_id=digest,confirmation_artifact=inputs['confirmation_artifact'],
            inputs=inputs,input_sha256=canonical_sha256(inputs),confirmation_status='completed',confirmation_interpretation='confirmatory',
            status='completed',attempt_ids=[aid],artifact_path=str(path),attempt_journal=file_reference(journal),
            numerical=numerical,timing={'attempts':timing},contamination={'status':'completed','attempts':contamination}))
        ledger.finish(aid,'completed',{}, {'diagnostics':file_reference(path)})
    # A corrected rerun is additional executed work; scientific rows remain its
    # current six evaluations, while original confirmation/diagnostics persist.
    confirmation._event(lock,'correction_authorized',correction_id='corrected',reason='fixture correction')
    with ledger.work('repair',0,6) as aid:
        clone=dict(saved,run_id='corrected',interpretation='corrected_exploratory',attempt_ids=[aid],artifact_path=str(root/'confirmation-corrected.json'))
        save(Path(clone['artifact_path']),clone)
        confirmation._event(lock,'run_started',run_id='corrected',ledger_attempt_id=aid)
        confirmation._event(lock,'run_finished',run_id='corrected',artifact=file_reference(clone['artifact_path']),interpretation='corrected_exploratory')
        ledger.finish(aid,'completed',{}, {'confirmation':file_reference(clone['artifact_path'])})
    with ledger.work('G4',0,2) as aid:
        confirmation._event(lock,'run_started',run_id='000-unavailable',ledger_attempt_id=aid)
        confirmation._event(lock,'run_finished',run_id='000-unavailable',
            artifact={'path':str(root/'lost-run.json'),'sha256':'0'*64},interpretation='invalid_confirmatory')
        ledger.finish(aid,'failed',{}, {})
    result=report.render_report(root)
    with next(Path(p) for p in result['tables'] if p.endswith('actual-work.csv')).open() as f:
        counts={r['work']:r for r in csv.DictReader(f)}
    assert (counts['calibration']['attempted'],counts['calibration']['completed'],counts['calibration']['failed'])==('3','2','1')
    assert counts['smoke']['attempted']=='2'
    assert (counts['development']['attempted'],counts['development']['pilot_attempted'])==('5','2')
    assert (counts['development']['completed'],counts['development']['failed'],counts['development']['unfinished'])==('3','1','1')
    assert counts['confirmation']['attempted']=='12'
    assert (counts['diagnostics']['attempted'],counts['diagnostics']['completed'],counts['diagnostics']['failed'],counts['diagnostics']['unavailable'])==('5','4','1','1')
    assert counts['unassigned_G3']['attempted']=='' and counts['unassigned_G3']['reserved']=='7'
    before=next(Path(p).read_bytes() for p in result['tables'] if p.endswith('actual-work.csv'))
    with ledger.work('G4',0,0): pass  # metadata/current-context read is not an evaluation
    report.render_report(root)
    assert next(Path(p).read_bytes() for p in result['tables'] if p.endswith('actual-work.csv'))==before


def test_alternate_stage_calibration_cannot_borrow_root_publication(saved_evidence):
    from signed_study import report
    from signed_study.stage_evidence import file_reference
    root,ledger,lock,_=saved_evidence
    lock.unlink();lock.with_suffix('.sha256').unlink()
    reason='Smallest permitted scale cannot fit remaining allocations and reserve'
    original=save(root/'calibration.json',dict(status='completed',actual_evaluations=1,
        forecast=dict(status='infeasible',tested_scales=[[256,1000]],stop_reason=reason)))
    root_bytes=(root/'calibration.json').read_bytes()
    with ledger.work('G2',0,1) as aid:
        ledger.finish(aid,'failed',{'exception':reason},{'calibration':file_reference(root/'calibration.json')})
    save(root/'stages'/'calibrate.failed-test.json',dict(stage='calibrate',status='blocked',artifacts={},attempt_ids=[aid],stop_reason=reason))
    alternate=root/'alternate'/'calibration.json'
    original['forecast']['tested_scales']=[[1,1]]
    save(alternate,original)
    save(root/'stages'/'calibrate.json',dict(stage='calibrate',status='completed',attempt_ids=[aid],
        artifacts={'calibration':file_reference(alternate)},stop_reason=None))
    result=report.render_report(root)
    assert (root/'calibration.json').read_bytes()==root_bytes
    assert result['outcome']=='inconclusive evidence'
    assert any('publication' in p for p in result['limitations'])
    # A differently spelled path to the same original file remains valid.
    save(root/'stages'/'calibrate.json',dict(stage='calibrate',status='completed',attempt_ids=[aid],
        artifacts={'calibration':file_reference((root/'calibration.json').resolve())},stop_reason=None))
    assert report.render_report(root)['outcome']=='practical infeasibility'


def test_corrupt_ledger_renders_invalid_report_without_mutation(local_config):
    import csv
    from signed_study import report
    root=local_config.paths['results'];root.mkdir(parents=True,exist_ok=True)
    path=root/'budget_ledger.jsonl';path.write_bytes(b'{corrupt accounting}\n')
    before=path.read_bytes()
    result=report.render_report(root)
    assert result['outcome']=='invalid experiment' and path.read_bytes()==before
    assert any('Corrupt ledger' in p for p in result['limitations'])
    with next(Path(p) for p in result['tables'] if p.endswith('actual-work.csv')).open() as f:
        rows=list(csv.DictReader(f))
    assert rows and all(r['attempted']=='' and r['status']=='unavailable' for r in rows)
    assert 'invalid' in Path(result['report']).read_text()


@pytest.mark.parametrize('missing',[0,1])
def test_saved_paired_se_and_complete_case_in_report_and_table(saved_evidence,missing):
    import csv
    from signed_study import report,confirmation
    from signed_study.stage_evidence import file_reference
    root,ledger,lock,saved=saved_evidence
    complete=dict(label='complete_case_diagnostic',n_users=2-missing,mean_difference=.03,
                  standard_error=.01 if not missing else None)
    saved['primary'].update(standard_error=.01 if not missing else None,complete_case=complete,n_missing=missing)
    saved['comparisons']['MeanNegative']=saved['primary']
    path=Path(saved['artifact_path'])
    with ledger.work('G4',0,0) as publication:
        saved['attempt_ids'].append(publication)
        save(path,saved)
        ledger.finish(publication,'completed',{}, {'confirmation':file_reference(path)})
    events=confirmation.read_events(lock);confirmation._events_path(lock).unlink()
    for e in events:
        fields={k:v for k,v in e.items() if k not in ('schema_version','output_sha256','previous_sha256','lock_sha256','kind')}
        if e['kind']=='run_finished':fields['artifact']=file_reference(path)
        confirmation._event(lock,e['kind'],**fields)
    rendered=report.render_report(root)
    with (root/'comparisons.csv').open() as f: row=next(csv.DictReader(f))
    assert row['comparator']=='MeanNegative'
    assert row['standard_error']==('' if missing else '0.01')
    assert json.loads(row['complete_case'])==complete
    text=Path(rendered['report']).read_text()
    assert 'complete_case_diagnostic' in text and 'standard_error' in text
    assert 'Primary decision: practical_advantage' in text
    assert 'user_ids' not in text
