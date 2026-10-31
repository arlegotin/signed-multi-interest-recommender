"""Synthetic saved-evidence packaging checks; no native bundle is required."""
import csv
import importlib.util
import json
import subprocess
import sys
import shutil
from pathlib import Path

import pytest


@pytest.fixture
def packaging():
    path=Path(__file__).resolve().parents[1]/'scripts/render_results_first_study.py'
    spec=importlib.util.spec_from_file_location('saved_packaging',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def write_csv(path,fields,rows):
    with path.open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writer.writerows(rows)


@pytest.fixture
def exports(tmp_path):
    methods=['SignedMax',*[f'comparator_{i}' for i in range(12)]]
    metric=dict(n_complete=2,n_scheduled=2,auc=.6,ndcg10=.7,negative_at_10=.2,tie_fraction=0.)
    comparison=dict(sample_mean_low=.1,sample_mean_high=.1,low=.05,high=.15,
                    n_users=2,n_complete=2,n_missing=0,standard_error=.02)
    work=dict(work='confirmation',reserved=30,attempted=26,completed=25,failed=1,
              unfinished=0,unavailable=0,pilot_attempted=None,status='available',evidence='bound fixture')
    summary=dict(metric_means={name:dict(metric) for name in methods},
                 comparisons={name:dict(comparison) for name in methods[1:]},
                 attempted_evaluations=30,actual_work=[work])
    metric_rows=[dict(method=name,n_completed=2,n_scheduled=2,
                      **{k:metric[k] for k in ('auc','ndcg10','negative_at_10','tie_fraction')}) for name in methods]
    diagnostic=dict(mean_difference=.1,standard_error=.02,n_users=2,label='complete_case_diagnostic')
    comparison_rows=[dict(comparator=name,**comparison,complete_case=json.dumps(diagnostic)) for name in methods[1:]]
    write_csv(tmp_path/'metrics.csv',list(metric_rows[0]),metric_rows)
    write_csv(tmp_path/'comparisons.csv',list(comparison_rows[0]),comparison_rows)
    write_csv(tmp_path/'actual-work.csv',list(work),[work])
    return tmp_path,summary,metric_rows,comparison_rows,work


def test_exports_compare_actual_starts_separately_from_reservations(packaging,exports):
    root,summary,*_=exports
    packaging.verify_csvs(root,summary)


@pytest.mark.parametrize('damage',['methods','comparators','work','counters','metadata','schema','nonfinite'])
def test_exports_require_exact_identity_and_bound_work(packaging,exports,damage):
    root,summary,metrics,comparisons,work=exports
    # Retain the old validator's single-total equality while corrupting identity.
    summary['attempted_evaluations']=26
    if damage=='methods': metrics=[metrics[0]]*13
    if damage=='comparators': comparisons=[comparisons[0]]*12
    if damage=='work': work=dict(work,work='invented',completed=100,failed=100)
    if damage=='counters': work=dict(work,completed=100,failed=100)
    if damage=='metadata': comparisons[0]['complete_case']=json.dumps(dict(label='invented',n_users=99,mean_difference=999,standard_error=999))
    if damage=='schema': metrics[0]['extra']='ignored'
    if damage=='nonfinite':
        metrics[0]['auc']='nan';summary['metric_means']['SignedMax']['auc']=float('nan')
    write_csv(root/'metrics.csv',list(metrics[0]),metrics)
    write_csv(root/'comparisons.csv',list(comparisons[0]),comparisons)
    write_csv(root/'actual-work.csv',list(work),[work])
    with pytest.raises(ValueError): packaging.verify_csvs(root,summary)


def test_rejected_prior_receipt_does_not_touch_publications(packaging,exports,monkeypatch):
    root,summary,*_=exports
    docs=root/'docs';docs.mkdir();published=docs/'published.csv';published.write_bytes(b'valid prior export')
    report=root/'report.md';report.write_bytes(b'valid prior report')
    for name in ('paired-intervals.png','paired-differences.png'):(root/name).write_bytes(b'fixture figure')
    (root/'results-first-reproduction.json').write_text('{}')
    monkeypatch.setattr(packaging,'summary_snapshot',lambda *a:(summary,{}))
    def export(*args):published.write_bytes(b'bad replacement');return [published]
    monkeypatch.setattr(packaging,'public_exports',export)
    monkeypatch.setattr(packaging,'report_addendum',lambda *a:report.write_bytes(b'bad report'))
    with pytest.raises(ValueError):packaging.package(root,docs,None,'fixture')
    assert published.read_bytes()==b'valid prior export'
    assert report.read_bytes()==b'valid prior report'


@pytest.mark.parametrize('failure',['replacement','selector','terminal'])
def test_recoverable_publication_restores_old_files_on_failure(tmp_path,monkeypatch,failure):
    from signed_study import publication
    root=tmp_path/'results';root.mkdir()
    targets=[root/'report.md',root/'artifact-inventory.json']
    selector=root/'report.active.json'
    for path in [*targets,selector]:path.write_bytes(b'previous '+path.name.encode())
    old={path:path.read_bytes() for path in [*targets,selector]}
    sources=[]
    for i in range(2):
        source=root/f'prospective-{i}';source.write_bytes(b'new '+str(i).encode());sources.append(source)
    real=publication._replace_from
    calls=[0]
    def replace(source,target):
        calls[0]+=1
        if (failure=='replacement' and calls[0]==2 or failure=='selector' and target==selector):
            raise OSError('injected replacement failure')
        real(source,target)
    monkeypatch.setattr(publication,'_replace_from',replace)
    class Accounting:
        def snapshot(self):return {'attempts':{'owner':{'status':'started'}}}
        def finish(self,*args):raise OSError('injected terminal accounting failure')
    with pytest.raises(OSError):
        publication.publish_files(root,'fixture',dict(zip(targets,sources)),selector,{'receipt':'new'},
                                  ledger=Accounting() if failure=='terminal' else None,
                                  owner='owner' if failure=='terminal' else None,outputs={'fixture':'new'})
    monkeypatch.setattr(publication,'_replace_from',real)
    publication.recover_publication(root,'fixture')
    assert {path:path.read_bytes() for path in old}==old


def test_bundle_ancestor_links_are_durable_before_transaction_publication(tmp_path,monkeypatch):
    from signed_study import publication
    root=tmp_path/'results';root.mkdir()
    prospective=tmp_path/'prospective';prospective.write_bytes(b'validated new report')
    synced=[]
    real_sync=publication._sync_directory
    real_write=publication.atomic_write_json
    def sync(path):
        synced.append(Path(path).resolve())
        return real_sync(path)
    def write(path,value):
        if Path(path)==root/'.fixture.publication.json' and value.get('phase')=='prepared':
            assert root/'.publications' in synced, 'bundle parent must be durable before transaction'
            assert root in synced, 'bundle ancestry must be durable before transaction'
        return real_write(path,value)
    monkeypatch.setattr(publication,'_sync_directory',sync)
    monkeypatch.setattr(publication,'atomic_write_json',write)
    publication.publish_files(root,'fixture',{root/'report.md':prospective},root/'active.json',{'receipt':'new'})
    assert (root/'report.md').read_bytes()==b'validated new report'


def test_rendering_error_preserves_previous_aggregate_bundle(local_config,monkeypatch):
    from signed_study import report
    root=local_config.paths['results'];root.mkdir(exist_ok=True)
    first=report.render_report(root)
    paths=[Path(first['report']),Path(first['inventory']),*map(Path,first['tables'])]
    old={p:p.read_bytes() for p in paths}
    real=report._table
    calls=[0]
    def fail(path,*args):
        calls[0]+=1
        if calls[0]==2:raise OSError('injected rendering error')
        result=real(path,*args)
        path.write_bytes(path.read_bytes()+b'prospective changed costs')
        return result
    monkeypatch.setattr(report,'_table',fail)
    with pytest.raises(OSError):report.render_report(root)
    assert {p:p.read_bytes() for p in paths}==old


@pytest.mark.parametrize('terminal',['before','after'])
@pytest.mark.parametrize('recovery',['api','command'])
def test_process_exit_between_selector_and_terminal_accounting_is_recoverable(local_config,terminal,recovery,packaging,monkeypatch,capsys):
    from signed_study.budget import Ledger
    from signed_study.publication import recover_publication
    from signed_study.stage_evidence import config_record
    root=local_config.paths['results'].resolve();root.mkdir(exist_ok=True)
    config_path=root/'fixture-config.json';config_path.write_text(json.dumps(config_record(local_config)))
    (root/'report.md').write_bytes(b'old report')
    (root/'active.json').write_bytes(b'old selector')
    (root/'prospective').write_bytes(b'new report')
    name='fixture' if recovery=='api' else 'results-first'
    code='''
import json, os, sys
from pathlib import Path
from signed_study.budget import Ledger
from signed_study.protocol import config_from_record
from signed_study.publication import publish_files
root=Path(sys.argv[1]);config=config_from_record(json.loads((root/'fixture-config.json').read_text()))
sample=dict(rss_bytes=1,installed_ram_bytes=64*2**30,available_memory_bytes=8*2**30,project_disk_bytes=1,free_disk_bytes=20*2**30)
ledger=Ledger(root/'budget_ledger.jsonl',config,clock=lambda:0.,sampler=lambda:sample,monitor=False)
with ledger.work('G6',0,0) as owner:
    finish=ledger.finish
    def interrupted(*args):
        if sys.argv[2]=='after':finish(*args)
        os._exit(19)
    ledger.finish=interrupted
    publish_files(root,sys.argv[3],{root/'report.md':root/'prospective'},root/'active.json',{'receipt':'fixture'},
                  ledger=ledger,owner=owner,outputs={'bundle':'fixture'})
'''
    completed=subprocess.run([sys.executable,'-c',code,str(root),terminal,name],capture_output=True,text=True)
    assert completed.returncode==19,completed.stderr
    # The process died after selector replacement, without finally/cleanup.
    assert (root/'report.md').read_bytes()==b'new report'
    assert (root/f'.{name}.publication.json').exists()
    ledger=Ledger(root/'budget_ledger.jsonl',local_config,monitor=False)
    try:
        ledger_bytes=ledger.path.read_bytes()
        if recovery=='api':recovered=recover_publication(root,name,ledger=ledger)
        else:
            monkeypatch.setattr(packaging,'report_config',lambda root:local_config)
            packaging.main(['--results',str(root),'--docs',str(root/'docs'),'--recover-only'])
            recovered=json.loads(capsys.readouterr().out)['publication_state']
        assert ledger.path.read_bytes()==ledger_bytes
        attempts=list(ledger.snapshot()['attempts'].values())
    finally:ledger.close()
    if terminal=='before':
        assert recovered=='restored' and attempts[0]['status']!='completed'
        assert (root/'report.md').read_bytes()==b'old report'
        assert (root/'active.json').read_bytes()==b'old selector'
    else:
        assert recovered=='committed' and attempts[0]['status']=='completed'
        assert (root/'report.md').read_bytes()==b'new report'
    assert not (root/f'.{name}.publication.json').exists()
    assert list((root/'.publications').glob(f'{name}-*/manifest.json'))


@pytest.fixture
def packaging_campaign(packaging,exports,local_config,monkeypatch):
    from signed_study.budget import Ledger
    from signed_study.stage_evidence import file_reference
    source,summary,*_=exports
    root=local_config.paths['results'].resolve();root.mkdir(exist_ok=True)
    docs=source/'published';docs.mkdir()
    for name in ('metrics.csv','comparisons.csv','actual-work.csv'):shutil.copyfile(source/name,root/name)
    for name in ('paired-intervals.png','paired-differences.png'):(root/name).write_bytes(b'fixed synthetic figure')
    (root/'report.md').write_text('saved synthetic report\n')
    (root/'artifact-inventory.json').write_text(json.dumps(dict(schema_version=1,artifacts=[dict(name='report.md',status='generated aggregate export',sha256=file_reference(root/'report.md')['sha256'])])))
    summary['environment_publication']=dict(attempt_id='fixture',reference={})
    summary_path=root/'summary.json';summary_path.write_text('fixed saved synthetic summary')
    monkeypatch.setattr(packaging,'summary_snapshot',lambda *args:(summary,file_reference(summary_path)))
    def export(staged_root,staged_docs,comparisons):
        staged_docs.mkdir(exist_ok=True)
        target=staged_docs/'fixture-metrics.csv';shutil.copyfile(staged_root/'metrics.csv',target)
        return [target]
    monkeypatch.setattr(packaging,'public_exports',export)
    def addendum(staged_root,summary):
        (staged_root/'report.md').write_text('saved synthetic report\nvalidated synthetic addendum\n')
    monkeypatch.setattr(packaging,'report_addendum',addendum)
    sample=dict(rss_bytes=1,installed_ram_bytes=64*2**30,available_memory_bytes=8*2**30,project_disk_bytes=1,free_disk_bytes=20*2**30)
    ledger=Ledger(root/'budget_ledger.jsonl',local_config,clock=lambda:0.,sampler=lambda:sample,monitor=False)
    yield root,docs,ledger
    ledger.close()


def test_packaging_retains_old_receipts_and_exact_scientific_exports(packaging,packaging_campaign):
    from signed_study.stage_evidence import file_reference,verify_record,verify_reference
    root,docs,ledger=packaging_campaign
    with ledger.work('G6',0,0) as first_owner:first=packaging.package(root,docs,ledger,first_owner)
    old_receipt=file_reference(first['receipt'])
    old_bytes=Path(first['receipt']).read_bytes()
    with ledger.work('G6',0,0) as second_owner:second=packaging.package(root,docs,ledger,second_owner)
    assert second['repeated_scientific_hashes_equal'] is True
    assert first['scientific_hashes']==second['scientific_hashes']
    verify_reference(old_receipt)
    assert Path(first['receipt']).read_bytes()==old_bytes
    pointer=json.loads((root/'results-first-reproduction.json').read_text());verify_record(pointer)
    owner=ledger.snapshot()['attempts'][second_owner]
    assert owner['status']=='completed' and owner['outputs']['results_first_packaging']==pointer['receipt']
    assert len(list((root/'.publications').glob('results-first-*/manifest.json')))==2


@pytest.mark.parametrize('damage',['deleted','stale'])
def test_packaging_rejects_missing_or_stale_committed_selector(packaging,packaging_campaign,damage):
    root,docs,ledger=packaging_campaign
    with ledger.work('G6',0,0) as owner:packaging.package(root,docs,ledger,owner)
    pointer=root/'results-first-reproduction.json';old_pointer=pointer.read_bytes()
    if damage=='deleted':pointer.unlink()
    else:
        with ledger.work('G6',0,0) as owner:packaging.package(root,docs,ledger,owner)
        pointer.write_bytes(old_pointer)
    old_report=(root/'report.md').read_bytes()
    with pytest.raises(ValueError,match='selector'):
        with ledger.work('G6',0,0) as owner:packaging.package(root,docs,ledger,owner)
    assert (root/'report.md').read_bytes()==old_report


@pytest.mark.parametrize('damage',['selector','report','inventory'])
def test_report_refuses_corrupt_prior_publication(local_config,damage):
    from signed_study import report
    root=local_config.paths['results'];root.mkdir(exist_ok=True)
    report.render_report(root)
    path=root/{'selector':'report.active.json','report':'report.md','inventory':'artifact-inventory.json'}[damage]
    path.write_bytes(b'corrupt prior publication')
    before={p:p.read_bytes() for p in (root/'report.md',root/'artifact-inventory.json',root/'report.active.json')}
    with pytest.raises(ValueError):report.render_report(root)
    assert {p:p.read_bytes() for p in before}==before


@pytest.mark.parametrize('failure',['render','validation','replacement','pointer','terminal'])
def test_failed_packaging_preserves_completed_bundle(packaging,packaging_campaign,monkeypatch,failure):
    from signed_study import publication
    root,docs,ledger=packaging_campaign
    with ledger.work('G6',0,0) as first_owner:first=packaging.package(root,docs,ledger,first_owner)
    paths=[root/'report.md',root/'artifact-inventory.json',root/'report.active.json',
           root/'results-first-reproduction.json',Path(first['receipt']),*docs.rglob('*.csv')]
    before={path:path.read_bytes() for path in paths}
    if failure=='render':
        def fail(*a):raise OSError('injected renderer failure')
        monkeypatch.setattr(packaging,'public_exports',fail)
    elif failure=='validation':
        real=packaging.verify_csvs;calls=[0]
        def fail(*args):
            calls[0]+=1
            if calls[0]==2:raise OSError('injected staged validation failure')
            return real(*args)
        monkeypatch.setattr(packaging,'verify_csvs',fail)
    elif failure in ('replacement','pointer'):
        real=publication._replace_from;fired=[False]
        target=root/('report.active.json' if failure=='replacement' else 'results-first-reproduction.json')
        def fail(source,destination):
            if destination==target and not fired[0]:
                fired[0]=True;raise OSError('injected projection failure')
            return real(source,destination)
        monkeypatch.setattr(publication,'_replace_from',fail)
    else:
        real=ledger.finish
        def fail(owner,status,measurements,outputs):
            if outputs.get('results_first_packaging'):raise OSError('injected terminal accounting failure')
            return real(owner,status,measurements,outputs)
        monkeypatch.setattr(ledger,'finish',fail)
    with pytest.raises(OSError):
        with ledger.work('G6',0,0) as failed_owner:packaging.package(root,docs,ledger,failed_owner)
    assert {path:path.read_bytes() for path in before}==before
    failed=ledger.snapshot()['attempts'][failed_owner]
    assert failed['status']=='failed' and 'results_first_packaging' not in failed['outputs']
    assert ledger.snapshot()['attempted_evaluations']==0
