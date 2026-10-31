"""Explicit pre-freeze continuation without repeating scientific work."""
import json
from pathlib import Path
from types import SimpleNamespace
from dataclasses import asdict, replace
import hashlib

import numpy as np
import pytest

from signed_study.budget import BudgetExceeded, atomic_write_json
from signed_study.protocol import canonical_sha256
from signed_study.stage_evidence import file_reference
from test_budget_transfers import transfer_case
from test_calibration import cohort, measured
from test_stage_pipeline import pipeline, prepare
from test_protocol_lock import frozen_study, full_cohort_schedule, ready_to_freeze


def adopt(ledger):
    assert hasattr(ledger, 'adopt_results_first'), 'durable results-first adoption is missing'
    return ledger.adopt_results_first('User authorized completion of scientific results on 31 October')


def test_cli_freeze_uses_superseded_development_receipt(ready_to_freeze, monkeypatch, capsys):
    from signed_study import __main__ as cli, budget, protocol, protocol_lock
    from signed_study.stage_evidence import read_stage, save_stage, stage_path
    config, ledger, _, _ = ready_to_freeze
    original = file_reference(stage_path(config, 'develop'))
    old = read_stage(config, 'develop')
    with ledger.work('G3', 0, 0) as aid:
        current = save_stage(config, 'develop', old['inputs'],
            {k:Path(v['path']) for k,v in old['artifacts'].items()}, [aid],
            supersedes=original, complete_suite_eligible=True, confirmation_outcomes_accessed=False)
    monkeypatch.setattr(protocol, 'load_config', lambda p:config)
    monkeypatch.setattr(budget, 'Ledger', lambda *a,**kw:ledger)
    assert cli.main(['freeze']) == 0
    result = json.loads(capsys.readouterr().out)
    _, lock = protocol_lock.read_lock(Path(result['lock']))
    assert lock['identities']['artifacts']['stage_develop'] == file_reference(stage_path(config, 'develop'))
    assert lock['identities']['artifacts']['stage_develop'] != original
    assert current['supersedes'] == original
    assert file_reference(original['path']) == original


@pytest.mark.parametrize('damage', ['deleted', 'stale', 'corrupt'])
def test_committed_supersession_selector_fails_closed(pipeline, damage):
    from signed_study.stage_evidence import read_stage, save_stage, stage_path
    config, ledger, *_ = pipeline
    with ledger.work('G3', 0, 0) as aid:
        save_stage(config, 'smoke', {}, {}, [aid])
    previous = file_reference(stage_path(config, 'smoke'))
    with ledger.work('G3', 0, 0) as aid:
        save_stage(config, 'smoke', {'revision':2}, {}, [aid], supersedes=previous)
    selector = config.paths['results']/'stages'/'smoke.active.json'
    if damage == 'deleted': selector.unlink()
    elif damage == 'corrupt': selector.write_text('{}')
    else:
        value = {'schema_version':1, 'stage':'smoke', 'receipt':previous}
        value['output_sha256'] = canonical_sha256(value)
        atomic_write_json(selector, value)
    with pytest.raises(ValueError, match='selector|selection|receipt|digest|reference'):
        read_stage(config, 'smoke')
    assert file_reference(previous['path']) == previous


def test_orphaned_receipt_does_not_supersede_legacy(pipeline, monkeypatch):
    from signed_study import stage_evidence as evidence
    config, ledger, *_ = pipeline
    with ledger.work('G3', 0, 0) as aid:
        original = evidence.save_stage(config, 'smoke', {}, {}, [aid])
    previous = file_reference(evidence.stage_path(config, 'smoke'))
    write = evidence.atomic_write_json
    def interrupted(path, value):
        if Path(path).name == 'smoke.active.json':
            raise OSError('injected selector failure')
        return write(path, value)
    monkeypatch.setattr(evidence, 'atomic_write_json', interrupted)
    with pytest.raises(OSError, match='selector failure'):
        with ledger.work('G3', 0, 0) as aid:
            evidence.save_stage(config, 'smoke', {'revision':2}, {}, [aid], supersedes=previous)
    with pytest.warns(RuntimeWarning, match='interrupted'):
        assert evidence.read_stage(config, 'smoke') == original
    assert file_reference(previous['path']) == previous


@pytest.fixture
def access_fallback_pipeline(pipeline, monkeypatch):
    """Inject only archive parsing/encoding; keep stage, hashes and ledger real.

    A tiny local archive stands in for pinned bytes via an explicit source-table
    test double. No production package or hardware evidence is manufactured.
    """
    import numpy as np
    from signed_study import data
    from signed_study.data import Dataset, EpisodeStore, Movie, episode_store_digest
    config, ledger, clock, resources, events, stages, evidence, _ = pipeline
    raw = config.paths['data']/'raw'/'ml-100k.zip'
    raw.write_bytes(b'synthetic official-parser boundary')
    monkeypatch.setitem(data.SOURCES, 'ml-100k', replace(data.SOURCES['ml-100k'],
        sha256=hashlib.sha256(raw.read_bytes()).hexdigest(), size=raw.stat().st_size))
    attempts = [dict(kind=kind, url=url, status=status,
        started_at='2026-10-31T00:00:00+00:00', ended_at='2026-10-31T00:00:01+00:00')
        for kind,url,status in [('primary',data.SOURCES['ml-1m'].url,'unavailable'),
            ('primary',data.SOURCES['ml-1m'].url,'unavailable'),
            ('official_page',data._OFFICIAL_PAGE,'retrieved'),
            ('fallback',data.SOURCES['ml-100k'].url,'retrieved')]]
    acquisition = dict(schema_version=1,attempts=attempts,
        input_sha256=canonical_sha256({name:asdict(source) for name,source in data.SOURCES.items()}),
        output_sha256=canonical_sha256({'attempts':attempts}))
    atomic_write_json(config.paths['results']/'data_acquisition.json', acquisition)
    files = {name:dict(sha256=hashlib.sha256(name.encode()).hexdigest(),bytes=len(name))
             for name in ('u.data','u.item','u.genre','README')}
    manifest = dict(schema_version=1,dataset='ml-100k',status='validated',access_fallback=True,
        archive_sha256=data.SOURCES['ml-100k'].sha256, archive_bytes=raw.stat().st_size,
        source_url=data.SOURCES['ml-100k'].url, files=files,
        counts={**data._EXPECTED_COUNTS['ml-100k'],'positive_ratings':55000,'negative_ratings':18000,'neutral_ratings':27000},
        discrepancies={}, input_sha256=data.SOURCES['ml-100k'].sha256,
        acquisition_attempts_sha256=acquisition['output_sha256'])
    manifest['output_sha256'] = canonical_sha256({'files':files,'counts':manifest['counts']})
    metadata = {i:Movie(i,f'Movie {i}',('Drama',)) for i in range(256)}
    dataset = Dataset((),metadata,manifest)
    initial = cohort(62,225,'ml-100k')
    root = config.paths['data']/'derived'/'fallback-fixture'
    paths = {'audit':config.paths['results']/'cohort_audit.json','episodes':root/'episodes.json',
        **{role+'_outcomes':root/'outcomes'/role for role in ('development','confirmation')}}
    audit = dict(schema_version=1,dataset='ml-100k',status='validated',input_sha256='a'*64)
    store = EpisodeStore(initial.development,initial.confirmation,audit,paths)
    audit['episodes_sha256'] = episode_store_digest(store)
    records = {'schema_version':1, **{role:[asdict(e) for e in sorted(getattr(store,role),key=lambda e:e.history.user_id)]
        for role in ('development','confirmation')}}
    atomic_write_json(paths['episodes'],{**records,'input_sha256':'a'*64,'output_sha256':audit['episodes_sha256']})
    outcomes = {'development':{},'confirmation':{}}
    for role in outcomes:
        paths[role+'_outcomes'].mkdir(parents=True,exist_ok=True)
        for episode in getattr(store,role):
            path = paths[role+'_outcomes']/f'{episode.history.user_id}.npy'
            np.save(path,np.arange(len(episode.candidate_ids),dtype=np.int8)%2)
            outcomes[role][str(episode.history.user_id)] = hashlib.sha256(path.read_bytes()).hexdigest()
    audit['outputs'] = dict(episodes_file_sha256=hashlib.sha256(paths['episodes'].read_bytes()).hexdigest(),outcome_sha256=outcomes)
    audit['output_sha256'] = canonical_sha256(audit['outputs'])
    atomic_write_json(paths['audit'],audit)
    monkeypatch.setattr(stages,'load_dataset',lambda c:dataset)
    monkeypatch.setattr(stages,'make_episodes',lambda d,c:store)
    return config,ledger,clock,resources,events,stages,evidence,store


def test_authorized_access_fallback_full_cohort_reaches_freeze(access_fallback_pipeline, monkeypatch, capsys):
    from signed_study import __main__ as cli, budget, protocol, protocol_lock, workload
    config, ledger, _, _, _, stages, _, store = access_fallback_pipeline
    prepare(access_fallback_pipeline)
    adopt(ledger)
    calibrated = stages.run_stage('calibrate',config,ledger)
    assert calibrated['status'] == 'completed', calibrated
    assert calibrated['forecast']['scale'] == [62,225]
    monkeypatch.setattr(stages,'measure_users',workload.measure_users)
    assert stages.run_stage('smoke',config,ledger)['status'] == 'completed'
    developed = stages.run_stage('develop',config,ledger)
    assert developed['status'] == 'completed', developed
    monkeypatch.setattr(protocol_lock,'source_identity',lambda:dict(commit='a'*40,tree='b'*40,clean=True,content={'module.py':'c'*64}))
    monkeypatch.setattr(protocol,'load_config',lambda p:config)
    monkeypatch.setattr(budget,'Ledger',lambda *a,**kw:ledger)
    assert cli.main(['freeze']) == 0
    _, payload = protocol_lock.read_lock(Path(json.loads(capsys.readouterr().out)['lock']))
    assert payload['invocation_config']['dataset'] == 'ml-1m'
    assert payload['config']['dataset'] == 'ml-1m'
    assert len(payload['development']['development_user_ids']) == len(store.development) == 62
    assert len(payload['data_access']['confirmation_user_ids']) == len(store.confirmation) == 225
    assert payload['identities']['artifacts']['stage_acquire']['sha256']


@pytest.mark.parametrize('damage', ['primary_count','primary_status','page_url','page_status','digest','archive','schema','unbound',
    'rating_total','negative_class_count','fractional_class_count','missing_class_count'])
def test_access_fallback_rejects_invalid_access_or_archive_evidence(access_fallback_pipeline, damage):
    config, ledger, _, _, _, stages, _, _ = access_fallback_pipeline
    path = config.paths['results']/'data_acquisition.json'
    value = json.loads(path.read_text())
    if damage == 'primary_count': value['attempts'].pop(0)
    elif damage == 'primary_status': value['attempts'][0]['status'] = 'retrieved'
    elif damage == 'page_url': value['attempts'][2]['url'] = 'https://example.test/unofficial'
    elif damage == 'page_status': value['attempts'][2]['status'] = 'started'
    elif damage == 'digest': value['output_sha256'] = 'f'*64
    elif damage == 'archive': (config.paths['data']/'raw'/'ml-100k.zip').write_bytes(b'changed')
    elif damage == 'schema':
        dataset = stages.load_dataset(config)
        dataset.manifest['files'].pop('u.genre')
    elif damage in ('rating_total','negative_class_count','fractional_class_count','missing_class_count'):
        manifest = stages.load_dataset(config).manifest
        if damage=='rating_total': manifest['counts']['positive_ratings'] += 1
        elif damage=='negative_class_count': manifest['counts']['negative_ratings'] = -1
        elif damage=='fractional_class_count': manifest['counts']['neutral_ratings'] = 27000.5
        else: manifest['counts'].pop('neutral_ratings')
        manifest['output_sha256'] = canonical_sha256({'files':manifest['files'],'counts':manifest['counts']})
    else: path.unlink()
    if damage not in ('unbound','archive','schema','rating_total','negative_class_count','fractional_class_count','missing_class_count'):
        if damage!='digest':
            value['output_sha256'] = canonical_sha256({'attempts':value['attempts']})
            stages.load_dataset(config).manifest['acquisition_attempts_sha256'] = value['output_sha256']
        atomic_write_json(path,value)
    result = stages.run_stage('acquire',config,ledger)
    assert result['status'] == 'blocked', result


def test_access_fallback_accepts_full_real_parser_count_schema(access_fallback_pipeline):
    config,ledger,_,_,_,stages,_,_ = access_fallback_pipeline
    counts = stages.load_dataset(config).manifest['counts']
    assert len(counts)==7
    assert sum(counts[key] for key in ('positive_ratings','negative_ratings','neutral_ratings'))==100000
    result=stages.run_stage('acquire',config,ledger)
    assert result['status']=='completed',result


@pytest.mark.parametrize('damage', ['truncated','reordered','duplicate','foreign','bool','episodes','audit','failed_owner'])
def test_fallback_full_cohort_membership_and_binding_fail_closed(access_fallback_pipeline, monkeypatch, damage):
    from signed_study.calibration import require_full_cohort, user_digest
    config, ledger, _, _, _, stages, _, store = access_fallback_pipeline
    assert stages.run_stage('acquire',config,ledger)['status'] == 'completed'
    assert stages.run_stage('audit',config,ledger)['status'] == 'completed'
    adopt(ledger)
    planned = {role:sorted([e.history.user_id for e in getattr(store,role)],key=user_digest)
               for role in ('development','confirmation')}
    # No full-cohort authorization or binding check may expose outcome arrays.
    import numpy as np
    monkeypatch.setattr(np,'load',lambda *a,**kw:pytest.fail('fallback validation opened outcomes'))
    state = ledger.snapshot()
    require_full_cohort(config,store,planned,state)
    if damage == 'truncated': planned['development'].pop()
    elif damage == 'reordered': planned['confirmation'].reverse()
    elif damage == 'duplicate': planned['development'][0] = planned['development'][1]
    elif damage == 'foreign': planned['confirmation'][0] = 999999
    elif damage == 'bool': planned['development'][planned['development'].index(1)] = True
    elif damage == 'episodes':
        episode = store.development[0]
        changed = replace(episode, candidate_ids=(*episode.candidate_ids,255))
        store = replace(store,development=(changed,*store.development[1:]))
    elif damage == 'audit': store.audit['input_sha256'] = 'f'*64
    else:
        from signed_study.stage_evidence import read_stage
        acquired = read_stage(config,'acquire')
        state['attempts'][acquired['attempt_ids'][-1]]['status'] = 'failed'
    with pytest.raises(ValueError): require_full_cohort(config,store,planned,state)


def test_legacy_completed_supersession_without_witness_cannot_reactivate_old_receipt(pipeline):
    from signed_study.stage_evidence import save_stage, read_stage, stage_path
    config, ledger, *_ = pipeline
    with ledger.work('G3',0,0) as aid: save_stage(config,'smoke',{}, {},[aid])
    previous = file_reference(stage_path(config,'smoke'))
    with ledger.work('G3',0,0) as aid:
        save_stage(config,'smoke',{'revision':2},{},[aid],supersedes=previous)
    root = config.paths['results']/'stages'
    # Historical supersession did not have the new selection witness. Its
    # replacement + completed ledger attempts still prove that legacy is stale.
    (root/'smoke.selection.json').unlink()
    (root/'smoke.active.json').unlink()
    with pytest.raises(ValueError,match='missing active selector'): read_stage(config,'smoke')


@pytest.mark.parametrize('damage', ['unbound','foreign','invocation','archive','audit_receipt'])
def test_primary_full_cohort_requires_bound_acquisition_and_episodes(pipeline, damage):
    from signed_study.calibration import require_full_cohort, user_digest
    config, ledger, _, _, _, stages, _, store = pipeline
    assert stages.run_stage('acquire',config,ledger)['status'] == 'completed'
    assert stages.run_stage('audit',config,ledger)['status'] == 'completed'
    adopt(ledger)
    planned = {role:sorted([e.history.user_id for e in getattr(store,role)],key=user_digest)
               for role in ('development','confirmation')}
    require_full_cohort(config,store,planned,ledger.snapshot())
    if damage == 'unbound': store = replace(store,paths={})
    elif damage == 'foreign':
        first = store.development[0]
        foreign = replace(first,history=replace(first.history,user_id=999999))
        store = replace(store,development=(foreign,*store.development[1:]))
        planned['development'] = sorted([e.history.user_id for e in store.development],key=user_digest)
    elif damage == 'invocation': config = replace(config,dataset='ml-100k')
    elif damage == 'archive': (config.paths['data']/'raw'/'ml-1m.zip').write_bytes(b'other same-name archive')
    else:
        path = config.paths['results']/'stages'/'audit.json'
        record = json.loads(path.read_text())
        record['inputs']['acquire'] = 'f'*64
        record['input_sha256'] = canonical_sha256(record['inputs'])
        record.pop('output_sha256')
        record['output_sha256'] = canonical_sha256(record)
        atomic_write_json(path,record)
    with pytest.raises(ValueError): require_full_cohort(config,store,planned,ledger.snapshot())


def test_historical_stale_selector_rejects_completed_successor(pipeline):
    from signed_study.stage_evidence import save_stage, read_stage, stage_path
    config, ledger, *_ = pipeline
    root = config.paths['results']/'stages'
    with ledger.work('G3',0,0) as aid: save_stage(config,'smoke',{}, {},[aid])
    with ledger.work('G3',0,0) as aid:
        save_stage(config,'smoke',{'revision':2},{},[aid],supersedes=file_reference(stage_path(config,'smoke')))
    prior = (root/'smoke.active.json').read_bytes()
    with ledger.work('G3',0,0) as aid:
        save_stage(config,'smoke',{'revision':3},{},[aid],supersedes=file_reference(stage_path(config,'smoke')))
    (root/'smoke.selection.json').unlink()
    (root/'smoke.active.json').write_bytes(prior)
    with pytest.raises(ValueError,match='stale.*selector|selector.*stale'):
        read_stage(config,'smoke')


def test_policy_preserves_history_and_is_durable_idempotent(transfer_case):
    ledger, clock, reopen = transfer_case
    with ledger.work('G2', 0, 7): clock.now += 2
    before = ledger.snapshot()
    prefix = ledger.path.read_bytes()
    event_id = adopt(ledger)
    assert adopt(ledger) == event_id
    state = reopen().snapshot()
    assert state['execution_policy']['mode'] == 'results_first'
    assert state['execution_policy']['event_id'] == event_id
    assert ledger.path.read_bytes().startswith(prefix)
    assert len(state['events']) == len(before['events']) + 1
    for key in ('resources_sha256', 'stage_limits', 'stage_seconds', 'experimental_seconds',
                'setup_seconds', 'repair_cycles', 'attempted_evaluations', 'attempts'):
        assert state[key] == before[key]


def test_results_first_waives_all_clocks_and_material_cycle_quota(transfer_case):
    ledger, clock, _ = transfer_case
    assert ledger.snapshot().get('execution_policy', {}).get('mode', 'bounded') == 'bounded'
    with pytest.raises(BudgetExceeded): ledger.admit('G1', 1e6, 0)
    adopt(ledger)
    for stage in ('setup', 'G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'repair', 'repair', 'repair'):
        with ledger.work(stage, 1e6, 0):
            clock.now += 1e6
            ledger.checkpoint()
    assert ledger.snapshot()['repair_cycles'] == 3
    assert ledger.snapshot()['experimental_seconds'] == 9e6


@pytest.mark.parametrize('fault', ['evaluations', 'rss', 'disk', 'memory', 'free_disk', 'duration'])
def test_results_first_retains_hard_resources(transfer_case, fault):
    ledger, _, _ = transfer_case
    adopt(ledger)
    original = ledger.sampler
    sample = original()
    if fault == 'rss': sample['rss_bytes'] = 20*1024**3
    if fault == 'disk': sample['project_disk_bytes'] += 7*1024**3
    if fault == 'memory': sample['available_memory_bytes'] = 0
    if fault == 'free_disk': sample['free_disk_bytes'] = 0
    ledger.sampler = lambda: sample
    with pytest.raises((BudgetExceeded, ValueError)):
        ledger.admit('G2', float('nan') if fault == 'duration' else 0, 100001 if fault == 'evaluations' else 0)


@pytest.mark.parametrize('marker', ['protocol.lock.json', 'protocol.lock.sha256',
                                   'confirmation.exposure.json', 'confirmation.events.jsonl'])
def test_policy_is_pre_freeze_only(transfer_case, local_config, marker):
    ledger, _, _ = transfer_case
    path = local_config.paths['results']/marker
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('retained evidence')
    with pytest.raises(ValueError, match='freeze|exposure|confirmation'):
        adopt(ledger)


@pytest.mark.parametrize('damage', ['duplicate', 'reason', 'timestamp', 'resources', 'extra', 'mode'])
def test_policy_replay_validates_records(transfer_case, damage):
    ledger, _, reopen = transfer_case
    adopt(ledger)
    rows = [json.loads(line) for line in ledger.path.read_text().splitlines()]
    row = rows[-1]
    if damage == 'duplicate': rows.append(dict(row))
    elif damage == 'reason': row['reason'] = ''
    elif damage == 'timestamp': row['recorded_at'] = 'unknown'
    elif damage == 'resources': row['resources_sha256'] = 'f'*64
    elif damage == 'extra': row['evaluation_limit'] = 200000
    else: row['mode'] = 'unbounded'
    ledger.path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    with pytest.raises(ValueError): reopen().snapshot()


def test_results_first_keeps_full_cohort_and_canonical_chunk(local_config):
    from signed_study.calibration import forecast_scale
    state = dict(experimental_seconds=1e9, attempted_evaluations=1056,
                 stage_seconds={s:1e9 for s in local_config.resources['stage_seconds']},
                 execution_policy={'mode':'results_first', 'event_id':'authorized'})
    ledger = SimpleNamespace(config=local_config, snapshot=lambda: state)
    value = measured()
    for row in value['user_measurements']: row['overhead_seconds'] = 5.
    result = forecast_scale(value, cohort(), ledger)
    assert result['status'] == 'feasible'
    assert result['scale'] == [418,1776]
    assert result['tested_scales'] == [[418,1776]]
    assert result['policy']['score_chunk_size'] == 512
    assert result['adjustments'] == []
    state['attempted_evaluations'] = 99999
    assert forecast_scale(value, cohort(), ledger)['status'] == 'infeasible'


def test_contamination_time_is_advisory_but_applicability_and_evaluations_remain(local_config):
    from signed_study.diagnostics import contamination_skip_reason
    state = dict(experimental_seconds=1e9, attempted_evaluations=0,
                 stage_seconds={'G5':1e9}, stage_limits={'G5':1200},
                 execution_policy={'mode':'results_first'})
    plan = dict(seconds=1e9, evaluations=2400)
    assert contamination_skip_reason(True, state, plan, local_config) is None
    assert contamination_skip_reason(False, state, plan, local_config)
    state['attempted_evaluations'] = 99999
    assert contamination_skip_reason(True, state, plan, local_config)


def saved_calibration(pipeline, monkeypatch, *, numerical_ok=True):
    """Publish synthetic completed work with genuine external failed accounting."""
    from signed_study.workload import calibration_entries
    from signed_study.data import episode_store_digest
    config, ledger, _, _, _, stages, evidence, store = pipeline
    prepare(pipeline)
    def workload(store, cache, users, resolved, ledger, *, output_path, **kwargs):
        entries = calibration_entries(resolved)
        chunks = sorted(set(resolved.numerics['score_chunk_sizes_check']) | {resolved.numerics['score_chunk_size']})
        episodes = {e.history.user_id:e for e in store.development}
        result = dict(status='completed', actual_evaluations=0, attempts=[], user_measurements=[],
                      chunk_invariance=numerical_ok, repair_decisions=[], memory_bound=False,
                      prototype_cap_user_fraction=0., fixed_seconds=0., optional_seconds=0.)
        inputs = dict(episodes=episode_store_digest(store), user_ids=users, entries=[e.id for e in entries],
                      numerical_checks=True, timing_only=False)
        for uid in users:
            ep = episodes[uid]
            for entry in entries:
                timings = {str(chunk):dict(score_seconds=.0001, write_seconds=.0001,
                            overhead_seconds=0., numerical_accepted=numerical_ok) for chunk in chunks}
                result['user_measurements'].append(dict(user_id=uid, family=entry.family,
                    configuration_id=entry.id, parameters=entry.parameters, K=entry.parameters.get('K',4),
                    support_count=len(ep.history.positive_ids)+len(ep.history.negative_ids),
                    positive_count=len(ep.history.positive_ids), negative_count=len(ep.history.negative_ids),
                    candidate_count=len(ep.candidate_ids), chunk_size=512, fit_seconds=.0001,
                    score_seconds=.0001, write_seconds=.0001, overhead_seconds=0.,
                    diagnostics={'status':'ok'}, score_measurements=timings))
                for chunk in [512, *[c for c in chunks if c != 512]]:
                    # Model the complete historical journal schema. The reader
                    # must reject missing final score/timing evidence, even in
                    # a synthetic adoption fixture.
                    from signed_study.metrics import evaluate_user
                    scores=np.zeros(len(ep.candidate_ids),dtype=np.float64)
                    metrics=evaluate_user(scores,np.arange(len(scores),dtype=np.int8)%2,ep.candidate_ids)
                    row = dict(evaluation_index=len(result['attempts']), user_id=uid,
                               configuration_id=entry.id, family=entry.family, parameters=entry.parameters,
                               chunk_size=chunk, kind='fit_and_score' if chunk==512 else 'score_repeat',
                               status='completed', diagnostics={'status':'ok'}, accepted=numerical_ok,
                               score_seconds=.0001,write_seconds=.0001,scores=scores.tolist(),
                               candidate_ids=list(ep.candidate_ids),metrics=metrics)
                    if chunk==512:row['fit_seconds']=.0001
                    result['attempts'].append(row)
        result['actual_evaluations'] = len(result['attempts'])
        digest = canonical_sha256(inputs)
        with output_path.with_suffix('.jsonl').open('w') as handle:
            for row in result['attempts']:
                for status in ('started', 'completed'):
                    event = dict(schema_version=1, input_sha256=digest, attempt={**row,'status':status})
                    event['output_sha256'] = canonical_sha256(event)
                    handle.write(json.dumps(event)+'\n')
        atomic_write_json(output_path, dict(schema_version=1, input_sha256=digest,
                                           output_sha256=canonical_sha256(result), measurements=result))
        return result
    monkeypatch.setattr(stages, 'measure_users', workload)
    with monkeypatch.context() as old_forecast:
        old_forecast.setattr(stages, 'forecast_scale', lambda *a:dict(status='infeasible',
            tested_scales=[[256,1000]], stop_reason='Smallest permitted scale cannot fit remaining allocations and reserve'))
        failed = stages.run_stage('calibrate', config, ledger)
    assert failed['status'] == 'blocked'
    old = file_reference(config.paths['results']/'calibration.json')
    adopt(ledger)
    assert evidence.record_correctness(config, ledger)['status'] == 'completed'
    return failed, old


def test_saved_calibration_adoption_preserves_original_and_consumers_follow_receipt(pipeline, monkeypatch):
    config, ledger, _, _, events, stages, _, _ = pipeline
    failed, old = saved_calibration(pipeline, monkeypatch)
    before = ledger.snapshot()
    monkeypatch.setattr(stages, 'measure_users', lambda *a,**kw:pytest.fail('adoption must not fit'))
    monkeypatch.setattr(stages, 'load_encoder', lambda *a:pytest.fail('adoption must not encode'))
    result = stages.run_stage('resume-calibration', config, ledger)
    assert result['status'] == 'completed', result
    assert result['artifacts']['calibration']['path'] != old['path']
    assert file_reference(old['path']) == old
    state = ledger.snapshot()
    assert state['attempted_evaluations'] == before['attempted_evaluations'] == 1056
    assert all(state['attempts'][aid]['status']=='completed' for aid in result['attempt_ids'])
    assert not set(result['attempt_ids']) & set(failed['attempt_ids'])
    assert result['reused_measurement_provenance']['parent_attempt_id'] == failed['attempt_ids'][-1]
    assert result['forecast']['scale'] == [418,1776]
    resumed = json.loads(Path(result['artifacts']['calibration']['path']).read_text())
    assert resumed['input_sha256'] == result['input_sha256']
    assert stages.run_stage('resume-calibration', config, ledger)['output_sha256'] == result['output_sha256']
    # Development consumes the versioned stage-selected calibration bytes.
    from signed_study import development
    smoke_inputs = {'calibrate':result['output_sha256'], 'development_users':8}
    from signed_study.stage_evidence import save_stage
    with ledger.work('G3',0,0) as aid: save_stage(config,'smoke',smoke_inputs,{},[aid])
    def develop(resolved, ledger, store, cache, measurements, **kwargs):
        assert measurements == resumed
        return {'status':'blocked', 'stop_reason':'synthetic downstream observation',
                'artifact_name':'not-written.json', 'batches':[]}
    monkeypatch.setattr(development,'execute_development',develop)
    assert stages.run_stage('develop',config,ledger)['stop_reason']=='synthetic downstream observation'
    from signed_study.report import render_report
    report = render_report(config.paths['results'])
    assert report['outcome'] != 'practical infeasibility'
    text = (config.paths['results']/'report.md').read_text()
    assert 'results_first' in text
    import csv
    actual = list(csv.DictReader((config.paths['results']/'actual-work.csv').open()))
    assert next(r for r in actual if r['work']=='calibration')['attempted'] == '1056'


@pytest.mark.parametrize('damage', ['publication', 'workload', 'journal', 'cache', 'numerical', 'source', 'policy'])
def test_saved_calibration_adoption_rejects_unqualified_evidence(pipeline, monkeypatch, damage):
    config, ledger, _, _, _, stages, evidence, _ = pipeline
    failed, old = saved_calibration(pipeline, monkeypatch, numerical_ok=damage!='numerical')
    if damage == 'publication': Path(old['path']).write_text('{}')
    if damage in ('workload','journal'):
        ref = ledger.snapshot()['attempts'][failed['attempt_ids'][-1]]['outputs']['attempts']
        path = Path(ref['path'])
        (path if damage=='workload' else path.with_suffix('.jsonl')).write_text('{}\n')
    if damage == 'cache': (config.paths['cache']/'embeddings'/'vectors.npy').unlink()
    if damage == 'source':
        original = evidence.code_identity
        monkeypatch.setattr(evidence, 'code_identity', lambda:{**original(), 'src/signed_study/models.py':'f'*64})
        evidence.record_correctness(config,ledger)
    if damage == 'policy':
        from dataclasses import replace
        config = replace(config, prototype={**config.prototype,'max_updates':100},
            pre_freeze_repairs=({'kind':'prototype_iterations','stage':'G2','reason':'different measured solver'},))
        evidence.record_correctness(config,ledger)
    result = stages.run_stage('resume-calibration', config, ledger)
    assert result['status'] == 'blocked', result
    assert not (config.paths['results']/'stages'/'calibrate.json').exists()
    assert ledger.snapshot()['attempted_evaluations'] == 1056


def test_cli_explicit_policy_and_saved_adoption_are_available(pipeline, monkeypatch, capsys):
    from signed_study import __main__ as cli, budget, protocol
    config, ledger, _, _, _, _, _, _ = pipeline
    monkeypatch.setattr(protocol,'load_config',lambda p:config)
    monkeypatch.setattr(budget,'Ledger',lambda *a,**kw:ledger)
    assert cli.main(['results-first','--reason','authorized scientific completion']) == 0
    assert json.loads(capsys.readouterr().out)['execution_policy']['mode'] == 'results_first'
    assert cli.main(['resume-calibration']) == 2


def test_selected_and_endpoint_forecasts_ignore_only_elapsed_capacity(local_config, monkeypatch):
    from signed_study import development
    from signed_study.calibration import planned_comparison_schedule
    from signed_study.models import initial_configurations
    store = cohort()
    entries = initial_configurations(local_config)
    schedule = [{**slot,'parameters':next(e.parameters for e in entries if e.family==slot['fit_family'])}
                for slot in planned_comparison_schedule()]
    calibration = {'calibration_user_ids':[e.history.user_id for e in store.development[:32]],
                   'forecast':{'catalog_items':3883, 'planning':{'confirmation_entries':13,
                       'timing_evaluations':2496,'numerical_diagnostic_evaluations':704},
                       'planned_user_ids':{'confirmation':[e.history.user_id for e in store.confirmation]}}}
    result = {'comparison_schedule':schedule, 'confirmation_user_ids':calibration['forecast']['planned_user_ids']['confirmation']}
    state = dict(experimental_seconds=1e9, attempted_evaluations=1056,
                 stage_seconds={s:1e9 for s in local_config.resources['stage_seconds']},
                 execution_policy={'mode':'results_first'})
    ledger = SimpleNamespace(snapshot=lambda:state)
    monkeypatch.setattr(development,'projected_cost',lambda *a,**kw:1e6)
    def selected():
        return development.refresh_selected_forecast(result,store,[],local_config,ledger,calibration)
    def endpoint(seconds=1e6):
        return development.endpoint_campaign_forecast(store.development,entries,entries[:1],[],store,
                                                       calibration,local_config,ledger,seconds)
    assert selected()['status'] == endpoint()['status'] == 'feasible'
    for invalid in (-1, float('inf'), float('nan')):
        with pytest.raises(ValueError, match='finite|nonnegative'): endpoint(invalid)
    state['attempted_evaluations'] = 99999
    assert selected()['status'] == endpoint()['status'] == 'infeasible'


def test_correctness_archive_preserves_original_bytes(pipeline):
    config, ledger, _, _, _, _, evidence, _ = pipeline
    prepare(pipeline)
    path = config.paths['results']/'correctness.json'
    original = path.read_bytes()
    digest = json.loads(original)['output_sha256']
    evidence.record_correctness(config,ledger)
    archive = path.with_name(f'correctness-{digest}.json')
    assert archive.read_bytes() == original


def test_original_failed_calibration_cannot_be_overwritten_by_calibrate(pipeline, monkeypatch):
    config, ledger, _, _, events, stages, _, _ = pipeline
    _, original = saved_calibration(pipeline,monkeypatch)
    count = len(events)
    result = stages.run_stage('calibrate',config,ledger)
    assert result['status']=='blocked'
    assert file_reference(original['path'])==original
    assert len(events)==count


def test_policy_transition_requalifies_reduced_receipt_without_losing_its_identity(pipeline,monkeypatch):
    from signed_study.stage_evidence import read_stage, save_stage
    config,ledger,_,_,_,stages,_,_=pipeline
    prepare(pipeline)
    original_forecast=stages.forecast_scale
    def reduced(*args):
        result=original_forecast(*args)
        result['scale']=[256,1500]
        result['planned_user_ids']={role:ids[:n] for (role,ids),n in
                                   zip(result['planned_user_ids'].items(),(256,1500))}
        return result
    with monkeypatch.context() as bounded:
        bounded.setattr(stages,'forecast_scale',reduced)
        old=stages.run_stage('calibrate',config,ledger)
    assert old['status']=='completed'
    old_receipt=file_reference(config.paths['results']/'stages'/'calibrate.json')
    old_calibration=old['artifacts']['calibration']
    old_policy=old['artifacts']['policy']
    with ledger.work('G3',0,0) as aid:
        smoke=save_stage(config,'smoke',{'calibrate':old['output_sha256'],'development_users':8},{},[aid])
    with ledger.work('G3',0,0) as aid:
        save_stage(config,'develop',{'calibrate':old['output_sha256'],'smoke':smoke['output_sha256']},{},[aid])
    stale={name:file_reference(config.paths['results']/'stages'/f'{name}.json') for name in ('smoke','develop')}
    event_id=adopt(ledger)
    monkeypatch.setattr(stages,'measure_users',lambda *a,**kw:pytest.fail('policy requalification must not refit'))
    monkeypatch.setattr(stages,'load_encoder',lambda *a:pytest.fail('policy requalification must not encode'))
    for action in ('calibrate','smoke','develop'):
        result=stages.run_stage(action,config,ledger)
        assert result['status']=='blocked',(action,result)
    result=stages.run_stage('resume-calibration',config,ledger)
    assert result['status']=='completed',result
    assert result['forecast']['scale']==[418,1776]
    assert result['execution_policy']['event_id']==event_id
    assert result['output_sha256']!=old['output_sha256']
    assert read_stage(config,'calibrate')==result
    assert file_reference(old_receipt['path'])==old_receipt
    assert file_reference(old_calibration['path'])==old_calibration
    assert file_reference(old_policy['path'])==old_policy
    assert result['artifacts']['previous_receipt']==old_receipt
    assert ledger.snapshot()['attempted_evaluations']==1056
    for name,reference in stale.items():
        assert file_reference(reference['path'])==reference
    from signed_study import report
    interim=report.render_report(config.paths['results'])
    interim_text=Path(interim['report']).read_text()
    assert 'smoke: stale' in interim_text and 'develop: stale' in interim_text
    with pytest.raises(ValueError,match='inputs changed'):
        read_stage(config,'smoke',{'calibrate':result['output_sha256'],'development_users':8})
    def smoke_workload(store,cache,users,resolved,ledger,*,output_path,**kwargs):
        value=dict(status='completed',actual_evaluations=624,attempts=[])
        atomic_write_json(output_path,dict(schema_version=1,input_sha256=canonical_sha256({'users':users}),
            output_sha256=canonical_sha256(value),measurements=value))
        return value
    monkeypatch.setattr(stages,'measure_users',smoke_workload)
    new_smoke=stages.run_stage('smoke',config,ledger)
    assert new_smoke['status']=='completed',new_smoke
    assert new_smoke['supersedes']==stale['smoke']
    from signed_study import development, report
    def develop(resolved,ledger,store,cache,measurements,**kwargs):
        value=dict(schema_version=1,status='completed',attempts=[],actual_evaluations=0)
        value['output_sha256']=canonical_sha256(value)
        atomic_write_json(config.paths['results']/'new-development.json',value)
        return dict(status='completed',artifact_name='new-development.json',batches=[],complete_suite_eligible=True)
    monkeypatch.setattr(development,'execute_development',develop)
    new_develop=stages.run_stage('develop',config,ledger)
    assert new_develop['status']=='completed',new_develop
    assert new_develop['supersedes']==stale['develop']
    assert read_stage(config,'smoke')==new_smoke
    assert read_stage(config,'develop')==new_develop
    for reference in stale.values(): assert file_reference(reference['path'])==reference
    rendered=report.render_report(config.paths['results'])
    assert rendered['outcome']!='practical infeasibility'
    inventory=json.loads(Path(rendered['inventory']).read_text())['artifacts']
    effective=json.loads((config.paths['results']/'stages'/'calibrate.active.json').read_text())['receipt']
    assert any(Path(row['name']).name==Path(effective['path']).name for row in inventory)
    assert any(row['status'].startswith('superseded') and Path(row['name']).name=='calibrate.json' for row in inventory)
    import csv
    actual={row['work']:row for row in csv.DictReader((config.paths['results']/'actual-work.csv').open())}
    assert actual['calibration']['reserved']=='1056'
    assert actual['smoke']['reserved']=='624'
    assert not (config.paths['results']/'protocol.lock.json').exists()
    assert not (config.paths['results']/'confirmation.exposure.json').exists()
    (config.paths['results']/'protocol.lock.json').write_text('synthetic frozen marker')
    with pytest.raises(ValueError,match='freeze|exposure'):
        save_stage(config,'calibrate',result['inputs'],{},[],supersedes=effective)
    selector=config.paths['results']/'stages'/'calibrate.active.json'
    selector.write_text('{}')
    with pytest.raises(ValueError): read_stage(config,'calibrate')


def test_results_first_development_rejects_reduced_cohort_before_work(pipeline,monkeypatch):
    from signed_study import development
    config,ledger,_,_,_,_,_,store=pipeline
    adopt(ledger)
    calibration={'forecast':{'planned_user_ids':{
        'development':[e.history.user_id for e in store.development[:256]],
        'confirmation':[e.history.user_id for e in store.confirmation[:1500]]}}}
    monkeypatch.setattr(development,'measure_users',lambda *a,**kw:pytest.fail('reduced evidence reached model/outcome work'))
    with pytest.raises(ValueError,match='full.*cohort'):
        development.execute_development(config,ledger,store,None,calibration)
    assert ledger.snapshot()['attempted_evaluations']==0


def test_adopted_calibration_requalification_retains_original_workload_owner(pipeline,monkeypatch):
    from signed_study.stage_evidence import read_stage
    config,ledger,_,_,_,stages,evidence,_=pipeline
    failed,_=saved_calibration(pipeline,monkeypatch)
    first=stages.run_stage('resume-calibration',config,ledger)
    assert first['status']=='completed',first
    previous=file_reference(config.paths['results']/'stages'/'calibrate.json')
    assert evidence.record_correctness(config,ledger)['status']=='completed'
    monkeypatch.setattr(stages,'measure_users',lambda *a,**kw:pytest.fail('administrative G1 refresh must not refit'))
    monkeypatch.setattr(stages,'load_encoder',lambda *a:pytest.fail('administrative G1 refresh must not encode'))
    current=stages.run_stage('resume-calibration',config,ledger)
    assert current['status']=='completed',current
    assert current['reused_measurement_provenance']['parent_attempt_id']==failed['attempt_ids'][-1]
    assert current['supersedes']==previous
    assert read_stage(config,'calibrate')==current
    assert file_reference(previous['path'])==previous
    assert ledger.snapshot()['attempted_evaluations']==1056


def test_repeated_policy_adoption_after_freeze_is_rejected(transfer_case, local_config):
    ledger, _, _ = transfer_case
    adopt(ledger)
    local_config.paths['results'].mkdir(parents=True,exist_ok=True)
    (local_config.paths['results']/'protocol.lock.json').write_text('frozen')
    with pytest.raises(ValueError, match='freeze|exposure'): adopt(ledger)


@pytest.fixture
def versioned_frozen_study(pipeline, monkeypatch, request):
    config, _, _, _, _, stages, _, _ = pipeline
    save = stages.save_stage
    def versioned(c, name, inputs, files, attempts, **kwargs):
        if name == 'calibrate':
            alternate = config.paths['results']/'calibration-versioned.json'
            alternate.write_bytes(Path(files['calibration']).read_bytes())
            files = {**files, 'calibration':alternate}
            (config.paths['results']/'calibration.json').write_text('unrelated historical publication')
        return save(c,name,inputs,files,attempts,**kwargs)
    monkeypatch.setattr(stages,'save_stage',versioned)
    return request.getfixturevalue('frozen_study')


def test_freeze_confirmation_and_diagnostics_follow_versioned_calibration(versioned_frozen_study):
    from signed_study import run, protocol_lock
    config, ledger, _, lock = versioned_frozen_study
    _, payload = protocol_lock.read_lock(lock)
    assert Path(payload['identities']['artifacts']['calibration.json']['path']).name == 'calibration-versioned.json'
    assert run.confirm(lock,ledger)['status'] == 'completed'
    assert run.diagnose(lock,ledger)['status'] == 'completed'
    assert (config.paths['results']/'calibration.json').read_text() == 'unrelated historical publication'


def test_freeze_schedule_rechecks_non_time_constraints_with_advisory_clocks(frozen_study):
    from signed_study import protocol_lock
    from signed_study.protocol import config_from_record, ProtocolMismatch
    _, ledger, store, lock = frozen_study
    _, payload = protocol_lock.read_lock(lock)
    config = config_from_record(payload['config'])
    selected=full_cohort_schedule(payload['development'],store,config)
    state = ledger.snapshot()
    state.update(experimental_seconds=1e9, stage_seconds={s:1e9 for s in state['stage_seconds']},
                 execution_policy={'mode':'results_first'})
    fake = SimpleNamespace(snapshot=lambda:state)
    protocol_lock._validate_schedule(selected,config,store,fake)
    state['execution_policy'] = {'mode':'bounded'}
    with pytest.raises(ProtocolMismatch): protocol_lock._validate_schedule(selected,config,store,fake)
    state['execution_policy'] = {'mode':'results_first'}
    state['attempted_evaluations'] = 99999
    with pytest.raises(ProtocolMismatch): protocol_lock._validate_schedule(selected,config,store,fake)


def test_report_qualifies_third_long_repair_without_waiving_cost_integrity(transfer_case):
    from signed_study.report import _repair_summaries
    ledger, clock, _ = transfer_case
    adopt(ledger)
    for _ in range(2):
        with ledger.work('repair',0,0): clock.now += 1
    start = len(ledger.snapshot()['events'])
    with ledger.work('repair',0,0) as repair: clock.now += 1201
    end = len(ledger.snapshot()['events'])
    trace = ledger.path.parent/'retained-failure.log'
    trace.write_text('original diagnostic failure')
    record = dict(schema_version=1,review_ids=['R1'],source_base='a'*40,source_sha256='b'*64,
        failure_attempt_ids=[],failure_trace=file_reference(trace),validation_attempt_ids=[repair],
        repair_attempt_id=repair,repair_cycle=3,ledger_event_start=start,ledger_event_end=end,
        attributed_seconds=1201.)
    path = ledger.path.parent/'long-third-repair.json'
    record['output_sha256'] = canonical_sha256(record)
    atomic_write_json(path,record)
    with ledger.work('G0',0,0) as publication:
        ledger.finish(publication,'completed',{}, {'repair_log':file_reference(path)})
    problems = []
    summaries = _repair_summaries(ledger.snapshot(),problems)
    assert len(summaries)==1 and summaries[0]['repair_cycle']==3 and not problems
    assert summaries[0]['attributed_seconds']==1201.
    # The same publication cannot invent its elapsed cost under the waiver.
    record['attributed_seconds'] = 1.
    record.pop('output_sha256')
    record['output_sha256'] = canonical_sha256(record)
    atomic_write_json(path,record)
    with ledger.work('G0',0,0) as publication:
        ledger.finish(publication,'completed',{}, {'repair_log':file_reference(path)})
    problems = []
    assert _repair_summaries(ledger.snapshot(),problems)==[]
    assert problems
