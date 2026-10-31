"""Explicit pre-freeze continuation without repeating scientific work."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from signed_study.budget import BudgetExceeded, atomic_write_json
from signed_study.protocol import canonical_sha256
from signed_study.stage_evidence import file_reference
from test_budget_transfers import transfer_case
from test_calibration import cohort, measured
from test_stage_pipeline import pipeline, prepare
from test_protocol_lock import frozen_study


def adopt(ledger):
    assert hasattr(ledger, 'adopt_results_first'), 'durable results-first adoption is missing'
    return ledger.adopt_results_first('User authorized completion of scientific results on 31 October')


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
                    row = dict(evaluation_index=len(result['attempts']), user_id=uid,
                               configuration_id=entry.id, family=entry.family, parameters=entry.parameters,
                               chunk_size=chunk, kind='fit_and_score' if chunk==512 else 'score_repeat',
                               status='completed', diagnostics={'status':'ok'}, accepted=numerical_ok)
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
    state = ledger.snapshot()
    state.update(experimental_seconds=1e9, stage_seconds={s:1e9 for s in state['stage_seconds']},
                 execution_policy={'mode':'results_first'})
    fake = SimpleNamespace(snapshot=lambda:state)
    protocol_lock._validate_schedule(payload['development'],config,store,fake)
    state['execution_policy'] = {'mode':'bounded'}
    with pytest.raises(ProtocolMismatch): protocol_lock._validate_schedule(payload['development'],config,store,fake)
    state['execution_policy'] = {'mode':'results_first'}
    state['attempted_evaluations'] = 99999
    with pytest.raises(ProtocolMismatch): protocol_lock._validate_schedule(payload['development'],config,store,fake)


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
