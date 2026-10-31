"""Durable workload transitions, crash recovery and publication scaling."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from signed_study import workload
from signed_study.models import ModelConfig, UserModel
from signed_study.protocol import canonical_sha256
from test_stage_pipeline import pipeline


def replay(path, **kwargs):
    from signed_study import workload_journal
    return workload_journal.replay_workload_journal(path, **kwargs)


def fake_work(monkeypatch):
    entry = ModelConfig('SignedMax', {'K': 2, 'lambda': 2}, 'one')
    monkeypatch.setattr(workload, 'fit_user', lambda *a, **kw: UserModel(
        'SignedMax', entry.parameters, (), {}, {'status': 'ok', 'failures': []}))
    monkeypatch.setattr(workload, 'score_user', lambda m, ids, c, **kw: np.arange(len(ids), dtype=np.float64))
    return [entry]


def run_work(pipeline, monkeypatch, *, count=4, numeric=False, path=None):
    config, ledger, _, _, _, _, _, store = pipeline
    entries = fake_work(monkeypatch)
    users = [e.history.user_id for e in store.development[:count]]
    path = path or config.paths['results'] / 'durable.json'
    with ledger.work('G2', 1, len(users) * (3 if numeric else 1)):
        result = workload.measure_users(store, None, users, config, ledger,
            numerical_checks=numeric, output_path=path, declared_entries=entries)
    return result, path


def test_final_replay_recovers_every_measurement_and_resource_field(pipeline, monkeypatch):
    result, path = run_work(pipeline, monkeypatch, numeric=True)
    recovered = replay(path.with_suffix('.jsonl'))
    assert recovered['complete'] is True
    assert recovered['measurements'] == result
    assert recovered['measurements'] == json.loads(path.read_text())['measurements']
    assert all('write_seconds' in row for row in recovered['measurements']['attempts'])
    assert len(recovered['measurements']['resource_trace']) == 4


def test_checkpoints_serialize_linear_evidence_volume(pipeline, monkeypatch):
    writes = []
    original = workload.atomic_write_json

    def counted(path, payload):
        if 'measurements' in payload:
            writes.append(len(payload['measurements']['attempts']))
        return original(path, payload)

    monkeypatch.setattr(workload, 'atomic_write_json', counted)
    result, path = run_work(pipeline, monkeypatch, count=16)
    assert len(writes) <= 6
    assert sum(writes) <= 3 * result['actual_evaluations']
    assert replay(path.with_suffix('.jsonl'))['complete'] is True


def test_existing_output_or_journal_is_never_silently_rerun(pipeline, monkeypatch):
    result, path = run_work(pipeline, monkeypatch, count=1)
    before = path.read_bytes(), path.with_suffix('.jsonl').read_bytes()
    with pytest.raises(FileExistsError):
        run_work(pipeline, monkeypatch, count=1, path=path)
    assert (path.read_bytes(), path.with_suffix('.jsonl').read_bytes()) == before
    assert result['actual_evaluations'] == 1


def rehash(event):
    value = {k: v for k, v in event.items() if k != 'output_sha256'}
    event['output_sha256'] = hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
        allow_nan=False).encode()).hexdigest()
    return event


def write_events(path, events, tail=b''):
    path.write_bytes(b''.join(json.dumps(e, sort_keys=True).encode() + b'\n' for e in events) + tail)


@pytest.mark.parametrize('damage', ['mixed_inputs', 'wrong_user', 'wrong_chunk',
                                    'duplicate_start', 'conflicting_terminal', 'bad_internal'])
def test_replay_rejects_corrupt_identity_and_transitions(pipeline, monkeypatch, damage):
    _, path = run_work(pipeline, monkeypatch, count=1)
    journal = path.with_suffix('.jsonl')
    events = [json.loads(line) for line in journal.read_text().splitlines()]
    start = next(i for i, e in enumerate(events) if e.get('event') == 'attempt_started')
    terminal = next(i for i, e in enumerate(events) if e.get('event') == 'attempt_terminal')
    if damage == 'mixed_inputs':
        events[terminal]['input_sha256'] = '0' * 64
    elif damage == 'wrong_user':
        events[terminal]['attempt']['user_id'] = 10000
    elif damage == 'wrong_chunk':
        events[terminal]['attempt']['chunk_size'] += 1
    elif damage == 'duplicate_start':
        events.insert(start + 1, json.loads(json.dumps(events[start])))
    elif damage == 'conflicting_terminal':
        value = json.loads(json.dumps(events[terminal]))
        value['attempt']['scores'][0] += 1
        events.insert(terminal + 1, value)
    else:
        events[start]['event'] = 'unrecognized_internal_record'
    # Recompute envelope hashes: integrity alone does not validate transitions.
    write_events(journal, [rehash(event) for event in events])
    with pytest.raises(ValueError):
        replay(journal)


def test_torn_tail_is_incomplete_and_internal_corruption_is_rejected(pipeline, monkeypatch):
    _, path = run_work(pipeline, monkeypatch, count=1)
    journal = path.with_suffix('.jsonl')
    complete = journal.read_bytes()
    journal.write_bytes(complete + b'{"schema_version":2')
    recovered = replay(journal)
    assert recovered['complete'] is False
    assert recovered['torn_tail_bytes'] > 0
    assert recovered['measurements']['status'] == 'interrupted'
    journal.write_bytes(complete.split(b'\n', 1)[0] + b'\n{broken}\n' + complete.split(b'\n', 1)[1])
    with pytest.raises(ValueError):
        replay(journal)


@pytest.mark.parametrize('damage', ['metrics_missing', 'metrics_text', 'metrics_nonfinite',
                                    'duplicate_final_measurement', 'missing_numerical_repeats'])
def test_version_two_rejects_malformed_complete_science(pipeline, monkeypatch, damage):
    result, path = run_work(pipeline, monkeypatch, count=1)
    journal = path.with_suffix('.jsonl')
    events = [json.loads(line) for line in journal.read_text().splitlines()]
    terminal = next(e for e in events if e.get('event') == 'attempt_terminal')
    if damage == 'metrics_missing':
        terminal['attempt'].pop('metrics')
    elif damage == 'metrics_text':
        terminal['attempt']['metrics'] = 'garbage'
    elif damage == 'metrics_nonfinite':
        terminal['attempt']['metrics']['auc'] = -1
    elif damage == 'duplicate_final_measurement':
        final = next(e for e in events if e.get('event') == 'user_completed')
        final['measurements'].append(final['measurements'][0])
    else:
        header = events[0]
        header['inputs']['numerical_checks'] = True
        header['identity']['inputs']['numerical_checks'] = True
        input_digest = canonical_sha256(header['inputs'])
        workload_digest = canonical_sha256(header['identity'])
        for event in events:
            event['input_sha256'], event['workload_sha256'] = input_digest, workload_digest
    if damage.startswith('metrics_'):
        if 'metrics' in terminal['attempt']:
            result['attempts'][0]['metrics'] = terminal['attempt']['metrics']
        else:
            result['attempts'][0].pop('metrics')
        events[-1]['measurements_sha256'] = canonical_sha256(result)
    write_events(journal, [rehash(event) for event in events])
    with pytest.raises(ValueError):
        replay(journal)


def test_version_two_binds_the_verified_execution_config(pipeline, monkeypatch):
    from signed_study.stage_evidence import config_record
    result, path = run_work(pipeline, monkeypatch, count=1)
    identity = config_record(pipeline[0])
    assert replay(path.with_suffix('.jsonl'), expected_config=identity)['measurements'] == result
    identity['numerics']['score_chunk_size'] = 7
    with pytest.raises(ValueError, match='config'):
        replay(path.with_suffix('.jsonl'), expected_config=identity)


def test_final_overhead_includes_supplemental_journal_publication(pipeline, monkeypatch):
    from signed_study.workload_journal import WorkloadJournal
    clock = pipeline[2]
    monkeypatch.setattr(workload.time, 'perf_counter_ns', lambda: int(clock[0] * 1e9))
    original = WorkloadJournal.append

    def delayed(self, event, **payload):
        value = original(self, event, **payload)
        if event == 'measurement':
            clock[0] += .025
        return value

    monkeypatch.setattr(WorkloadJournal, 'append', delayed)
    result, path = run_work(pipeline, monkeypatch, count=1)
    assert result['user_measurements'][0]['overhead_seconds'] == pytest.approx(.025)
    assert replay(path.with_suffix('.jsonl'))['measurements'] == result


def test_failure_during_checkpoint_keeps_raw_terminal_and_reservation(pipeline, monkeypatch):
    original = workload.atomic_write_json

    def fail_checkpoint(path, value):
        if str(path).endswith('.partial.json'):
            raise OSError('injected checkpoint failure')
        return original(path, value)

    monkeypatch.setattr(workload, 'atomic_write_json', fail_checkpoint)
    config, ledger, _, _, _, _, _, _ = pipeline
    reserved = ledger.snapshot()['attempted_evaluations']
    with pytest.raises(OSError, match='checkpoint'):
        run_work(pipeline, monkeypatch, count=4)
    assert ledger.snapshot()['attempted_evaluations'] == reserved + 4
    recovered = replay((config.paths['results'] / 'durable.json').with_suffix('.jsonl'))
    assert recovered['complete'] is False
    assert recovered['measurements']['actual_evaluations'] == 1
    assert recovered['measurements']['attempts'][0]['status'] == 'completed'


@pytest.mark.parametrize('boundary,started', [('attempt_started', 0), ('measurement', 1)])
def test_io_failure_before_start_or_after_terminal_never_launches_more_work(
        pipeline, monkeypatch, boundary, started):
    from signed_study.workload_journal import JournalWriteError, WorkloadJournal
    original = WorkloadJournal.append

    def fail(self, event, **payload):
        if event == boundary:
            self.failed = True
            raise JournalWriteError('injected append failure')
        return original(self, event, **payload)

    monkeypatch.setattr(WorkloadJournal, 'append', fail)
    config, ledger, _, _, _, _, _, _ = pipeline
    reserved = ledger.snapshot()['attempted_evaluations']
    with pytest.raises(JournalWriteError):
        run_work(pipeline, monkeypatch, count=4)
    assert ledger.snapshot()['attempted_evaluations'] == reserved + 4
    value = replay((config.paths['results'] / 'durable.json').with_suffix('.jsonl'))
    assert value['complete'] is False
    assert value['measurements']['actual_evaluations'] == started
    assert not (config.paths['results'] / 'durable.json').exists()


def test_final_rename_failure_keeps_prior_evidence_and_complete_raw_journal(pipeline, monkeypatch):
    from signed_study import budget
    config, ledger, _, _, _, _, _, _ = pipeline
    previous = config.paths['results'] / 'previous.json'
    previous.write_bytes(b'prior complete publication\n')
    original = budget.os.replace

    def fail_final(source, target):
        if Path(target).name == 'durable.json':
            raise OSError('injected final rename failure')
        return original(source, target)

    monkeypatch.setattr(budget.os, 'replace', fail_final)
    with pytest.raises(OSError, match='rename'):
        run_work(pipeline, monkeypatch, count=4)
    assert previous.read_bytes() == b'prior complete publication\n'
    assert not (config.paths['results'] / 'durable.json').exists()
    value = replay((config.paths['results'] / 'durable.json').with_suffix('.jsonl'))
    assert value['complete'] is True  # Scientific work survived; owner still failed.
    assert value['measurements']['actual_evaluations'] == 4
    assert list(ledger.snapshot()['attempts'].values())[-1]['status'] == 'failed'


@pytest.mark.parametrize('boundary', ['fit_user', 'score_user'])
def test_io_failure_in_evaluation_stops_before_any_later_fit(pipeline, monkeypatch, boundary):
    config, ledger, _, _, _, _, _, store = pipeline
    entries = fake_work(monkeypatch)
    calls = []

    def fail(*args, **kwargs):
        calls.append(boundary)
        raise OSError('injected evaluation I/O failure')

    monkeypatch.setattr(workload, boundary, fail)
    path = config.paths['results'] / 'io.json'
    with ledger.work('G2', 1, 4):
        with pytest.raises(OSError, match='evaluation I/O'):
            workload.measure_users(store, None, [e.history.user_id for e in store.development[:4]],
                config, ledger, numerical_checks=False, output_path=path, declared_entries=entries)
    assert calls == [boundary]
    value = replay(path.with_suffix('.jsonl'))
    assert value['complete'] is False
    assert value['measurements']['actual_evaluations'] == 1
    assert value['measurements']['attempts'][0]['status'] == 'interrupted'


def test_legacy_reader_accepts_enriched_repeated_terminal_but_not_conflicts(tmp_path):
    path = tmp_path / 'legacy.jsonl'
    start = dict(evaluation_index=0, user_id=1, configuration_id='one', family='SignedMax',
                 parameters={'K': 2, 'lambda': 2}, kind='fit_and_score', chunk_size=512,
                 status='started', diagnostics={})
    done = dict(start, status='completed', fit_seconds=.1, score_seconds=.2,
                scores=[.5, .1], candidate_ids=[10, 11], metrics={'auc': 1.})
    def envelope(row):
        return rehash({'schema_version': 1, 'input_sha256': 'a' * 64, 'attempt': row})
    events = [envelope(start), envelope(done), envelope(dict(done, write_seconds=.01))]
    write_events(path, events)
    value = replay(path, expected_input_sha256='a' * 64)
    assert value['measurements']['attempts'] == [dict(done, write_seconds=.01)]
    assert value['legacy_fields_unavailable'] == ['user_measurements', 'resource_trace', 'overhead']
    events.append(envelope(dict(done, scores=[.6, .1])))
    write_events(path, events)
    with pytest.raises(ValueError, match='(?i)conflict'):
        replay(path)


def test_subprocess_exit_preserves_durable_start_without_finally(tmp_path):
    path = tmp_path / 'killed.jsonl'
    code = """
import os, sys
from signed_study.workload_journal import WorkloadJournal
inputs={'episodes':'a'*64,'user_ids':[1],'entries':['one'],'numerical_checks':False,'timing_only':True}
identity={'inputs':inputs,'config':{'scope':'explicit synthetic fixture'},'entries':[{'id':'one','family':'SignedMax','parameters':{'K':2,'lambda':2}}],
'score_chunk_size':512,'score_chunk_sizes_check':[1,7,512],
'users':[{'user_id':1,'positive_count':10,'negative_count':5,'candidate_count':2,'candidate_ids_sha256':'b'*64}]}
writer=WorkloadJournal(sys.argv[1],inputs,identity)
writer.append('attempt_started',attempt={'evaluation_index':0,'user_id':1,'configuration_id':'one',
'family':'SignedMax','parameters':{'K':2,'lambda':2},'kind':'fit_and_score','chunk_size':512,
'status':'started','diagnostics':{}})
os._exit(23)
"""
    result = subprocess.run([sys.executable, '-c', code, str(path)], check=False, capture_output=True, text=True)
    assert result.returncode == 23, result.stderr
    recovered = replay(path)
    assert recovered['complete'] is False
    assert recovered['measurements']['actual_evaluations'] == 1
    assert recovered['measurements']['attempts'][0]['status'] == 'started'
    assert recovered['measurements']['status'] == 'interrupted'
