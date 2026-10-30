"""Bounded preliminary-validation reallocations, replay, and real consumers."""
import json
from types import SimpleNamespace

import pytest

from signed_study.budget import BudgetExceeded, CommandCleanupError, Ledger, LedgerBusy
from signed_study.protocol import canonical_sha256
from signed_study.stage_evidence import plain
from test_calibration import cohort, measured


@pytest.fixture
def transfer_case(tmp_path, local_config):
    clock = SimpleNamespace(now=0.)
    sample = dict(installed_ram_bytes=32*1024**3, available_memory_bytes=16*1024**3,
                  rss_bytes=100*1024**2, project_disk_bytes=1000, free_disk_bytes=20*1024**3)
    def reopen():
        return Ledger(tmp_path/'events.jsonl', local_config, clock=lambda: clock.now,
                      wall_clock=lambda: 1_000_000+clock.now, sampler=lambda: dict(sample))
    ledger = reopen()
    yield ledger, clock, reopen
    ledger.close()


def transfer(ledger):
    return ledger.reallocate('G4', 'G1', 600, 'Bounded preliminary validation support')


def test_transfer_is_durable_and_conserves_caps(transfer_case, local_config):
    ledger, clock, reopen = transfer_case
    original = canonical_sha256(plain(local_config.resources))
    attempt = ledger.admit('G1', 1, 7)
    clock.now += 3
    ledger.finish(attempt, 'failed', {}, {})
    with ledger.work('setup', 1, 0):
        clock.now += 2
    with ledger.work('repair', 1, 1):
        clock.now += 1
    before = ledger.snapshot()
    event_id = transfer(ledger)
    state = reopen().snapshot()
    assert state['stage_limits'] == {'G0':600, 'G1':1200, 'G2':1200, 'G3':2400,
                                     'G4':4200, 'G5':1200, 'G6':1800, 'repair':1200}
    assert sum(state['stage_limits'].values()) == 13800
    assert state['reallocations'][0]['event_id'] == event_id
    assert state['reallocations'][0]['reason'] == 'Bounded preliminary validation support'
    assert len(state['reallocations']) == 1
    assert reopen().snapshot()['stage_limits'] == state['stage_limits']
    for key in ('stage_seconds', 'experimental_seconds', 'setup_seconds', 'attempted_evaluations',
                'repair_cycles', 'attempts', 'unfinished_attempts', 'resources_sha256',
                'baseline_disk_bytes', 'initial_free_disk_bytes', 'report_reserve_seconds'):
        assert state[key] == before[key]
    assert state['attempts'][attempt]['status'] == 'failed'
    assert canonical_sha256(plain(local_config.resources)) == original


def test_admission_and_live_checks_use_effective_limits(transfer_case):
    ledger, clock, _ = transfer_case
    with pytest.raises(BudgetExceeded, match='G1 stage'):
        ledger.admit('G1', 450, 0)
    transfer(ledger)
    attempt = ledger.admit('G1', 450, 0)
    clock.now += 700
    ledger.checkpoint()  # Beyond the original 600 seconds, within effective 1200.
    clock.now += 501
    with pytest.raises(BudgetExceeded, match='G1 stage'):
        ledger.checkpoint()
    with pytest.raises(BudgetExceeded, match='G1 stage'):
        ledger.finish(attempt, 'completed', {}, {})
    assert ledger.snapshot()['attempts'][attempt]['status'] == 'budget_exceeded'
    with pytest.raises(BudgetExceeded, match='G4 stage'):
        ledger.admit('G4', 2801, 0)
    attempt = ledger.admit('G4', 1, 0)
    clock.now += 4201
    with pytest.raises(BudgetExceeded, match='G4 stage'):
        ledger.finish(attempt, 'completed', {}, {})


@pytest.mark.parametrize('source,target,seconds,reason', [
    ('G4','G1',True,'reason'), ('G4','G1',0,'reason'), ('G4','G1',-1,'reason'),
    ('G4','G1',float('nan'),'reason'), ('G4','G1',float('inf'),'reason'),
    ('G4','G1','600','reason'), ('G4','G1',None,'reason'),
    ('G4','G1',600,''), ('G4','G1',600,'   '), ('G4','G1',600,None),
    ('G1','G1',1,'reason'), ('setup','G1',1,'reason'), ('repair','G1',1,'reason'),
    ('G6','G1',1,'reason'), ('unknown','G1',1,'reason'), ('G4','G3',1,'reason'),
    ('G4','G1',4800,'reason'), ('G4','G1',4801,'reason'),
    pytest.param('G4','G1',10**500,'reason',id='oversized-integer'),
])
def test_transfer_rejects_invalid_or_protected_changes(transfer_case, source,target,seconds,reason):
    ledger, _, _ = transfer_case
    with pytest.raises(ValueError):
        ledger.reallocate(source,target,seconds,reason)
    assert ledger.snapshot()['stage_seconds']['G1'] == 0


def test_transfer_rejects_already_admitted_donor(transfer_case):
    ledger, _, _ = transfer_case
    with ledger.work('G0', 0, 0):
        pass
    with pytest.raises(ValueError, match='admitted'):
        ledger.reallocate('G0','G1',1,'donor must be untouched')


def test_transfer_preserves_cleanup_blocked_protection(transfer_case, monkeypatch):
    from signed_study import budget
    ledger, _, reopen = transfer_case
    attempt = ledger.admit('G1', 1, 0)
    ledger._append('command_started', attempt_id=attempt, process_group_id=123)
    ledger.close()
    monkeypatch.setattr(budget, '_command_group_state', lambda pgid: {'live_pids':[123]})
    with pytest.raises(CommandCleanupError):
        transfer(reopen())
    state = reopen().snapshot()
    assert attempt in state['unfinished_attempts']
    assert state['reallocations'] == []


def test_transfer_keeps_positive_unspent_donor_capacity(transfer_case):
    ledger, clock, _ = transfer_case
    with ledger.work('G1', 0, 0):
        pass
    sample = ledger.sampler
    def costly_sample():
        clock.now += 1
        return sample()
    ledger.sampler = costly_sample
    with pytest.raises(BudgetExceeded, match='evaluations'):
        ledger.admit('G0',0,100001)
    assert ledger.snapshot()['stage_seconds']['G0'] == 1
    with pytest.raises(ValueError):
        ledger.reallocate('G0','G1',599,'would consume all unspent donor time')
    ledger.reallocate('G0','G1',598,'leave one unspent second')
    assert ledger.snapshot()['stage_limits']['G0'] == 2


@pytest.mark.parametrize('stage', ['G2','G3','G4','G5'])
def test_transfer_rejects_active_or_confirmation_work(transfer_case, stage):
    ledger, _, reopen = transfer_case
    attempt = ledger.admit('G1', 0, 0)
    with pytest.raises(LedgerBusy):
        transfer(ledger)
    other = reopen()
    with pytest.raises(LedgerBusy):
        transfer(other)
    ledger.finish(attempt, 'completed', {}, {})
    with ledger.work(stage, 0, 0):
        pass
    with pytest.raises(ValueError, match='admitted'):
        transfer(ledger)


@pytest.mark.parametrize('damage', ['duplicate','negative','protected','empty_reason','missing_id','extra_cap'])
def test_transfer_replay_rejects_invalid_records(transfer_case, damage):
    ledger, _, reopen = transfer_case
    transfer(ledger)
    rows = [json.loads(line) for line in ledger.path.read_text().splitlines()]
    record = rows[-1]
    if damage == 'duplicate':
        rows.append(dict(record))
    elif damage == 'negative':
        record['seconds'] = -1
    elif damage == 'protected':
        record['source_stage'] = 'G6'
    elif damage == 'empty_reason':
        record['reason'] = ''
    elif damage == 'missing_id':
        del record['event_id']
    else:
        record['stage_limits'] = {'G1': 999999}
    ledger.path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    with pytest.raises(ValueError):
        reopen().snapshot()
    with pytest.raises(ValueError):
        ledger.admit('G1', 0, 0)


@pytest.mark.parametrize('consumer', ['calibration','selected_development'])
def test_forecasts_use_reallocated_stage_limits(transfer_case, local_config, consumer):
    from signed_study.calibration import forecast_scale
    from signed_study.development import refresh_selected_forecast
    ledger, _, _ = transfer_case
    transfer(ledger)
    state = ledger.snapshot()
    # Isolate the G4 forecast boundary: 1 second remains after reallocation,
    # while the configured default would leave 601 seconds. No workload runs.
    state['stage_seconds']['G4'] = 4199
    fake = SimpleNamespace(config=local_config, snapshot=lambda: state)
    if consumer == 'calibration':
        evaluate = lambda: forecast_scale(measured(), cohort(), fake)
    else:
        store = cohort(32,1)
        result = dict(comparison_schedule=[dict(fit_family='SignedMax',parameters={'K':1,'lambda':0},geometric=True)],
                      confirmation_user_ids=[10000],development_user_ids=[1])
        timings = [dict(family='SignedMax',support_count=100,positive_count=95,negative_count=5,
                        candidate_count=20,K=1,chunk_size=512,fit_seconds=1.,score_seconds=0.,
                        write_seconds=0.,overhead_seconds=0.)]
        calibration = {'forecast': {'catalog_items': 300, 'planning': {'confirmation_entries':13,
                       'timing_evaluations':2496,'numerical_diagnostic_evaluations':704}}}
        evaluate = lambda: refresh_selected_forecast(result,store,timings,local_config,fake,calibration)
    assert evaluate()['status'] == 'infeasible'
    del state['stage_limits']
    with pytest.raises(ValueError, match='effective stage limits'):
        evaluate()
    # Legacy fake ledgers have no transfer information and retain defaults.
    del state['reallocations']
    assert evaluate()['status'] == 'feasible'
