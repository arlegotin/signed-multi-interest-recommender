"""Protocol contracts used by later stage/provenance tests."""

import hashlib
import json
from pathlib import Path

import pytest

from signed_study.protocol import StudyConfig, VerifiedProtocol, canonical_sha256, load_config


def test_fixed_protocol_and_canonical_hash(study_config_path):
    cfg = load_config(study_config_path)
    assert isinstance(cfg, StudyConfig)
    assert cfg.split_numerator == 7 and cfg.split_denominator == 10
    assert cfg.support_caps == (128, 64)
    assert cfg.delta == 0.01 and cfg.bootstrap_seed == 20261029
    assert cfg.bootstrap_resamples == 10000
    assert cfg.model_revision is None and cfg.environment_hash is None
    assert canonical_sha256({"a": 1, "b": 2}) == canonical_sha256({"b": 2, "a": 1})
    assert canonical_sha256({"a": 1, "b": 2}) == (
        "43258cff783fe7036d8a43033f830adfc60ec037382473548ac742b888292777"
    )


def write_settings(path, settings):
    path.write_text(json.dumps(settings), encoding="utf-8")


@pytest.mark.parametrize("caps", [[0, 64], [129, 64], [128, 65], [128], [128.0, 64], [True, 64]])
def test_rejects_invalid_caps(study_config_path, study_settings, caps):
    study_settings["support_caps"] = caps
    write_settings(study_config_path, study_settings)
    with pytest.raises(ValueError, match="support_caps"):
        load_config(study_config_path)


@pytest.mark.parametrize("field,value", [
    ("delta", 0.005), ("schema_version", 2), ("schema_version", True),
    ("split_numerator", 6), ("model_id", "another/model"),
    ("eligibility", [10, 5, 4, 5]), ("bootstrap_resamples", 9999),
])
def test_rejects_changed_fixed_protocol(study_config_path, study_settings, field, value):
    study_settings[field] = value
    write_settings(study_config_path, study_settings)
    with pytest.raises(ValueError, match=field):
        load_config(study_config_path)


@pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity", "1e9999"])
def test_rejects_nonfinite_json(study_config_path, literal):
    text = study_config_path.read_text(encoding="utf-8")
    study_config_path.write_text(text.replace('"delta": 0.01', f'"delta": {literal}'), encoding="utf-8")
    with pytest.raises(ValueError, match="finite"):
        load_config(study_config_path)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_canonical_hash_rejects_nonfinite_values(value):
    with pytest.raises(ValueError):
        canonical_sha256({"nested": [value]})


def test_rejects_duplicate_and_unknown_keys(study_config_path, study_settings):
    text = study_config_path.read_text(encoding="utf-8")
    study_config_path.write_text(text.replace('"schema_version": 1', '"schema_version": 1, "schema_version": 1'), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        load_config(study_config_path)
    study_settings["unrecognized"] = 1
    write_settings(study_config_path, study_settings)
    with pytest.raises(ValueError, match="unrecognized"):
        load_config(study_config_path)


def test_rejects_missing_settings(study_config_path, study_settings):
    del study_settings["bootstrap_seed"]
    write_settings(study_config_path, study_settings)
    with pytest.raises(ValueError, match="bootstrap_seed"):
        load_config(study_config_path)


@pytest.mark.parametrize("group,field,value", [
    ("encoder", "device", "mps"), ("encoder", "inference_dtype", "float16"),
    ("numerics", "dtype", "float32"), ("linear_logistic", "tol", 1e-4),
    ("resources", "evaluation_limit", 100001), ("prototype", "max_updates", 75),
])
def test_rejects_changed_nested_settings(study_config_path, study_settings, group, field, value):
    study_settings[group][field] = value
    write_settings(study_config_path, study_settings)
    with pytest.raises(ValueError, match=field):
        load_config(study_config_path)


@pytest.mark.parametrize("path", ["../outside", "/tmp/results", ".", "data/../results"])
def test_rejects_unsafe_output_paths(study_config_path, study_settings, path):
    study_settings["paths"]["results"] = path
    write_settings(study_config_path, study_settings)
    with pytest.raises(ValueError, match="paths.results"):
        load_config(study_config_path)


def test_resolved_acquisition_outputs_are_validated(study_config_path, study_settings):
    study_settings["model_revision"] = "a" * 40
    study_settings["environment_hash"] = "b" * 64
    write_settings(study_config_path, study_settings)
    cfg = load_config(study_config_path)
    assert cfg.model_revision == "a" * 40 and cfg.environment_hash == "b" * 64
    study_settings["model_revision"] = "main"
    write_settings(study_config_path, study_settings)
    with pytest.raises(ValueError, match="model_revision"):
        load_config(study_config_path)


@pytest.mark.parametrize("group,field,value,kind", [
    ("prototype", "max_updates", 100, "prototype_iterations"),
    ("linear_logistic", "max_iter", 4000, "solver_iterations"),
    ("rbf_svm", "max_iter", 200000, "solver_iterations"),
])
def test_authorized_iteration_repairs_require_a_record(study_config_path, study_settings, group, field, value, kind):
    study_settings[group][field] = value
    write_settings(study_config_path, study_settings)
    with pytest.raises(ValueError, match="repair"):
        load_config(study_config_path)
    study_settings["pre_freeze_repairs"] = [{"kind": kind, "stage": "G2", "reason": "Calibration evidence; budget admitted."}]
    write_settings(study_config_path, study_settings)
    assert getattr(load_config(study_config_path), group)[field] == value


def test_loaded_settings_cannot_be_mutated(study_config_path):
    cfg = load_config(study_config_path)
    with pytest.raises(TypeError):
        cfg.resources["evaluation_limit"] = 1
    with pytest.raises(TypeError):
        cfg.resources["stage_seconds"]["G1"] = 1


def test_repository_config_matches_blueprint_contract(study_settings):
    path = Path(__file__).resolve().parents[1] / "config" / "study.json"
    assert json.loads(path.read_text(encoding="utf-8")) == study_settings
    cfg = load_config(path)
    assert cfg.paths["results"] == Path("results")


def test_verified_protocol_record_preserves_lock_identity():
    protocol = VerifiedProtocol(lock_hash="a" * 64, payload={"schema_version": 1})
    assert protocol.lock_hash == "a" * 64 and protocol.payload == {"schema_version": 1}


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


@pytest.fixture
def budget_case(tmp_path, local_config):
    from signed_study.budget import Ledger

    clock = FakeClock()
    resources = dict(installed_ram_bytes=32 * 1024**3, available_memory_bytes=16 * 1024**3,
                     rss_bytes=100 * 1024**2, project_disk_bytes=1000,
                     free_disk_bytes=20 * 1024**3)
    ledger = Ledger(tmp_path / 'results' / 'budget_ledger.jsonl', local_config,
                    clock=clock, wall_clock=lambda: 1_000_000 + clock.now,
                    sampler=lambda: dict(resources), project_root=tmp_path)
    yield ledger, clock, resources
    ledger.close()


def test_restart_preserves_attempts_and_report_reserve(budget_case, local_config):
    from signed_study.budget import Ledger

    ledger, clock, _ = budget_case
    attempt = ledger.admit('G3', projected_seconds=2, evaluations=78)
    reopened = Ledger(ledger.path, local_config)
    assert reopened.snapshot()['attempted_evaluations'] == 78
    assert attempt in reopened.snapshot()['unfinished_attempts']
    assert reopened.snapshot()['report_reserve_seconds'] == 1800
    clock.now = 3
    ledger.finish(attempt, 'failed', {}, {})
    assert ledger.snapshot()['experimental_seconds'] == 3
    assert ledger.snapshot()['attempted_evaluations'] == 78


@pytest.mark.parametrize('stage,seconds,evaluations,change,reason', [
    ('G3', 1, 100001, {}, 'evaluations'),
    ('G3', 1601, 0, {}, 'stage'),
    ('G2', 1, 0, {'available_memory_bytes': 4 * 1024**3 - 1}, 'headroom'),
    ('G3', 1, 0, {'free_disk_bytes': 12 * 1024**3 - 1}, 'free disk'),
    ('G3', 1, 0, {'rss_bytes': 8 * 1024**3 + 1}, 'RSS'),
    ('G3', 1, 0, {'installed_ram_bytes': 16 * 1024**3, 'rss_bytes': 4 * 1024**3 + 1}, 'RSS'),
])
def test_budget_refuses_before_work(budget_case, stage, seconds, evaluations, change, reason):
    from signed_study.budget import BudgetExceeded

    ledger, _, resources = budget_case
    resources.update(change)
    with pytest.raises(BudgetExceeded, match=reason):
        ledger.admit(stage, seconds, evaluations)
    assert ledger.snapshot()['attempted_evaluations'] == 0
    assert ledger.snapshot()['unfinished_attempts'] == {}


def test_report_time_cannot_be_consumed_by_other_stages(budget_case):
    from signed_study.budget import BudgetExceeded

    ledger, clock, _ = budget_case
    # An interrupted G4 is charged conservatively; stage time can never reset.
    attempt = ledger.admit('G4', 1, 0)
    clock.now = 12600
    with pytest.raises(BudgetExceeded):
        ledger.checkpoint()
    ledger.finish(attempt, 'budget_exceeded', {}, {})
    with pytest.raises(BudgetExceeded, match='report reserve'):
        ledger.admit('G0', 1, 0)
    report = ledger.admit('G6', 1000, 0)
    clock.now += 1
    ledger.finish(report, 'completed', {}, {})
    assert ledger.snapshot()['experimental_seconds'] == 12601


def test_one_writer_allows_read_only_snapshot(budget_case, local_config):
    from signed_study.budget import Ledger, LedgerBusy

    ledger, clock, resources = budget_case
    attempt = ledger.admit('G3', 1, 7)
    other = Ledger(ledger.path, local_config, clock=clock, sampler=lambda: dict(resources))
    assert other.snapshot()['attempted_evaluations'] == 7
    with pytest.raises(LedgerBusy):
        other.admit('G2', 1, 1)
    ledger.finish(attempt, 'completed', {}, {})
    second = other.admit('G2', 1, 1)
    other.finish(second, 'completed', {}, {})
    other.close()


def test_torn_record_recovery_and_conservative_reconciliation(budget_case, local_config):
    from signed_study.budget import Ledger

    ledger, clock, resources = budget_case
    attempt = ledger.admit('G3', 10, 78)
    clock.now = 4
    ledger.checkpoint()
    ledger.close()  # Process interruption: no successful finish record.
    with ledger.path.open('ab') as handle:
        handle.write(b'{"event":"finish","attempt_id":')
    clock.now = 20
    reopened = Ledger(ledger.path, local_config, clock=clock,
                      wall_clock=lambda: 1_000_000 + clock.now,
                      sampler=lambda: dict(resources))
    assert attempt in reopened.snapshot()['unfinished_attempts']
    following = reopened.admit('G3', 1, 1)
    reopened.finish(following, 'completed', {}, {})
    state = reopened.snapshot()
    assert state['attempted_evaluations'] == 79
    assert state['experimental_seconds'] == 20
    assert state['attempts'][attempt]['status'] == 'interrupted'
    assert any(event['event'] == 'recovery' for event in state['events'])
    reopened.close()


def test_fallback_does_not_reset_evaluation_or_disk_caps(budget_case, local_config):
    from dataclasses import replace
    from signed_study.budget import BudgetExceeded, Ledger

    ledger, _, resources = budget_case
    attempt = ledger.admit('G3', 1, 100000)
    ledger.finish(attempt, 'failed', {}, {})
    fallback = Ledger(ledger.path, replace(local_config, dataset='ml-100k'), sampler=lambda: dict(resources))
    with pytest.raises(BudgetExceeded, match='evaluations'):
        fallback.admit('G4', 1, 1)
    resources['project_disk_bytes'] = 1000 + 6 * 1024**3 + 1
    with pytest.raises(BudgetExceeded, match='disk growth'):
        fallback.admit('G6', 1, 0)
    assert fallback.snapshot()['attempted_evaluations'] == 100000
    fallback.close()


def test_setup_and_repairs_are_separate_and_cumulative(budget_case):
    from signed_study.budget import BudgetExceeded

    ledger, clock, _ = budget_case
    setup = ledger.admit('setup', 10, 0)
    clock.now = 20
    ledger.finish(setup, 'failed', {}, {})
    for _ in range(2):
        repair = ledger.admit('repair', 1, 1)
        clock.now += 1
        ledger.finish(repair, 'failed', {}, {})
    assert ledger.snapshot()['experimental_seconds'] == 2
    assert ledger.snapshot()['setup_seconds'] == 20
    with pytest.raises(BudgetExceeded, match='repair cycles'):
        ledger.admit('repair', 1, 1)
    with pytest.raises(BudgetExceeded, match='setup'):
        ledger.admit('setup', 1800, 0)


def test_imports_bootstrap_once_including_failures_and_original_disk_baseline(budget_case):
    ledger, _, _ = budget_case
    bootstrap = ledger.path.parent / 'bootstrap_accounting.jsonl'
    rows = [
        dict(event='end', record_id='baseline', category='setup', duration_seconds=1,
             project_disk_bytes=400, filesystem_free_bytes=20 * 1024**3),
        dict(event='start', record_id='red', category='validation', started_at='1970-01-12T13:46:40+00:00'),
        dict(event='end', record_id='red', category='validation', duration_seconds=3, returncode=1),
        dict(event='end', record_id='setup-failed', category='setup', duration_seconds=2, returncode=1),
    ]
    bootstrap.parent.mkdir(parents=True, exist_ok=True)
    bootstrap.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    for _ in range(2):
        attempt = ledger.admit('G1', 1, 0)
        ledger.finish(attempt, 'completed', {}, {})
    state = ledger.snapshot()
    assert state['setup_seconds'] == 3
    assert state['experimental_seconds'] == 3
    assert state['stage_seconds']['G1'] == 3
    assert state['baseline_disk_bytes'] == 400
    assert len(state['bootstrap_ids']) == 3
    assert state['attempted_evaluations'] == 0


def test_bootstrap_interruption_is_charged_and_never_imported_twice(budget_case):
    ledger, clock, _ = budget_case
    bootstrap = ledger.path.parent / 'bootstrap_accounting.jsonl'
    bootstrap.parent.mkdir(parents=True, exist_ok=True)
    bootstrap.write_text(json.dumps(dict(event='start', record_id='lost', category='validation',
                                         started_at='1970-01-12T13:46:30+00:00')) + '\n')
    first = ledger.admit('G1', 1, 0)
    ledger.finish(first, 'completed', {}, {})
    clock.now = 20
    second = ledger.admit('G1', 1, 0)
    ledger.finish(second, 'completed', {}, {})
    assert ledger.snapshot()['stage_seconds']['G1'] == 10
    assert ledger.snapshot()['bootstrap_ids'] == ['lost']


def test_monitor_enforces_current_rss_disk_and_elapsed(budget_case):
    from signed_study.budget import BudgetExceeded

    ledger, clock, resources = budget_case
    attempt = ledger.admit('G2', 1, 10)
    resources['rss_bytes'] = 9 * 1024**3
    with pytest.raises(BudgetExceeded, match='RSS'):
        ledger.checkpoint()
    ledger.finish(attempt, 'budget_exceeded', {}, {})
    assert ledger.snapshot()['peak_rss_bytes'] == 9 * 1024**3
    assert ledger.snapshot()['attempted_evaluations'] == 10


def test_watchdog_interrupts_and_restores_prior_signal_state(local_config, tmp_path):
    import signal
    import time
    from signed_study.budget import BudgetExceeded, Ledger

    resource = dict(installed_ram_bytes=32 * 1024**3, available_memory_bytes=16 * 1024**3,
                    rss_bytes=1, project_disk_bytes=1000, free_disk_bytes=20 * 1024**3)
    old_handler = signal.getsignal(signal.SIGALRM)
    old_timer = signal.getitimer(signal.ITIMER_REAL)
    prior_handler = lambda *_: None
    signal.signal(signal.SIGALRM, prior_handler)
    signal.setitimer(signal.ITIMER_REAL, 20, 30)
    ledger = Ledger(tmp_path / 'events.jsonl', local_config, sampler=lambda: dict(resource), monitor=True)
    try:
        with pytest.raises(BudgetExceeded, match='RSS'):
            with ledger.work('G2', 1, 3):
                resource['rss_bytes'] = 9 * 1024**3
                time.sleep(0.4)
        assert signal.getsignal(signal.SIGALRM) is prior_handler
        delay, interval = signal.getitimer(signal.ITIMER_REAL)
        assert 19 < delay <= 20 and interval == 30
        assert ledger.snapshot()['unfinished_attempts'] == {}
        assert ledger.snapshot()['attempted_evaluations'] == 3
        resource['rss_bytes'] = 1
        with ledger.work('G1', 1, 0):
            pass
    finally:
        ledger.close()
        signal.signal(signal.SIGALRM, old_handler)
        signal.setitimer(signal.ITIMER_REAL, *old_timer)


def test_atomic_artifact_rejects_nonfinite_without_replacing_existing(tmp_path):
    from signed_study.budget import atomic_write_json

    path = tmp_path / 'artifact.json'
    atomic_write_json(path, {'schema_version': 1, 'value': 4})
    with pytest.raises(ValueError):
        atomic_write_json(path, {'value': float('nan')})
    assert json.loads(path.read_text())['value'] == 4


def test_non_target_inspection_is_explicit_blocker(local_config, monkeypatch):
    from signed_study import run

    monkeypatch.setattr(run, '_hardware_evidence', lambda: {
        'system': 'Linux', 'machine': 'x86_64', 'chip': 'Other CPU',
        'installed_ram_bytes': 16 * 1024**3, 'logical_cpus': 8,
        'translated': False, 'power': {'ac': True, 'low_power_mode': False},
        'commands': {},
    })
    import signed_study.budget as budget
    resources = dict(installed_ram_bytes=32 * 1024**3, available_memory_bytes=16 * 1024**3,
                     rss_bytes=1, project_disk_bytes=1000, free_disk_bytes=20 * 1024**3)
    monkeypatch.setattr(budget, 'ResourceSampler', lambda root: lambda: dict(resources))
    evidence = run.inspect_environment(local_config)
    assert evidence['target_compatible'] is False
    assert any('M3 Max' in reason for reason in evidence['blockers'])
    assert evidence['hardware']['chip'] == 'Other CPU'
    assert evidence['experimental_workload'] is False
    saved = json.loads((local_config.paths['results'] / 'environment.json').read_text())
    assert saved['output_sha256'] == canonical_sha256({k: v for k, v in saved.items() if k != 'output_sha256'})
    state = budget.Ledger(local_config.paths['results'] / 'budget_ledger.jsonl', local_config).snapshot()
    attempt = next(iter(state['attempts'].values()))
    assert attempt['status'] == 'blocked'
    assert attempt['outputs']['environment']['sha256'] == hashlib.sha256((local_config.paths['results'] / 'environment.json').read_bytes()).hexdigest()


def test_finish_rechecks_disk_and_releases_ownership(budget_case, local_config):
    from signed_study.budget import BudgetExceeded, Ledger

    ledger, clock, resources = budget_case
    attempt = ledger.admit('G2', 1, 1)
    resources['project_disk_bytes'] = 1000 + 6 * 1024**3 + 1
    with pytest.raises(BudgetExceeded, match='disk growth'):
        ledger.finish(attempt, 'completed', {}, {})
    assert ledger.snapshot()['attempts'][attempt]['status'] == 'budget_exceeded'
    assert ledger.snapshot()['peak_project_disk_bytes'] == 1000 + 6 * 1024**3 + 1
    resources['project_disk_bytes'] = 1000
    other = Ledger(ledger.path, local_config, clock=clock, sampler=lambda: dict(resources))
    with other.work('G6', 1, 0):
        pass
    other.close()


def test_work_failure_preserves_evaluations_and_releases_monitor(budget_case):
    ledger, clock, _ = budget_case
    with pytest.raises(RuntimeError, match='work failed'):
        with ledger.work('G1', 1, 5):
            clock.now = 2
            raise RuntimeError('work failed')
    assert ledger.snapshot()['unfinished_attempts'] == {}
    assert ledger.snapshot()['experimental_seconds'] == 2
    assert ledger.snapshot()['attempted_evaluations'] == 5
    with ledger.work('G1', 1, 1):
        pass
    assert ledger.snapshot()['attempted_evaluations'] == 6


@pytest.mark.parametrize('stage', ['setup', 'G6'])
def test_evaluation_work_cannot_use_setup_or_reporting_reserve(budget_case, stage):
    ledger, _, _ = budget_case
    with pytest.raises(ValueError, match='evaluations'):
        ledger.admit(stage, 1, 1)
    assert ledger.snapshot()['attempted_evaluations'] == 0


def test_current_low_power_api_handles_modern_pmset_output(monkeypatch):
    from signed_study import run

    def command(argv):
        outputs = {'uname': 'arm64', 'sw_vers': 'macOS 26.7.1',
                   'sysctl': {'machdep.cpu.brand_string': 'Apple M3 Max', 'hw.memsize': '38654705664',
                              'hw.logicalcpu': '14', 'sysctl.proc_translated': '0'}.get(argv[-1], ''),
                   'pmset': "Now drawing from 'AC Power'" if argv[-1] == 'batt' else 'AC Power:\n powermode 0',
                   'osascript': 'false', 'df': '71Gi available'}
        return dict(command=argv, returncode=0, stdout=outputs[argv[0]], stderr='')

    monkeypatch.setattr(run, '_command', command)
    hardware = run._hardware_evidence()
    assert hardware['power']['low_power_mode'] is False
    assert hardware['commands']['power_mode']['stdout'] == 'AC Power:\n powermode 0'
    assert hardware['power']['ac'] is True


def test_recorded_command_counts_failed_validation_and_preserves_outputs(budget_case):
    import sys

    ledger, _, _ = budget_case
    code = ledger.run_command([sys.executable, '-c', 'raise SystemExit(3)'],
                              stage='G1', projected_seconds=1)
    assert code == 3
    attempt = next(iter(ledger.snapshot()['attempts'].values()))
    assert attempt['status'] == 'failed'
    assert attempt['outputs']['returncode'] == 3
    assert attempt['outputs']['command'][0] == sys.executable
    assert ledger.snapshot()['attempted_evaluations'] == 0


def test_recorded_command_terminates_child_on_limit(budget_case, tmp_path, monkeypatch):
    import sys
    from signed_study.budget import BudgetExceeded

    ledger, _, _ = budget_case
    child_pid = tmp_path / 'child.pid'
    original = ledger.checkpoint

    def limit_when_running():
        if child_pid.exists():
            raise BudgetExceeded('synthetic command limit')
        return original()

    monkeypatch.setattr(ledger, 'checkpoint', limit_when_running)
    with pytest.raises(BudgetExceeded, match='synthetic command limit'):
        ledger.run_command([sys.executable, '-c',
                            'import os,pathlib,time; pathlib.Path(__import__("sys").argv[1]).write_text(str(os.getpid())); time.sleep(20)',
                            str(child_pid)], stage='G1', projected_seconds=1)
    import os
    with pytest.raises(ProcessLookupError):
        os.kill(int(child_pid.read_text()), 0)
    assert ledger.snapshot()['unfinished_attempts'] == {}


def test_process_sampler_permission_failure_is_explicit_blocker(tmp_path, monkeypatch):
    import signed_study.budget as budget

    class RestrictedProcess:
        def memory_info(self):
            from types import SimpleNamespace
            return SimpleNamespace(rss=1000)

        def children(self, recursive):
            raise PermissionError('host process query denied')

    monkeypatch.setattr(budget.psutil, 'Process', RestrictedProcess)
    with pytest.raises(budget.BudgetExceeded, match='process-tree measurement unavailable'):
        budget.ResourceSampler(tmp_path)()


def test_inspect_cli_reports_configured_artifact_location(local_config, monkeypatch, capsys):
    from dataclasses import replace
    from signed_study import __main__, protocol, run

    settings = replace(local_config, paths=dict(local_config.paths, results=Path('results/nested')))
    monkeypatch.setattr(protocol, 'load_config', lambda path: settings)
    monkeypatch.setattr(run, 'inspect_environment', lambda config: dict(target_compatible=True, blockers=[]))
    assert __main__.main(['inspect', '--config', 'study.json']) == 0
    assert json.loads(capsys.readouterr().out)['environment'] == 'results/nested/environment.json'
