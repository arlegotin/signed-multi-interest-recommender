"""Append-only, cumulative study accounting and bounded single-worker admission.

Use ``with ledger.work(stage, projected_seconds, evaluations)`` for workloads;
``checkpoint()`` is also available at loop/write boundaries. Production admission
installs a 100 ms SIGALRM guard on the main thread. Python handles signals only
when native calls return to its interpreter: sampling/interrupts can be delayed
by native code, so use bounded chunks and checkpoints as well. Disk walks occur
at stage boundaries and at most every five seconds, not per evaluation.

Injection: clock()/wall_clock() return monotonic/UTC epoch seconds respectively;
sampler() returns integer installed_ram_bytes, available_memory_bytes, rss_bytes
(process tree), project_disk_bytes, free_disk_bytes. Missing measurements fail
closed. Injected clocks disable the automatic guard unless monitor=True.
"""

from contextlib import contextmanager
from datetime import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import shutil
import signal
import subprocess
import tempfile
import threading
import time
import uuid

import psutil

from .protocol import StudyConfig, canonical_sha256


class BudgetExceeded(RuntimeError):
    """A fixed resource ceiling would be exceeded or has been reached."""


class CommandCleanupError(RuntimeError):
    """An admitted command group is still live or cannot be verified stopped."""


class LedgerBusy(RuntimeError):
    """Another writer owns this study's workload lock."""


def atomic_write_json(path: Path, payload: dict) -> None:
    """Replace a complete JSON artifact atomically; preserve old output on error."""
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, allow_nan=False) + '\n'
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8') as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        Path(temporary).unlink(missing_ok=True)


class ResourceSampler:
    """Current process-tree RSS and system headroom; cached whole-project bytes."""

    def __init__(self, root: Path):
        self.root = root
        self._disk_at = -math.inf
        self._disk_bytes = 0

    def refresh_disk(self):
        total = 0
        for current, _, files in os.walk(self.root):
            for name in files:
                try:
                    total += (Path(current) / name).lstat().st_size
                except FileNotFoundError:
                    pass
        self._disk_bytes = total
        self._disk_at = time.monotonic()

    def __call__(self):
        if time.monotonic() - self._disk_at >= 5:
            self.refresh_disk()
        process = psutil.Process()
        rss = process.memory_info().rss
        try:
            for child in process.children(recursive=True):
                try:
                    rss += child.memory_info().rss
                except psutil.NoSuchProcess:
                    pass
        except (PermissionError, psutil.AccessDenied) as error:
            raise BudgetExceeded('Current process-tree measurement unavailable: ' + str(error)) from error
        memory = psutil.virtual_memory()
        return dict(installed_ram_bytes=memory.total, available_memory_bytes=memory.available,
                    rss_bytes=rss, project_disk_bytes=self._disk_bytes,
                    free_disk_bytes=shutil.disk_usage(self.root).free)


def _command_group_state(pgid, known_pids=()):
    """Find live members independently of the reaped/reparented group leader.

    Terminal zombies cannot execute work and may belong to another reaper. Keep
    their PIDs as evidence instead of waiting forever for somebody else's wait().
    """
    state = dict(process_group_id=pgid, live_pids=[], terminal_pids=[])
    try:
        for pid in set(psutil.pids()).union(known_pids):
            try:
                if pid in known_pids or os.getpgid(pid) == pgid:
                    status = psutil.Process(pid).status()
                    key = 'terminal_pids' if status in (psutil.STATUS_ZOMBIE, psutil.STATUS_DEAD) else 'live_pids'
                    state[key].append(pid)
            except (ProcessLookupError, psutil.NoSuchProcess):
                pass
    except (PermissionError, psutil.AccessDenied) as error:
        raise CommandCleanupError(f'Command group {pgid} cleanup membership unavailable: {error}') from error
    return state


def _stop_command_group(process):
    """Bound TERM/KILL cleanup by group membership, never leader wait alone."""
    process.poll()  # Reap our direct child when possible; it is not the whole group.
    state = _command_group_state(process.pid)
    had_live_members = bool(state['live_pids'])
    known_pids = set(state['live_pids'])
    signals = []
    for sig in (signal.SIGTERM, signal.SIGKILL):
        if not state['live_pids']:
            break
        try:
            os.killpg(process.pid, sig)
            signals.append(sig.name)
        except ProcessLookupError:
            pass
        except PermissionError as error:
            raise CommandCleanupError(f'Command group {process.pid} cleanup signal denied: {error}') from error
        deadline = time.monotonic() + 1
        while True:
            process.poll()
            # macOS can remove group membership before a dying PID becomes
            # terminal in process-status queries. Verify observed members too.
            state = _command_group_state(process.pid, known_pids)
            known_pids.update(state['live_pids'])
            if not state['live_pids'] or time.monotonic() >= deadline:
                break
            time.sleep(min(.02, max(0, deadline - time.monotonic())))
    if state['live_pids']:
        raise CommandCleanupError(f'Command group {process.pid} cleanup timed out with live members {state["live_pids"]}')
    return dict(state, had_live_members=had_live_members, signals=signals, verified_no_live_members=True)


def _plain(value):
    if hasattr(value, 'items'):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    return value


def _read_events(path):
    """Only an unterminated final record may be ignored; other corruption blocks."""
    if not path.exists():
        return [], b''
    data = path.read_bytes()
    rows, tail = [], b''
    for index, line in enumerate(data.splitlines(keepends=True)):
        if not line.endswith(b'\n'):
            tail = line
            break
        try:
            row = json.loads(line, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
        except (ValueError, UnicodeError) as error:
            raise ValueError(f'Corrupt ledger record {index + 1}') from error
        if not isinstance(row, dict):
            raise ValueError(f'Corrupt ledger record {index + 1}')
        rows.append(row)
    return rows, tail


class Ledger:
    """Construction/snapshot are read-only; a workload owns the lock until finish.

    close() releases OS resources without inventing a successful finish. The next
    writer reconciles unfinished work as max(forecast*1.5, last elapsed, wall gap)
    with an explicit interrupted record; evaluations never roll back. Resuming a
    different dataset/model cannot change the resource caps or original baseline.
    """

    def __init__(self, path: Path, config: StudyConfig, *, clock=None, wall_clock=None,
                 sampler=None, project_root=None, monitor=None):
        self.path = Path(path).resolve()
        self.config = config
        self.clock = clock or time.monotonic
        self.wall_clock = wall_clock or time.time
        self.sampler = sampler or ResourceSampler(Path(project_root or Path.cwd()).resolve())
        self.monitor = (clock is None) if monitor is None else monitor
        self._lock = None
        self._active = None
        self._started = None
        self._peak = 0
        self._last_checkpoint = -math.inf
        self._timer = None
        self._cached = None

    @contextmanager
    def _critical(self):
        # Never let our interval interrupt append/fsync or ownership transitions.
        previous = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGALRM})
        try:
            yield
        finally:
            signal.pthread_sigmask(signal.SIG_SETMASK, previous)

    def _append(self, event, **fields):
        row = dict(schema_version=1, event=event, recorded_at=self.wall_clock(), **fields)
        encoded = json.dumps(row, sort_keys=True, allow_nan=False) + '\n'
        with self.path.open('a', encoding='utf-8') as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        if self._cached is not None:
            self._cached.append(row)

    def _acquire(self):
        if self._lock is not None:
            raise LedgerBusy('This ledger already has an admitted workload')
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.with_suffix(self.path.suffix + '.lock').open('a')
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            handle.close()
            raise LedgerBusy('Another writer owns the study ledger') from error
        self._lock = handle

    def _release(self):
        if self._lock is not None:
            fcntl.flock(self._lock, fcntl.LOCK_UN)
            self._lock.close()
            self._lock = None
        self._cached = None

    def _sample(self, boundary=False):
        if boundary and hasattr(self.sampler, 'refresh_disk'):
            self.sampler.refresh_disk()
        measurements = self.sampler()
        required = ('installed_ram_bytes', 'available_memory_bytes', 'rss_bytes',
                    'project_disk_bytes', 'free_disk_bytes')
        for key in required:
            value = measurements.get(key)
            if type(value) is not int or value < 0 or (key == 'installed_ram_bytes' and value == 0):
                raise BudgetExceeded(f'Missing/invalid resource measurement: {key}')
        self._peak = max(self._peak, measurements['rss_bytes'])
        return measurements

    def _bootstrap(self):
        path = self.path.parent / 'bootstrap_accounting.jsonl'
        rows, tail = _read_events(path)
        if tail:
            raise ValueError('Unterminated bootstrap accounting record; preserve/repair evidence first')
        return rows

    def _initialize(self):
        rows, tail = _read_events(self.path)
        self._cached = rows
        if tail:
            # Only the uncommitted tail is removed; preserve it durably first.
            # Completed event lines remain append-only, even across this recovery.
            recovery_path = self.path.with_name(self.path.name + '.torn.' + uuid.uuid4().hex)
            with recovery_path.open('xb') as handle:
                handle.write(tail)
                handle.flush()
                os.fsync(handle.fileno())
            with self.path.open('r+b') as handle:
                handle.truncate(self.path.stat().st_size - len(tail))
                handle.flush()
                os.fsync(handle.fileno())
            self._append('recovery', torn_sha256=hashlib.sha256(tail).hexdigest(),
                         preserved_at=str(recovery_path), reason='unterminated final record')
        resources_hash = canonical_sha256(_plain(self.config.resources))
        bootstrap = self._bootstrap()
        state = self._state(self._cached)
        if not any(row['event'] == 'initialize' for row in self._cached):
            sample = self._sample(boundary=True)
            disk_values = [row.get('project_disk_start_bytes', row.get('project_disk_bytes')) for row in bootstrap]
            baseline = next((value for value in disk_values if value is not None), sample['project_disk_bytes'])
            free_values = [row['filesystem_free_bytes'] for row in bootstrap if 'filesystem_free_bytes' in row]
            self._append('initialize', resources_sha256=resources_hash, baseline_disk_bytes=baseline,
                         initial_free_disk_bytes=free_values[0] if free_values else sample['free_disk_bytes'])
        elif state['resources_sha256'] != resources_hash:
            raise ValueError('Resource caps differ from the persistent ledger')
        imported = set(state['bootstrap_ids'])
        grouped = {}
        for row in bootstrap:
            grouped.setdefault(row['record_id'], []).append(row)
        for record_id, records in grouped.items():
            if record_id in imported:
                continue
            endings = [r for r in records if r['event'] in ('end', 'interrupted')]
            last = endings[-1] if endings else records[-1]
            elapsed = last.get('duration_seconds')
            interrupted = not endings or last['event'] == 'interrupted'
            if elapsed is None:
                start = datetime.fromisoformat(last['started_at']).timestamp()
                elapsed = max(0, self.wall_clock() - start)
            if not isinstance(elapsed, (int, float)) or not math.isfinite(elapsed) or elapsed < 0:
                raise ValueError('Invalid bootstrap elapsed time')
            stage = 'setup' if last['category'] == 'setup' else 'G1'
            self._append('bootstrap_import', record_id=record_id, stage=stage,
                         elapsed_seconds=elapsed, interrupted=interrupted, source_records=records)
        for attempt_id, attempt in self._state(self._cached)['unfinished_attempts'].items():
            if attempt.get('process_group_id') is not None:
                group = _command_group_state(attempt['process_group_id'])
                if group['live_pids']:
                    raise CommandCleanupError(f'Unfinished command group {attempt["process_group_id"]} requires cleanup before admission')
            elapsed = max(attempt['forecast_seconds'], attempt.get('elapsed_seconds', 0),
                          max(0, self.wall_clock() - attempt['started_at']))
            self._append('finish', attempt_id=attempt_id, status='interrupted', elapsed_seconds=elapsed,
                         measurements=attempt.get('measurements', {}), outputs={},
                         reconciliation='max(safety forecast, last elapsed, UTC wall gap); idle gap conservatively charged')

    def _state(self, rows):
        stages = {stage: 0.0 for stage in self.config.resources['stage_seconds']}
        state = dict(schema_version=1, attempted_evaluations=0, experimental_seconds=0.0,
                     setup_seconds=0.0, stage_seconds=stages, repair_cycles=0, peak_rss_bytes=0,
                     baseline_disk_bytes=0, peak_project_disk_bytes=0, initial_free_disk_bytes=None, resources_sha256=None,
                     report_reserve_seconds=self.config.resources['report_reserve_seconds'],
                     attempts={}, unfinished_attempts={}, bootstrap_ids=[], events=rows)
        for row in rows:
            event = row['event']
            if event == 'initialize':
                for key in ('baseline_disk_bytes', 'initial_free_disk_bytes', 'resources_sha256'):
                    state[key] = row[key]
            elif event == 'bootstrap_import':
                state['bootstrap_ids'].append(row['record_id'])
                self._charge(state, row['stage'], row['elapsed_seconds'])
            elif event == 'start':
                state['attempts'][row['attempt_id']] = dict(row, status='unfinished', elapsed_seconds=0)
                state['attempted_evaluations'] += row['evaluations']
                state['repair_cycles'] += row['stage'] == 'repair'
            elif event in ('checkpoint', 'finish', 'command_started', 'cleanup_blocked'):
                attempt = state['attempts'][row['attempt_id']]
                attempt.update(row)
            elif event == 'overhead':
                self._charge(state, row['stage'], row['elapsed_seconds'])
            sample = row.get('measurements', {})
            state['peak_rss_bytes'] = max(state['peak_rss_bytes'], sample.get('rss_bytes', 0))
            state['peak_project_disk_bytes'] = max(state['peak_project_disk_bytes'], sample.get('project_disk_bytes', 0))
        for attempt_id, attempt in state['attempts'].items():
            elapsed = attempt['elapsed_seconds']
            if attempt['status'] == 'unfinished':
                state['unfinished_attempts'][attempt_id] = attempt
                # Do not estimate live foreign-owner wall time from a different injected clock.
                if attempt_id == self._active:
                    elapsed = max(elapsed, self.clock() - self._started)
            self._charge(state, attempt['stage'], elapsed)
        return state

    @staticmethod
    def _charge(state, stage, elapsed):
        if stage == 'setup':
            state['setup_seconds'] += elapsed
        else:
            state['experimental_seconds'] += elapsed
            state['stage_seconds'][stage] += elapsed

    def snapshot(self):
        rows, _ = _read_events(self.path)
        return self._state(rows)

    def _limits(self, state, stage, projected, evaluations, sample, admission=False):
        caps = self.config.resources
        if sample['rss_bytes'] > min(caps['rss_bytes'], sample['installed_ram_bytes'] * caps['rss_ram_fraction']):
            raise BudgetExceeded('Process-tree RSS cap exceeded')
        if sample['project_disk_bytes'] - state['baseline_disk_bytes'] > caps['disk_growth_bytes']:
            raise BudgetExceeded('Project disk growth cap exceeded')
        if sample['free_disk_bytes'] <= 0:
            raise BudgetExceeded('No free disk remaining')
        if admission and state['initial_free_disk_bytes'] < caps['initial_free_disk_bytes']:
            raise BudgetExceeded('Insufficient initial free disk')
        if admission and stage in ('G2', 'G3', 'G4', 'G5', 'repair'):
            if sample['available_memory_bytes'] < caps['available_memory_headroom_bytes']:
                raise BudgetExceeded('Insufficient available-memory headroom')
        if state['attempted_evaluations'] + evaluations > caps['evaluation_limit']:
            raise BudgetExceeded('Attempted evaluations cap exceeded')
        if stage == 'setup':
            if state['setup_seconds'] + projected > caps['setup_seconds']:
                raise BudgetExceeded('Separate setup time cap exceeded')
            return
        total = state['experimental_seconds'] + projected
        if stage != 'G6' and total > caps['experimental_seconds'] - caps['report_reserve_seconds']:
            raise BudgetExceeded('Protected report reserve would be consumed')
        if total > caps['experimental_seconds']:
            raise BudgetExceeded('Experimental time cap exceeded')
        if state['stage_seconds'][stage] + projected > caps['stage_seconds'][stage]:
            raise BudgetExceeded(f'{stage} stage time allocation exceeded')
        if stage == 'repair' and admission and state['repair_cycles'] >= caps['repair_cycles']:
            raise BudgetExceeded('Material repair cycles exhausted')

    def admit(self, stage: str, projected_seconds: float, evaluations: int) -> str:
        if stage not in ('setup', *self.config.resources['stage_seconds']):
            raise ValueError('Unknown budget stage')
        if type(evaluations) is not int or evaluations < 0:
            raise ValueError('evaluations must be a nonnegative integer')
        if stage in ('setup', 'G6') and evaluations:
            raise ValueError('Setup/reporting cannot admit model evaluations')
        if isinstance(projected_seconds, bool) or not math.isfinite(projected_seconds) or projected_seconds < 0:
            raise ValueError('projected_seconds must be finite and nonnegative')
        if self.monitor and threading.current_thread() is not threading.main_thread():
            raise RuntimeError('Production monitoring requires the main thread')
        with self._critical():
            self._acquire()
            began = self.clock()
            try:
                self._initialize()
                sample = self._sample(boundary=True)
                self._append('overhead', stage=stage, elapsed_seconds=max(0, self.clock() - began), measurements=sample)
                began = self.clock()
                state = self._state(self._cached)
                forecast = projected_seconds * self.config.resources['safety_multiplier']
                self._limits(state, stage, forecast, evaluations, sample, admission=True)
                attempt_id = str(uuid.uuid4())
                self._append('start', attempt_id=attempt_id, stage=stage, evaluations=evaluations,
                             projected_seconds=projected_seconds, forecast_seconds=forecast,
                             started_at=self.wall_clock(), measurements=sample, pid=os.getpid())
                self._active, self._started = attempt_id, began
                self._peak = sample['rss_bytes']
                self._last_checkpoint = began
                self._start_watchdog()
                return attempt_id
            except BaseException:
                self._stop_watchdog()
                self._active = None
                self._release()
                raise

    def checkpoint(self):
        if self._active is None:
            raise RuntimeError('No admitted workload')
        with self._critical():
            sample = self._sample()
            elapsed = max(0, self.clock() - self._started)
            state = self._state(self._cached)
            stage = state['attempts'][self._active]['stage']
            try:
                self._limits(state, stage, 0, 0, sample)
            except BudgetExceeded:
                self._append('checkpoint', attempt_id=self._active, elapsed_seconds=elapsed, measurements=sample)
                raise
            if self.clock() - self._last_checkpoint >= 5:
                self._append('checkpoint', attempt_id=self._active, elapsed_seconds=elapsed, measurements=sample)
                self._last_checkpoint = self.clock()
        return sample

    def finish(self, attempt_id: str, status: str, measurements: dict, outputs: dict) -> None:
        if attempt_id != self._active or self._lock is None:
            raise ValueError('Attempt is not owned by this writer')
        if status not in ('completed', 'failed', 'interrupted', 'budget_exceeded', 'blocked'):
            raise ValueError('Unknown attempt status')
        requested_status = status
        with self._critical():
            self._stop_watchdog()
            violation = None
            try:
                sample = self._sample(boundary=True)
                state = self._state(self._cached)
                stage = state['attempts'][attempt_id]['stage']
                try:
                    self._limits(state, stage, 0, 0, sample)
                except BudgetExceeded as error:
                    status = 'budget_exceeded'
                    violation = error
                combined = dict(measurements, **sample)
                combined['rss_bytes'] = max(combined['rss_bytes'], self._peak)
                combined['os_max_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if os.uname().sysname == 'Darwin' else 1024)
                self._append('finish', attempt_id=attempt_id, status=status,
                             elapsed_seconds=max(0, self.clock() - self._started),
                             measurements=combined, outputs=outputs)
            finally:
                self._active = None
                self._release()
            if violation is not None and requested_status == 'completed':
                raise violation

    def _start_watchdog(self):
        if not self.monitor:
            return
        self._timer = (signal.getsignal(signal.SIGALRM), signal.getitimer(signal.ITIMER_REAL), self.clock())
        signal.signal(signal.SIGALRM, self._alarm)
        interval = self.config.resources['rss_sample_seconds']
        signal.setitimer(signal.ITIMER_REAL, interval, interval)

    def _stop_watchdog(self):
        if self._timer is None:
            return
        signal.setitimer(signal.ITIMER_REAL, 0)
        # A blocked tick from our timer must not reach the restored handler.
        if signal.SIGALRM in signal.sigpending():
            signal.sigwait({signal.SIGALRM})
        handler, (delay, interval), began = self._timer
        self._timer = None
        signal.signal(signal.SIGALRM, handler)
        elapsed = max(0, self.clock() - began)
        if delay > 0:
            signal.setitimer(signal.ITIMER_REAL, max(0.000001, delay - elapsed), interval)

    def _alarm(self, *_):
        try:
            self.checkpoint()
        except BaseException:
            # The owning context/command cleans up its work before releasing the lock.
            with self._critical():
                self._stop_watchdog()
            raise

    @contextmanager
    def work(self, stage: str, projected_seconds: float, evaluations: int):
        attempt = self.admit(stage, projected_seconds, evaluations)
        try:
            yield attempt
            if self._active == attempt:
                self.checkpoint()
        except BaseException as error:
            if self._active == attempt:
                status = 'budget_exceeded' if isinstance(error, BudgetExceeded) else 'failed'
                self.finish(attempt, status, {'exception': str(error)}, {})
            raise
        else:
            if self._active == attempt:
                self.finish(attempt, 'completed', {}, {})

    def run_command(self, command: list[str], *, stage='G1', projected_seconds=60) -> int:
        """Account one validation/setup command including bounded group cleanup.

        Never wrap a CLI stage already using this ledger. Commands must join and
        wait for their work: even normal leader exit with live descendants fails
        after terminating that group. Unverifiable cleanup leaves an unfinished
        command that blocks admission on reopen, including after close()/exit.
        """
        if stage not in ('G1', 'setup'):
            raise ValueError('Command accounting is for G1 validation or setup')
        attempt = self.admit(stage, projected_seconds, evaluations=0)
        process = None
        cleanup = None
        try:
            with self._critical():
                process = subprocess.Popen(command, start_new_session=True)
                self._append('command_started', attempt_id=attempt, process_group_id=process.pid,
                             command=command)
            while True:
                try:
                    code = process.wait(timeout=self.config.resources['rss_sample_seconds'])
                    break
                except subprocess.TimeoutExpired:
                    self.checkpoint()
            self.checkpoint()
            with self._critical():
                self._stop_watchdog()
                cleanup = _stop_command_group(process)
            if cleanup['had_live_members']:
                raise CommandCleanupError('Command leader exited with live descendants; group cleanup was required')
        except BaseException as error:
            with self._critical():
                self._stop_watchdog()
                if process is not None and cleanup is None:
                    try:
                        cleanup = _stop_command_group(process)
                    except CommandCleanupError as cleanup_error:
                        self._append('cleanup_blocked', attempt_id=attempt, process_group_id=process.pid,
                                     elapsed_seconds=max(0, self.clock() - self._started),
                                     cleanup_error=str(cleanup_error), original_exception=str(error))
                        # Keep the attempt/lock owned; close or process exit may
                        # release OS resources, but reopening still refuses work.
                        raise cleanup_error from error
                status = 'budget_exceeded' if isinstance(error, BudgetExceeded) else 'failed'
                self.finish(attempt, status, {'exception': str(error)},
                            dict(command=command, returncode=process.returncode if process else None,
                                 process_group_cleanup=cleanup))
            raise
        self.finish(attempt, 'completed' if code == 0 else 'failed', {},
                    dict(command=command, returncode=code, process_group_cleanup=cleanup))
        return code

    def close(self):
        with self._critical():
            self._stop_watchdog()
            self._active = None
            self._release()
