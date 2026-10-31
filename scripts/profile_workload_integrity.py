#!/usr/bin/env python3
"""Profile synthetic workload bookkeeping against the baseline implementation.

Loads baseline workload source read-only from Git, injects fixed fit/score/resource
and clock boundaries, and writes only temporary synthetic evidence. Records every
repeat and exact snapshot/event volume. Timings include cProfile overhead; traced
peak allocations are measured in separate runs. This does not time study fitting.
"""
import argparse
import cProfile
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import platform
import pstats
import statistics
import subprocess
import sys
import tempfile
import time
import tracemalloc
from types import MappingProxyType, ModuleType

import numpy as np

from signed_study import workload
from signed_study.budget import atomic_write_json
from signed_study.data import Episode, EpisodeStore, History, episode_store_digest
from signed_study.metrics import evaluate_user
from signed_study.models import ModelConfig, UserModel
from signed_study.protocol import canonical_sha256, load_config


class Clock:
    def __init__(self):
        self.ns = 0

    def perf_counter_ns(self):
        return self.ns


class Resources:
    """Explicit synthetic boundary; this profile never claims native accounting."""

    def checkpoint(self):
        return dict(installed_ram_bytes=32 * 1024**3, available_memory_bytes=16 * 1024**3,
                    rss_bytes=100 * 1024**2, project_disk_bytes=1000, free_disk_bytes=20 * 1024**3)


def profile_once(module, n, root, *, memory=False):
    output = root / 'work.json'
    config = replace(load_config(Path('config/study.json')), paths=MappingProxyType({
        'cache': root / 'cache', 'data': root / 'data', 'results': root}))
    candidates = tuple(range(1000, 1384))
    rows = tuple(Episode(History(i, 500, 'development', tuple(range(10)), tuple(range(10, 15))),
                         candidates) for i in range(1, n + 1))
    store = EpisodeStore(rows, (), {}, {})
    labels = np.arange(len(candidates), dtype=np.int8) % 2
    entry = ModelConfig('SignedMax', {'K': 2, 'lambda': 2}, 'fixed-synthetic')
    clock = Clock()
    boundaries = dict(fit=0., score=0., metrics=0., validation=0., resource=0., snapshot=0.)
    snapshots = []

    def fit(*args, **kwargs):
        start = time.perf_counter_ns()
        clock.ns += 1_000_000
        value = UserModel('SignedMax', dict(entry.parameters), (), {}, {'status': 'ok', 'failures': []})
        boundaries['fit'] += (time.perf_counter_ns() - start) / 1e9
        return value

    def score(model, ids, cache, **kwargs):
        start = time.perf_counter_ns()
        clock.ns += 2_000_000
        value = np.arange(len(ids), dtype=np.float64)
        boundaries['score'] += (time.perf_counter_ns() - start) / 1e9
        return value

    def metrics(*args, **kwargs):
        start = time.perf_counter_ns()
        value = evaluate_user(*args, **kwargs)
        boundaries['metrics'] += (time.perf_counter_ns() - start) / 1e9
        return value

    def outcomes(*args):
        start = time.perf_counter_ns()
        value = labels.copy()
        boundaries['validation'] += (time.perf_counter_ns() - start) / 1e9
        return value

    def digest(value):
        start = time.perf_counter_ns()
        value = episode_store_digest(value)
        boundaries['validation'] += (time.perf_counter_ns() - start) / 1e9
        return value

    def snapshot(path, value):
        start = time.perf_counter_ns()
        atomic_write_json(path, value)
        boundaries['snapshot'] += (time.perf_counter_ns() - start) / 1e9
        snapshots.append(dict(attempt_rows=len(value['measurements']['attempts']), bytes=Path(path).stat().st_size))

    resources = Resources()
    original_checkpoint = resources.checkpoint

    def checkpoint():
        start = time.perf_counter_ns()
        value = original_checkpoint()
        boundaries['resource'] += (time.perf_counter_ns() - start) / 1e9
        return value

    resources.checkpoint = checkpoint
    module.time, module.fit_user, module.score_user = clock, fit, score
    module.evaluate_user, module.load_development_outcomes = metrics, outcomes
    module.episode_store_digest, module.atomic_write_json = digest, snapshot
    profiler = cProfile.Profile()
    if memory:
        tracemalloc.start()
    start = time.perf_counter_ns()
    if not memory:
        profiler.enable()
    value = module.measure_users(store, None, list(range(1, n + 1)), config, resources,
        numerical_checks=False, output_path=output, declared_entries=[entry])
    if not memory:
        profiler.disable()
    elapsed = (time.perf_counter_ns() - start) / 1e9
    peak = None
    if memory:
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    journal = output.with_suffix('.jsonl')
    data = journal.read_bytes()
    journal_seconds = replay_seconds = 0.
    if not memory:
        for (filename, _, function), (_, _, _, cumulative, _) in pstats.Stats(profiler).stats.items():
            if (function == 'journal' and filename.startswith('baseline_')
                    or function == 'append' and filename.endswith('workload_journal.py')):
                journal_seconds += cumulative
            if function == 'replay_workload_journal' and filename.endswith('workload_journal.py'):
                replay_seconds += cumulative
    boundaries.update(journal=journal_seconds, final_replay=replay_seconds)
    boundaries['remaining_preparation_and_bookkeeping'] = max(0., elapsed - sum(boundaries.values()))
    return dict(total_seconds=elapsed, boundaries_seconds=boundaries,
                snapshot_writes=len(snapshots), snapshot_attempt_rows=sum(s['attempt_rows'] for s in snapshots),
                snapshot_bytes=sum(s['bytes'] for s in snapshots), snapshots=snapshots,
                journal_events=len(data.splitlines()), journal_bytes=len(data),
                actual_evaluations=value['actual_evaluations'], final_measurements_sha256=canonical_sha256(value),
                traced_peak_bytes=peak), value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--baseline-ref', default='55653a53b41603d98b89843057e7f52d3682d950')
    parser.add_argument('--repeats', type=int, default=5)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    old_source = subprocess.check_output(['git', 'show', f'{args.baseline_ref}:src/signed_study/workload.py'])
    legacy = ModuleType('signed_study._profile_legacy_workload')
    exec(compile(old_source, 'baseline_workload.py', 'exec'), legacy.__dict__)
    paths = ['src/signed_study/workload.py', 'src/signed_study/workload_journal.py',
             'src/signed_study/data.py', 'src/signed_study/budget.py',
             'src/signed_study/protocol.py', str(Path(__file__).relative_to(Path.cwd()))]
    hashes = {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths}
    record = dict(schema_version=1, scope='synthetic one-entry evidence volume and bookkeeping; fixed fake fit/score boundaries',
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        baseline_ref=args.baseline_ref, baseline_source_sha256=hashlib.sha256(old_source).hexdigest(),
        source_sha256=hashes, python=sys.version, platform=platform.platform(), numpy=np.__version__,
        warmups=1, repeats=args.repeats, candidate_count=384, positive_count=10, negative_count=5,
        injected_fit_seconds=.001, injected_score_seconds=.002,
        boundary_note='Wall boundaries include cProfile instrumentation; fit/score are explicit fakes, metrics are real. Memory uses a separate tracemalloc run and is not total RSS.',
        observations=[], summaries=[])
    with tempfile.TemporaryDirectory(prefix='workload-profile-', dir='.cache') as directory:
        root = Path(directory)
        for n in (16, 32, 64):
            outputs = {}
            for repeat in range(-1, args.repeats):
                order = [('before_legacy', legacy), ('after_durable', workload)]
                if repeat % 2:
                    order.reverse()
                for method, module in order:
                    destination = root / f'{n}-{repeat}-{method}'
                    destination.mkdir()
                    observed, value = profile_once(module, n, destination)
                    outputs[method] = value
                    if repeat >= 0:
                        record['observations'].append(dict(users=n, repeat=repeat, method=method, **observed))
                if outputs['before_legacy'] != outputs['after_durable']:
                    raise RuntimeError('Fixed-input old/new final measurements differ')
            for method, module in [('before_legacy', legacy), ('after_durable', workload)]:
                destination = root / f'{n}-memory-{method}'
                destination.mkdir()
                memory, _ = profile_once(module, n, destination, memory=True)
                rows = [r for r in record['observations'] if r['users'] == n and r['method'] == method]
                record['summaries'].append(dict(users=n, method=method,
                    median_total_seconds=statistics.median(r['total_seconds'] for r in rows),
                    snapshot_attempt_rows=rows[0]['snapshot_attempt_rows'], snapshot_bytes=rows[0]['snapshot_bytes'],
                    journal_events=rows[0]['journal_events'], journal_bytes=rows[0]['journal_bytes'],
                    traced_peak_bytes=memory['traced_peak_bytes'], exact_final_equivalence=True))
    record['source_unchanged'] = hashes == {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths}
    if not record['source_unchanged']:
        raise RuntimeError('Executed source changed during profiling')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')
    print(f'Saved synthetic workload integrity profile to {args.output}')


if __name__ == '__main__':
    main()
