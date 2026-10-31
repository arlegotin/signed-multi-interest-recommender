#!/usr/bin/env python3
"""Offline synthetic before/after lookup and immutable-record validation profile.

No dataset/model files or outcomes are read. Query timing includes row validation
and gathering; model-workload timing includes fit_user + full score_user. Episode
hashing is a separate serialization microbenchmark, not a pipeline speed claim.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
import tracemalloc

import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits

from signed_study.data import (Episode, EpisodeStore, History, _remember_verified_records,
                               episode_store_digest)
from signed_study.embed import EmbeddingCache
from signed_study.models import fit_user, score_user


class LegacyLookup:
    """Baseline lookup algorithm reconstructed from commit 55653a5, not an asset."""

    def __init__(self, cache):
        self.item_ids, self.vectors = cache.item_ids, cache.vectors

    def rows(self, item_ids):
        requested = list(item_ids)
        if any(type(item) is not int and not isinstance(item, np.integer) for item in requested):
            raise ValueError('Embedding row lookup requires integer IDs')
        index = {int(item): row for row, item in enumerate(self.item_ids)}
        return self.vectors[[index[int(item)] for item in requested]]


def sample(function, repeats, *, batch=1):
    function()
    elapsed = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        for _ in range(batch):
            function()
        elapsed.append((time.perf_counter_ns() - start) / 1e9 / batch)
    # Memory is measured in a separate call so tracemalloc does not skew timing.
    tracemalloc.start()
    function()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return dict(seconds=elapsed, median_seconds=statistics.median(elapsed),
                traced_peak_workspace_bytes=peak, batch=batch)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=5)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    rng = np.random.default_rng(1239)
    raw = rng.normal(size=(3883, 384)).astype(np.float32)
    start = time.perf_counter_ns()
    cache = EmbeddingCache(np.arange(1, 3884), raw, {})
    prepare = (time.perf_counter_ns() - start) / 1e9
    old = LegacyLookup(cache)
    ids = list(range(10, 26))
    np.testing.assert_array_equal(old.rows(ids), cache.rows(ids))
    history = History(1, 1000, 'development', tuple(range(1, 129)), tuple(range(129, 193)))
    candidates = tuple(range(193, 3884))

    def workload(lookup):
        model = fit_user('SignedMax', history, lookup, {'K': 2, 'lambda': 2})
        return score_user(model, candidates, lookup)

    paths = ['src/signed_study/embed.py', 'src/signed_study/data.py',
             'src/signed_study/models.py', str(Path(__file__).relative_to(Path.cwd()))]
    hashes = {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths}
    record = dict(schema_version=1, scope='synthetic lookup, fit+score and record hashing; no study outcomes',
                  source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                  source_sha256=hashes, python=sys.version, platform=platform.platform(), numpy=np.__version__,
                  catalog=dict(rows=3883, dimensions=384, dtype='normalized float64 from synthetic float32', seed=1239),
                  preparation_seconds=prepare, warmups=1, repeats=args.repeats,
                  lookup={}, fit_and_score={}, hashing=[])
    with threadpool_limits(limits=1):
        record['threadpools'] = threadpool_info()
        np.testing.assert_array_equal(workload(old), workload(cache))
        # Interleave the two independent boundaries by method; raw observations
        # describe this mechanism profile, not the rotated retrieval benchmark.
        for name, lookup in [('before_legacy', old), ('after_indexed', cache)]:
            record['lookup'][name] = sample(lambda: lookup.rows(ids), args.repeats, batch=100)
            record['fit_and_score'][name] = sample(lambda: workload(lookup), args.repeats)
        for n in (128, 512, 2194):
            rows = tuple(Episode(History(i, 1000, 'development' if i < n // 5 else 'confirmation',
                                        tuple(range(128)), tuple(range(128, 192))),
                                 tuple(range(192, 292))) for i in range(n))
            store = EpisodeStore(rows[:n // 5], rows[n // 5:], {}, {})
            digest = episode_store_digest(store)
            baseline = sample(lambda: episode_store_digest(store), args.repeats)
            start = time.perf_counter_ns()
            # Here the generator and full digest establish the synthetic frozen
            # records; production attaches this only after producer/loader checks.
            _remember_verified_records(store, digest)
            validation = (time.perf_counter_ns() - start) / 1e9
            optimized = sample(lambda: episode_store_digest(store), args.repeats, batch=100)
            assert episode_store_digest(store) == digest
            record['hashing'].append(dict(episodes=n, ids_per_episode=292, sha256=digest,
                before=baseline, after=optimized, one_time_structural_validation_seconds=validation))
    record['lookup_operations'] = dict(requests=20, catalog_entries_before=20 * 3883,
        catalog_entries_after_one_time=3883, catalog_entries_after_per_request=0, requested_ids_per_call=16)
    record['memory'] = dict(vectors_bytes=cache.vectors.nbytes, ids_bytes=cache.item_ids.nbytes,
        index_python_shallow_bytes=sys.getsizeof(dict(cache._id_index)),
        index_python_entries_bytes=sum(sys.getsizeof(k) + sys.getsizeof(v) for k, v in cache._id_index.items()),
        note='tracemalloc peaks are traced allocations in a separate call, not total RSS')
    record['source_unchanged'] = hashes == {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths}
    if not record['source_unchanged']:
        raise RuntimeError('Executed source changed during profiling')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')
    print(f'Saved synthetic identity profile to {args.output}')


if __name__ == '__main__':
    main()
