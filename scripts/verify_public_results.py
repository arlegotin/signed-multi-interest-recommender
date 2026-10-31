#!/usr/bin/env python3
"""Check the released aggregates using only Python's standard library.

This checks published integrity and consistency, not an independent replication.
No scientific package, model, raw data, private results, or network is accessed.
"""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys


METHODS = frozenset((
    'Centroid', 'PositiveMax', 'RocchioDirection', 'MeanNegative', 'SignedMax',
    'AllExemplarMargin', 'PrototypeRatio', 'ExemplarRatio', 'LinearLogistic',
    'RBFSVM', 'matched_MeanNegative', 'matched_PositiveMax', 'matched_K1_SignedMax',
))
METRICS = 'docs/native-confirmation-metrics.csv'
COMPARISONS = 'docs/native-confirmation-comparisons.csv'
TIMING = 'docs/native-timing-summary.csv'
ARTIFACTS = frozenset((METRICS, COMPARISONS, TIMING,
                       'docs/figures/native-paired-intervals.png'))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def number(value, label, low=None, high=None):
    result = float(value)
    require(math.isfinite(result), f'{label}: expected a finite number')
    require(low is None or result >= low, f'{label}: must be >= {low}')
    require(high is None or result <= high, f'{label}: must be <= {high}')
    return result


def same(actual, expected, label):
    actual = number(actual, label)
    expected = number(expected, label)
    require(math.isclose(actual, expected, rel_tol=0, abs_tol=1e-14),
            f'{label}: {actual!r} does not match {expected!r}')


def exact(record, field, expected, label):
    require(record[field] == expected, f'{label}.{field}: expected {expected!r}')


def read_rows(root, relative, identity, expected, fields):
    with (root / relative).open(encoding='utf-8', newline='') as handle:
        reader = csv.DictReader(handle)
        require(reader.fieldnames == fields, f'{relative}: unexpected CSV columns')
        rows = list(reader)
    identities = [row[identity] for row in rows]
    require(len(identities) == len(set(identities)),
            f'{relative}: duplicate {identity} identity')
    require(set(identities) == expected,
            f'{relative}: expected {len(expected)} frozen identities; '
            f'missing={sorted(expected-set(identities))}, '
            f'extra={sorted(set(identities)-expected)}')
    return {row[identity]: row for row in rows}


def verify(root):
    """Raise an actionable ValueError on inconsistency; return public aggregates."""
    root = Path(root).resolve()
    manifest = json.loads((root / 'docs/public-results.json').read_text(encoding='utf-8'))
    exact(manifest, 'schema_version', 1, 'manifest')
    require(set(manifest['artifacts']) == ARTIFACTS,
            'manifest.artifacts: expected the three aggregate CSVs and original paired figure')
    for relative, metadata in manifest['artifacts'].items():
        path = root / relative
        require(path.resolve().is_relative_to(root), f'{relative}: escapes repository root')
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        require(digest == metadata['sha256'],
                f'{relative}: SHA256 mismatch; restore the released file or review its provenance')

    metrics = read_rows(root, METRICS, 'method', METHODS, [
        'method', 'n_completed', 'n_scheduled', 'auc', 'ndcg10', 'negative_at_10', 'tie_fraction'])
    require(set(manifest['metric_means']) == METHODS, 'manifest.metric_means: wrong method identities')
    for method, row in metrics.items():
        for field in ('n_completed', 'n_scheduled'):
            require(row[field] == '1776', f'{METRICS}: {method}.{field}: expected 1776')
        saved = manifest['metric_means'][method]
        exact(saved, 'n_complete', 1776, method)
        exact(saved, 'n_scheduled', 1776, method)
        for field in ('auc', 'ndcg10', 'negative_at_10', 'tie_fraction'):
            number(row[field], f'{method}.{field}', 0, 1)
            same(row[field], saved[field], f'{method}.{field} CSV/manifest')

    comparisons = read_rows(root, COMPARISONS, 'comparator', METHODS - {'SignedMax'}, [
        'comparator', 'sample_mean_low', 'sample_mean_high', 'low', 'high', 'n_users',
        'n_complete', 'n_missing', 'standard_error', 'complete_case'])
    signed_auc = float(metrics['SignedMax']['auc'])
    for comparator, row in comparisons.items():
        label = f'{COMPARISONS}: {comparator}'
        for field in ('n_users', 'n_complete'):
            require(row[field] == '1776', f'{label}.{field}: expected 1776')
        require(row['n_missing'] == '0', f'{label}.n_missing: expected zero')
        mean = signed_auc - float(metrics[comparator]['auc'])
        for field in ('sample_mean_low', 'sample_mean_high'):
            same(row[field], mean, f'{label}.{field}: paired mean vs AUC difference')
        low, high = (number(row[field], f'{label}.{field}', -1, 1) for field in ('low', 'high'))
        require(low <= mean <= high, f'{label}: interval must contain the paired mean')
        number(row['standard_error'], f'{label}.standard_error', 0)
        diagnostic = json.loads(row['complete_case'])
        exact(diagnostic, 'label', 'complete_case_diagnostic', label)
        exact(diagnostic, 'n_users', 1776, label)
        same(diagnostic['mean_difference'], mean, f'{label}: diagnostic mean')
        same(diagnostic['standard_error'], row['standard_error'], f'{label}: diagnostic standard error')

    primary = manifest['primary']
    for field, expected in {
        'method': 'SignedMax', 'comparator': 'LinearLogistic', 'metric': 'macro user AUC',
        'difference_definition': 'SignedMax minus comparator', 'outcome': 'informative negative',
        'method_parameters': {'K': 2, 'lambda': 2}, 'comparator_parameters': {'C': 1},
        'n_users': 1776, 'n_complete': 1776, 'n_missing': 0,
        'confidence_level': .95, 'interval_method': 'paired user bootstrap',
        'bootstrap_resamples': 10000, 'bootstrap_seed': 20261029, 'bootstrap_batch_size': 200,
        'practical_threshold': .01,
    }.items():
        exact(primary, field, expected, 'primary')
    row = comparisons['LinearLogistic']
    for field, value in {
        'method_auc': metrics['SignedMax']['auc'],
        'comparator_auc': metrics['LinearLogistic']['auc'],
        'mean_difference': row['sample_mean_low'],
        'interval_low': row['low'], 'interval_high': row['high'],
    }.items():
        same(primary[field], value, f'primary.{field}')
    require(primary['interval_high'] < 0 < primary['practical_threshold'],
            'primary: informative negative requires an interval wholly below zero')
    require(primary['method_auc'] < primary['comparator_auc'],
            'primary: SignedMax must have lower AUC than frozen comparator')

    exact(manifest, 'cohorts', {'development_users': 418, 'confirmation_users': 1776}, 'manifest')
    exact(manifest, 'counts', {
        'development_configurations': 83, 'confirmation_slots': 13,
        'confirmation_evaluations': 23088, 'confirmation_missing': 0,
        'confirmation_failed': 0, 'all_study_evaluations': 62822,
    }, 'manifest')
    provenance = manifest['provenance']
    for field, value in {
        'frozen_scientific_commit': '5a0595d29869678c9b424ae0a9df42bfcee7f8f9',
        'protocol_lock_sha256': '7e13cb9bfbb6ddbaa27aa9d39c6291304f3120c74a107ce5eccdf1cb75902eca',
        'native_summary_output_sha256': 'ee34a77e466e2151f1a1686a648516ad6c1cd1143b75a0cd828da6bf73a1b01d',
        'encoder': 'sentence-transformers/all-MiniLM-L6-v2',
        'encoder_revision': '1110a243fdf4706b3f48f1d95db1a4f5529b4d41',
        'dataset': 'official MovieLens 1M',
        'official_zip_sha256': 'a6898adb50b9ca05aa231689da44c217cb524e7ebd39d264c56e2832f2c54e20',
    }.items():
        exact(provenance, field, value, 'provenance')

    timing = read_rows(root, TIMING, 'method', METHODS, [
        'method', 'n_completed', 'n_failed', 'fit_seconds', 'score_seconds',
        'score_rank_seconds', 'total_evaluation_seconds'])
    conditions = manifest['timing_conditions']
    for field, value in {
        'catalog_items': 3883, 'development_users': 32,
        'warmup_repetitions_per_user_slot': 1, 'timed_repetitions_per_user_slot': 5,
        'timed_evaluations_per_slot': 160, 'timed_evaluations_all_slots': 2080,
        'evaluations_including_warmup': 2496, 'units': 'seconds',
        'aggregation': 'median across users of each user’s repeat median',
        'score_rank_includes_scoring': True, 'configured_workers': 1,
        'configured_blas_threads': 1, 'detected_libomp_threads': 1,
        'numpy_blas': 'Accelerate', 'actual_accelerate_thread_count': 'unverified',
        'thread_control_environment_variables': 'unset',
    }.items():
        exact(conditions, field, value, 'timing_conditions')
    for method, row in timing.items():
        require(row['n_completed'] == '160' and row['n_failed'] == '0',
                f'{TIMING}: {method}: expected 160 completed timed repeats and zero failures')
        for field in ('fit_seconds', 'score_seconds', 'score_rank_seconds', 'total_evaluation_seconds'):
            require(number(row[field], f'{method}.{field}', 0) > 0,
                    f'{method}.{field}: expected a positive duration in seconds')
        require(float(row['score_rank_seconds']) >= float(row['score_seconds']),
                f'{method}: score+rank must include scoring')
    return manifest, metrics, comparisons, timing


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1],
                        help='repository root (defaults to the script’s parent repository)')
    args = parser.parse_args()
    try:
        verify(args.root)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f'Public results verification FAILED: {error}', file=sys.stderr)
        return 1
    print('Public results verified: 13 methods, 12 paired comparisons, 1776 complete users; '
          '23088 confirmation evaluations, zero missing/failed; informative negative. '
          'Aggregate hashes and consistency checked; no study rerun.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
