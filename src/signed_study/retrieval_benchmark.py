"""Reconstruct benchmark statistics from public ordinal/repeat observations."""
from collections import defaultdict
import math
from statistics import mean, median


def summarize(observations, *, build_seconds, expected_repeats=None, expected_ordinals=None,
              expected_methods=None, expected_workloads=None):
    """Pair each user's repeat medians; retain failures and missing denominators."""
    if not math.isfinite(build_seconds) or build_seconds < 0:
        raise ValueError('Invalid build duration')
    grouped = defaultdict(list)
    seen = set()
    for row in observations:
        key = tuple(row[k] for k in ('workload','ordinal','repeat','method'))
        if key in seen:
            raise ValueError('Duplicate benchmark observation')
        seen.add(key)
        if row['status'] not in ('ok','failed'):
            raise ValueError('Unknown observation status')
        if row['status'] == 'ok' and (not math.isfinite(row['seconds']) or row['seconds'] <= 0):
            raise ValueError('Timing must be finite and positive')
        grouped[row['workload']].append(row)
    if expected_workloads is not None:
        if not grouped.keys() <= set(expected_workloads):
            raise ValueError('Workload outside declared schedule')
        for workload in expected_workloads:
            grouped.setdefault(workload,[])
    result = {}
    for workload, rows in grouped.items():
        present = set(r['ordinal'] for r in rows)
        ordinals = present if expected_ordinals is None else set(expected_ordinals)
        repeats = (set(r['repeat'] for r in rows) if expected_repeats is None else set(range(expected_repeats)))
        if not {r['ordinal'] for r in rows} <= ordinals or not {r['repeat'] for r in rows} <= repeats:
            raise ValueError('Observation outside declared schedule')
        methods = sorted(set(r['method'] for r in rows) if expected_methods is None else expected_methods)
        if not {r['method'] for r in rows} <= set(methods):
            raise ValueError('Method outside declared schedule')
        medians = {}
        descriptions = {}
        for method in methods:
            selected = [r for r in rows if r['method'] == method]
            medians[method] = {}
            for ordinal in ordinals:
                attempts = [r for r in selected if r['ordinal'] == ordinal]
                # Any failed repeat makes this user's method incomplete.
                if ({r['repeat'] for r in attempts} == repeats and attempts
                        and all(r['status'] == 'ok' for r in attempts)):
                    medians[method][ordinal] = median(r['seconds'] for r in attempts)
            descriptions[method] = dict(median_seconds=median(medians[method].values())
                if medians[method] else None, complete_users=len(medians[method]),
                failed_observations=sum(r['status'] != 'ok' for r in selected),
                observations=len(selected))
        paired = {}
        for baseline in ('reference','dense','adapter_reference'):
            for target in ('gram_fast','gram_strict','adapter_cached'):
                if baseline not in medians or target not in medians:
                    continue
                users = sorted(medians[baseline].keys() & medians[target].keys())
                savings = mean(medians[baseline][u]-medians[target][u] for u in users) if users else None
                paired[f'{baseline}/{target}'] = dict(paired_users=len(users),
                    missing_users=len(ordinals)-len(users),
                    median_ratio=median(medians[baseline][u]/medians[target][u] for u in users) if users else None,
                    mean_savings_seconds=savings,
                    build_amortization_queries=build_seconds/savings if savings is not None and savings > 0 else None)
        result[workload] = dict(observed_ordinals=len(present), scheduled_ordinals=len(ordinals),
                               methods=descriptions, paired=paired)
    return result
