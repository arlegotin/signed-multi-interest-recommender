"""Independent candidate metrics and conservative paired user inference.

The caller owns protocol validity: numerical interval labels alone do not establish
that a comparator was valid, sufficiently tuned, or frozen before confirmation.
"""

from collections.abc import Mapping, Sequence
from numbers import Integral

import numpy as np


METRIC_KEYS = ('auc', 'ndcg10', 'negative_at_10', 'tie_fraction')


def select_family(rows: list[dict]):
    """Max-anchored tolerance, with deterministic lexicographic simplicity."""
    eligible = [r for r in rows if r.get('status') == 'completed'
                and r.get('complete', True) and np.isfinite(r.get('macro_auc', np.nan))]
    if not eligible:
        raise ValueError('No eligible complete configuration')
    maximum = max(r['macro_auc'] for r in eligible)
    tied = [r['configuration'] for r in eligible if maximum-r['macro_auc'] <= 1e-6]
    return min(tied, key=lambda c: tuple(c.parameters.get(k, 0) for k in ('K','lambda','C','g'))+(c.id,))


def select_comparator(family_choices: dict):
    required = {'Centroid','PositiveMax','RocchioDirection','MeanNegative',
                'AllExemplarMargin','PrototypeRatio','ExemplarRatio','LinearLogistic','RBFSVM'}
    eligible = {f:r for f,r in family_choices.items() if f in required and r.get('status') == 'completed'
                and r.get('complete', True) and np.isfinite(r.get('macro_auc', np.nan))}
    if not eligible:
        raise ValueError('No eligible required comparator')
    family = min(eligible, key=lambda f: (-eligible[f]['macro_auc'], f))
    return eligible[family]['configuration']


def _vector(values, name: str, *, low: float, high: float) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if (result.ndim != 1 or result.size == 0 or not np.isfinite(result).all()
            or np.any(result < low) or np.any(result > high)):
        raise ValueError(f'{name} must be a nonempty finite vector in [{low}, {high}]')
    return result


def _positive_integer(value, name: str, *, maximum: int | None = None) -> int:
    if (isinstance(value, bool) or not isinstance(value, Integral) or value < 1
            or (maximum is not None and value > maximum)):
        raise ValueError(f'{name} must be a positive integer'
                         + (f' at most {maximum}' if maximum is not None else ''))
    return int(value)


def evaluate_user(scores, labels, item_ids) -> dict:
    """Evaluate one eligible user; larger scores predict positive binary labels.

    AUC gives half credit to exact stored-score ties. ``tie_fraction`` is tied
    positive-negative pairs divided by all positive-negative pairs (not all
    candidate pairs). NDCG and negative-at-10 break ties by ascending integer ID.
    Both classes must occur; short candidate lists use k=min(10, n_candidates).
    """
    scores = _vector(scores, 'scores', low=-np.inf, high=np.inf)
    labels = np.asarray(labels)
    ids = np.asarray(item_ids)
    if (labels.shape != scores.shape or not np.isin(labels, [0, 1]).all()
            or ids.shape != scores.shape or ids.dtype.kind not in 'iu'
            or len(np.unique(ids)) != len(ids)):
        raise ValueError('scores, binary labels and distinct integer item IDs must align')
    n_positive = int(np.count_nonzero(labels))
    n_negative = len(labels) - n_positive
    if not n_positive or not n_negative:
        raise ValueError('metric eligibility requires both positive and negative candidates')

    # Sorted exact-score groups give the rank statistic without an O(pos*neg)
    # pair matrix. Each positive beats all earlier negatives and ties its group.
    _, groups, counts = np.unique(scores, return_inverse=True, return_counts=True)
    positive_counts = np.bincount(groups, weights=labels, minlength=len(counts))
    negative_counts = counts - positive_counts
    negatives_before = np.cumsum(negative_counts) - negative_counts
    pairs = n_positive * n_negative
    auc = np.sum(positive_counts * (negatives_before + .5 * negative_counts)) / pairs
    tie_fraction = np.sum(positive_counts * negative_counts) / pairs

    k = min(10, len(scores))
    ranked_labels = labels[np.lexsort((ids, -scores))[:k]]
    discounts = 1 / np.log2(np.arange(2, k + 2, dtype=np.float64))
    ideal = np.sum(discounts[:min(k, n_positive)])
    return {'auc': float(auc), 'ndcg10': float(np.dot(ranked_labels, discounts) / ideal),
            'negative_at_10': float(np.count_nonzero(ranked_labels == 0) / k),
            'tie_fraction': float(tie_fraction)}


def macro_average(user_metrics: Sequence[Mapping[str, float]]) -> dict:
    """Give every supplied user equal weight; reject missing or invalid rows.

    Callers must supply the entire intended cohort. This helper never drops a
    failed row; missing confirmation AUCs instead belong in ``paired_analysis``.
    """
    if not user_metrics:
        raise ValueError('macro average requires at least one user')
    rows = []
    for row in user_metrics:
        if set(row) != set(METRIC_KEYS):
            raise ValueError('each user must have every metric key')
        rows.append(_vector([row[key] for key in METRIC_KEYS], 'metrics', low=0, high=1))
    means = np.mean(rows, axis=0, dtype=np.float64)
    return dict(zip(METRIC_KEYS, map(float, means)))


def difference_bounds(auc_signed: float | None, auc_baseline: float | None) -> tuple[float, float]:
    """Bounds on signed-minus-baseline AUC; only None denotes a missing result."""
    for auc in (auc_signed, auc_baseline):
        if auc is not None and (not np.isscalar(auc) or not np.isfinite(auc) or not 0 <= auc <= 1):
            raise ValueError('known AUC must be finite and in [0, 1]')
    signed_low, signed_high = (0., 1.) if auc_signed is None else (float(auc_signed),) * 2
    base_low, base_high = (0., 1.) if auc_baseline is None else (float(auc_baseline),) * 2
    return signed_low - base_high, signed_high - base_low


def confidence_envelope(lower, upper, *, seed=20261029, resamples=10000, batch_size=200) -> dict:
    """Paired user bootstrap, including every scheduled row in every draw.

    Use the SAME full ordered user cohort, seed, and resample count for every
    method comparison. A fresh RNG stream per call then draws identical user
    indices across comparisons, independent of batch size. Never filter missing
    rows or sort each method's successes separately. Within a call both bound
    arrays use the same indices. No rating/pair bootstrap or item tensor is used.

    Endpoints are NumPy linear 2.5/97.5 percentiles of lower/upper sample means;
    sample_mean_low/high are whole-cohort point bounds, not confidence limits.
    """
    lower = _vector(lower, 'lower', low=-1, high=1)
    upper = _vector(upper, 'upper', low=-1, high=1)
    if lower.shape != upper.shape or np.any(lower > upper):
        raise ValueError('lower and upper must align with lower <= upper')
    resamples = _positive_integer(resamples, 'resamples')
    batch_size = _positive_integer(batch_size, 'batch_size', maximum=200)
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    n_users = len(lower)
    lower_means = np.empty(resamples, dtype=np.float64)
    upper_means = np.empty(resamples, dtype=np.float64)
    for start in range(0, resamples, batch_size):
        stop = min(start + batch_size, resamples)
        indices = rng.integers(0, n_users, size=(stop - start, n_users))
        lower_means[start:stop] = lower[indices].mean(axis=1, dtype=np.float64)
        upper_means[start:stop] = upper[indices].mean(axis=1, dtype=np.float64)
    return {'low': float(np.percentile(lower_means, 2.5, method='linear')),
            'high': float(np.percentile(upper_means, 97.5, method='linear')),
            'sample_mean_low': float(lower.mean(dtype=np.float64)),
            'sample_mean_high': float(upper.mean(dtype=np.float64)),
            'n_users': n_users, 'seed': int(seed), 'resamples': resamples,
            'batch_size': batch_size}


def interpret_interval(low: float, high: float, delta=.01) -> list[str]:
    """Return additive numerical labels; threshold equalities are explicit.

    The first label is practical_advantage (low > delta),
    against_practical_advantage (high < delta), or inconclusive (otherwise).
    Add harm for high < 0, and small_difference for [-delta,delta] containment,
    including endpoints. These labels do not certify study/comparator validity.
    """
    if (not np.isfinite([low, high, delta]).all() or low > high or delta <= 0
            or low < -1 or high > 1):
        raise ValueError('finite ordered AUC-difference interval and positive delta required')
    labels = ['practical_advantage' if low > delta else
              'against_practical_advantage' if high < delta else 'inconclusive']
    if high < 0:
        labels.append('harm')
    if low >= -delta and high <= delta:
        labels.append('small_difference')
    return labels


def paired_standard_error(differences) -> float:
    """Sample sd (ddof=1) / sqrt(users); at least two finite paired users."""
    differences = _vector(differences, 'differences', low=-1, high=1)
    if len(differences) < 2:
        raise ValueError('paired standard error requires at least two users')
    return float(np.std(differences, ddof=1) / np.sqrt(len(differences)))


def planned_half_width(development_differences, planned_confirmation_users: int) -> float:
    """1.96 * sample development sd / sqrt(planned users), not a power guarantee."""
    users = _positive_integer(planned_confirmation_users, 'planned_confirmation_users')
    differences = _vector(development_differences, 'development_differences', low=-1, high=1)
    if len(differences) < 2:
        raise ValueError('planned half-width requires at least two development users')
    return float(1.96 * np.std(differences, ddof=1) / np.sqrt(users))


def paired_analysis(scheduled_user_ids: Sequence[int], signed_auc: Mapping[int, float | None],
                    baseline_auc: Mapping[int, float | None], *, delta=.01, seed=20261029,
                    resamples=10000, batch_size=200) -> dict:
    """Enforce the full scheduled denominator before paired uncertainty.

    Absent mapping IDs and explicit None values both mean missing results.
    Extra IDs, duplicate scheduled IDs, noninteger IDs, and invalid known AUCs
    are rejected. All methods must reuse the same scheduled ID order. A caller
    must never derive the schedule from completed runs.
    """
    ids = list(scheduled_user_ids)
    if (not ids or any(isinstance(u, bool) or not isinstance(u, Integral) for u in ids)
            or len(set(ids)) != len(ids)):
        raise ValueError('scheduled user IDs must be nonempty distinct integers')
    scheduled = set(ids)
    for values in (signed_auc, baseline_auc):
        if (any(isinstance(u, bool) or not isinstance(u, Integral) for u in values)
                or not set(values).issubset(scheduled)):
            raise ValueError('AUC mappings may only contain scheduled integer user IDs')
    bounds, complete = [], []
    for user in ids:
        signed, baseline = signed_auc.get(user), baseline_auc.get(user)
        pair = difference_bounds(signed, baseline)
        bounds.append(pair)
        if signed is not None and baseline is not None:
            complete.append(pair[0])
    lower, upper = np.asarray(bounds, dtype=np.float64).T
    result = confidence_envelope(lower, upper, seed=seed, resamples=resamples, batch_size=batch_size)
    result.update({'user_ids': [int(u) for u in ids],
                   'n_complete': len(complete), 'n_missing': len(ids) - len(complete),
                   'lower_bounds': lower.tolist(), 'upper_bounds': upper.tolist(),
                   'labels': interpret_interval(result['low'], result['high'], delta),
                   'delta': float(delta)})
    # Complete-case estimates are descriptive; the full-cohort envelope alone
    # drives primary decisions. A one-user sample has no estimable sample SE.
    result['complete_case'] = {
        'label': 'complete_case_diagnostic', 'n_users': len(complete),
        'mean_difference': float(np.mean(complete)) if complete else None,
        'standard_error': paired_standard_error(complete) if len(complete) >= 2 else None,
    }
    result['standard_error'] = (result['complete_case']['standard_error']
                                if len(complete) == len(ids) else None)
    return result
