"""Deterministic spherical prototypes and exact signed/nearest-class geometry.

All computation is CPU float64. These functions consume support vectors only;
neither prototype construction nor scoring accepts evaluation labels.
"""

from dataclasses import dataclass

import numpy as np
from scipy.spatial.distance import cdist


@dataclass(frozen=True)
class PrototypeResult:
    vectors: np.ndarray
    objective: float
    iterations: int
    flags: dict


def _unit_rows(value, name):
    rows = np.asarray(value, dtype=np.float64)
    if rows.ndim != 2 or rows.shape[1] == 0:
        raise ValueError(f'{name} must be a matrix of unit rows')
    if not np.isfinite(rows).all():
        raise ValueError(f'{name} must contain finite unit rows')
    if np.any(np.abs(np.linalg.norm(rows, axis=1) - 1) > 1e-12):
        raise ValueError(f'{name} row norms must differ from one by at most 1e-12')
    return rows


def _positive_integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value <= 0:
        raise ValueError(f'{name} must be a positive integer')


def spherical_prototypes(X: np.ndarray, item_ids: np.ndarray, K: int, *, max_updates=50) -> PrototypeResult:
    """Bounded spherical Lloyd updates, initialized from exact distinct rows.

    IDs fix observation/tie order even if the input is shuffled. Flags preserve
    any empty/zero-sum event per prototype across all updates. The objective
    trace includes initialization, then one entry per completed update.
    """
    X = _unit_rows(X, 'X')
    if not len(X):
        raise ValueError('ineligible history: positive support is empty')
    _positive_integer(K, 'K')
    _positive_integer(max_updates, 'max_updates')
    item_ids = np.asarray(item_ids)
    if (item_ids.ndim != 1 or len(item_ids) != len(X)
            or not np.issubdtype(item_ids.dtype, np.integer)
            or len(np.unique(item_ids)) != len(item_ids)):
        raise ValueError('item_ids must be distinct integers, one per observation')
    X = X[np.argsort(item_ids, kind='stable')]
    # np.unique's lexical ordering must not override the item-ID tie rule.
    _, first_indices = np.unique(X, axis=0, return_index=True)
    distinct = X[np.sort(first_indices)]
    effective = min(int(K), len(distinct))
    mean = X.mean(axis=0)  # Duplicate observations retain their full weight.
    mean_fallback = bool(np.linalg.norm(mean) <= 1e-12)
    first = 0 if mean_fallback else int(np.argmax(distinct @ mean))
    selected = [first]
    while len(selected) < effective:
        closeness = np.max(distinct @ distinct[selected].T, axis=1)
        closeness[selected] = np.inf
        selected.append(int(np.argmin(closeness)))
    centers = distinct[selected].copy()
    similarities = X @ centers.T
    assignments = np.argmax(similarities, axis=1)
    objective = float(np.max(similarities, axis=1).sum())
    trace = [objective]
    empty_clusters = [False] * effective
    zero_sum_clusters = [False] * effective
    converged = False
    for iteration in range(1, max_updates + 1):
        for k in range(effective):
            members = X[assignments == k]
            if not len(members):
                empty_clusters[k] = True
                continue
            total = members.sum(axis=0)
            norm = np.linalg.norm(total)
            if norm <= 1e-12:
                zero_sum_clusters[k] = True
                continue
            centers[k] = total / norm
        similarities = X @ centers.T
        new_assignments = np.argmax(similarities, axis=1)
        updated_objective = float(np.max(similarities, axis=1).sum())
        improvement = updated_objective - objective
        if improvement < -1e-9 * len(X):
            raise ArithmeticError('spherical objective decreased beyond 1e-9 per observation')
        trace.append(updated_objective)
        objective = updated_objective
        converged = bool(np.array_equal(assignments, new_assignments) or improvement <= 1e-10 * len(X))
        if converged:
            break
        assignments = new_assignments
    # This checks the returned vectors as well as all normalized updates.
    _unit_rows(centers, 'prototypes')
    return PrototypeResult(centers, objective, iteration, {
        'requested_K': int(K), 'effective_K': effective,
        'empty_clusters': empty_clusters, 'zero_sum_clusters': zero_sum_clusters,
        'mean_fallback': mean_fallback, 'cap_hit': not converged,
        'objective_trace': trace,
    })


def _score_inputs(X, P, N):
    X, P, N = (_unit_rows(rows, name) for rows, name in [(X, 'X'), (P, 'P'), (N, 'N')])
    if X.shape[1] != P.shape[1] or X.shape[1] != N.shape[1]:
        raise ValueError('X, P, and N must have the same vector dimension')
    if not len(P):
        raise ValueError('ineligible history: positive support is empty')
    return X, P, N


def signed_scores(X, P, N, penalty: float, *, chunk_size=512) -> np.ndarray:
    """Positive maximum minus penalty times raw negative maximum; no clipping."""
    X, P, N = _score_inputs(X, P, N)
    if not np.isscalar(penalty) or not np.isfinite(penalty) or penalty < 0:
        raise ValueError('penalty must be finite and nonnegative')
    _positive_integer(chunk_size, 'chunk_size')
    scores = np.empty(len(X), dtype=np.float64)
    for start in range(0, len(X), chunk_size):
        chunk = X[start:start + chunk_size]
        positive = np.max(chunk @ P.T, axis=1)
        if len(N) and penalty != 0:
            positive -= penalty * np.max(chunk @ N.T, axis=1)
        scores[start:start + chunk_size] = positive
    if not np.isfinite(scores).all():
        raise ValueError('nonfinite signed scores')
    return scores


def ratio_scores(X, P, N) -> np.ndarray:
    """Nearest-class Euclidean ratio d_N/(d_P+d_N), including exact zeros.

    Direct distances avoid cancellation near identical rows. Bounded cdist
    blocks use no candidate-by-support-by-dimension difference tensor and need
    no cosine clipping or denominator epsilon.
    """
    X, P, N = _score_inputs(X, P, N)
    if not len(N):
        raise ValueError('ineligible history: negative support is empty for distance ratio')
    scores = np.empty(len(X), dtype=np.float64)
    for start in range(0, len(X), 512):
        chunk = X[start:start + 512]
        dp = np.min(cdist(chunk, P, metric='euclidean'), axis=1)
        dn = np.min(cdist(chunk, N, metric='euclidean'), axis=1)
        denominator = dp + dn
        values = np.full(len(chunk), .5, dtype=np.float64)
        np.divide(dn, denominator, out=values, where=denominator != 0)
        scores[start:start + 512] = values
    return scores
