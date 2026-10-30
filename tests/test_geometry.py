"""Tiny independent distance/loop oracles for the frozen geometric mechanism."""

import numpy as np
import pytest
from scipy.optimize import linprog

from signed_study.models import ratio_scores, signed_scores, spherical_prototypes


def unit(rows):
    rows = np.asarray(rows, dtype=np.float64)
    return rows / np.linalg.norm(rows, axis=1, keepdims=True)


def distance_oracle(X, P, N, penalty):
    """Equation (3), using direct Euclidean distances and scalar loops."""
    scores = []
    for x in X:
        dp = min(np.linalg.norm(x - p) for p in P)
        dn = min(np.linalg.norm(x - n) for n in N)
        scores.append(1 - penalty - dp**2 / 2 + penalty * dn**2 / 2)
    return np.array(scores)


def ratio_oracle(X, P, N):
    scores = []
    for x in X:
        dp = min(np.linalg.norm(x - p) for p in P)
        dn = min(np.linalg.norm(x - n) for n in N)
        scores.append(.5 if dp == dn == 0 else dn / (dp + dn))
    return np.array(scores)


def local_control():
    P = np.array([[0., 0., 1.], [0., 0., -1.]])
    N = np.array([[1., 0., 0.], [-1., 0., 0.]])
    X = np.array([[0., .6, .8], [0., -.6, -.8], [.6, 0., .8], [-.6, 0., -.8]])
    return X, P, N


def test_exact_local_negative_control():
    X, P, N = local_control()
    np.testing.assert_allclose(signed_scores(X, P, N, 1), [.8, .8, .2, .2], atol=1e-15)
    np.testing.assert_allclose(signed_scores(X, P, N, 0), [.8] * 4, atol=1e-15)
    np.testing.assert_array_equal(N.mean(axis=0), np.zeros(3))
    scores = ratio_scores(X, P, N)
    np.testing.assert_allclose(scores, ratio_oracle(X, P, N), atol=1e-15)
    assert min(scores[:2]) > max(scores[2:])


@pytest.mark.parametrize('penalty', [0, .25, .5, .7, 1, 2, 4, 8, 16])
def test_distance_max_min_rotation_and_chunk_oracles(penalty, record_property):
    rng = np.random.default_rng(413)
    P, N, X = [unit(rng.standard_normal(shape)) for shape in [(5, 16), (7, 16), (23, 16)]]
    # Include an exact tie and a near tie to exercise the gap-qualified criterion.
    X[-1] = X[0]
    X[-2] = unit([X[0] + 1e-13 * X[1]])[0]
    expected = distance_oracle(X, P, N, penalty)
    max_min = [max(min(float(x @ (p - penalty * n)) for n in N) for p in P) for x in X]
    np.testing.assert_allclose(max_min, expected, atol=1e-10, rtol=0)
    rotation, _ = np.linalg.qr(rng.standard_normal((16, 16)))
    np.testing.assert_allclose(signed_scores(X @ rotation, P @ rotation, N @ rotation, penalty),
                               expected, atol=1e-10, rtol=0)
    eta = 1e-10 * (1 + penalty)
    permutation = rng.permutation(len(X))
    near_tie_reorderings = 0
    for chunk in [1, 7, 512]:
        actual = signed_scores(X[permutation], P, N, penalty, chunk_size=chunk)[np.argsort(permutation)]
        np.testing.assert_allclose(actual, expected, atol=eta, rtol=0)
        for i in range(len(X)):
            for j in range(i):
                if abs(expected[i] - expected[j]) > 2 * eta:
                    assert (actual[i] > actual[j]) == (expected[i] > expected[j])
                elif np.sign(actual[i] - actual[j]) != np.sign(expected[i] - expected[j]):
                    near_tie_reorderings += 1
    record_property('near_tie_reorderings', near_tie_reorderings)
    np.testing.assert_allclose(ratio_scores(X, P, N), ratio_oracle(X, P, N), atol=1e-10, rtol=0)


def test_zero_lambda_single_pair_and_raw_negative_cosines():
    X, P, N = local_control()
    np.testing.assert_array_equal(signed_scores(X, P, N, 0), np.array([.8] * 4))
    for penalty in [0, 1, 16]:
        np.testing.assert_allclose(signed_scores(X, P[:1], N[:1], penalty),
                                   [x @ (P[0] - penalty * N[0]) for x in X], atol=1e-14)
    X = np.array([[1., 0.]])
    np.testing.assert_array_equal(signed_scores(X, X, -X, 2), [3.])


def test_negative_duplicates_and_identical_conflicting_vectors():
    X, P, N = local_control()
    np.testing.assert_array_equal(signed_scores(X, P, N, 4),
                                   signed_scores(X, P, np.repeat(N, [5, 1], axis=0), 4))
    same = np.repeat(P[:1], 3, axis=0)
    np.testing.assert_array_equal(signed_scores(same, P[:1], P[:1], 1), [0., 0., 0.])
    np.testing.assert_array_equal(ratio_scores(same, P[:1], P[:1]), [.5, .5, .5])


def test_empty_history_and_candidates():
    X, P, N = local_control()
    empty = np.empty((0, 3), dtype=np.float64)
    np.testing.assert_array_equal(signed_scores(X, P, empty, 16), [.8] * 4)
    for score in [lambda: signed_scores(X, empty, N, 1), lambda: ratio_scores(X, empty, N),
                  lambda: ratio_scores(X, P, empty),
                  lambda: spherical_prototypes(empty, np.array([], dtype=int), 2)]:
        with pytest.raises(ValueError, match='ineligible history'):
            score()
    assert signed_scores(empty, P, N, 1).shape == (0,)
    assert ratio_scores(empty, P, N).shape == (0,)


def test_ratio_exact_zero_cases_and_euclidean_not_squared_ratio():
    P = np.array([[1., 0.]])
    N = np.array([[0., 1.]])
    X = np.array([[1., 0.], [0., 1.], [-1., 0.]])
    np.testing.assert_allclose(ratio_scores(X, P, N), [1., 0., np.sqrt(2) / (2 + np.sqrt(2))],
                               atol=1e-15, rtol=0)
    np.testing.assert_array_equal(ratio_scores(P, P, P), [.5])


def test_ratio_distance_clip_tolerance_and_no_arbitrary_epsilon():
    # Still within the required norm tolerance; distance-only clipping is legal.
    almost = np.array([[1. + 1e-14, 0.]])
    other = np.array([[0., 1.]])
    np.testing.assert_array_equal(ratio_scores(almost, almost, other), [1.])
    np.testing.assert_array_equal(ratio_scores(almost, almost, almost), [.5])
    # A denominator epsilon or cancellation in 2-2*cosine destroys this ratio.
    X = np.array([[1., 0.]])
    np.testing.assert_allclose(ratio_scores(X, unit([[1., 1e-14]]), unit([[1., 3e-14]])),
                               [.75], atol=1e-15, rtol=0)


def test_ratio_multiple_bounded_chunks_keep_last_candidates():
    X, P, N = local_control()
    candidates = np.tile(X, (129, 1))[:514]
    np.testing.assert_allclose(ratio_scores(candidates, P, N), ratio_oracle(candidates, P, N),
                               atol=1e-15, rtol=0)


def test_positive_mean_is_duplicate_weighted():
    X = np.array([[1., 0.], [1., 0.], [0., 1.]])
    result = spherical_prototypes(X, np.array([1, 2, 3]), 1)
    np.testing.assert_allclose(result.vectors, [[2 / np.sqrt(5), 1 / np.sqrt(5)]], atol=1e-15)
    assert result.objective == pytest.approx(np.sqrt(5), abs=1e-14)
    assert result.iterations == 1
    assert not result.flags['cap_hit']


def test_distinct_initialization_exact_ties_use_item_ids():
    X = np.array([[1., 0.], [0., 1.], [-1., 0.], [0., -1.], [1., 0.]])
    # Nonzero mean selects +e1, and the farthest distinct vector is -e1.
    result = spherical_prototypes(X, np.array([1, 2, 3, 4, 5]), 2)
    np.testing.assert_array_equal(result.vectors, [[1., 0.], [-1., 0.]])
    assert result.objective == 3.
    # Exact assignment ties send both orthogonal observations to prototype 0.
    balanced = spherical_prototypes(X[:4], np.array([1, 2, 3, 4]), 2)
    np.testing.assert_array_equal(balanced.vectors, [[1., 0.], [-1., 0.]])
    assert balanced.flags['mean_fallback']
    # Sorting by IDs makes shuffled observations deterministic too.
    order = np.array([3, 1, 4, 2, 0])
    shuffled = spherical_prototypes(X[order], np.array([1, 2, 3, 4, 5])[order], 2)
    np.testing.assert_array_equal(shuffled.vectors, result.vectors)
    assert shuffled.flags == result.flags


def test_observed_row_nearest_mean_and_farthest_tie_order():
    X = np.array([[0., 1.], [1., 0.], [0., -1.]])
    # All three selected centers remain observed rows, exposing index ordering.
    result = spherical_prototypes(X, np.array([1, 2, 3]), 3)
    np.testing.assert_array_equal(result.vectors, [[1., 0.], [0., 1.], [0., -1.]])


def test_assignment_tie_changes_duplicate_weighted_center():
    X = np.array([[1., 0.]] * 3 + [[-1., 0.]] * 2 + [[0., 1.]])
    result = spherical_prototypes(X, np.arange(6), 2)
    # e2 ties between +e1/-e1 at initialization, and must join index 0.
    np.testing.assert_allclose(result.vectors, [[3 / np.sqrt(10), 1 / np.sqrt(10)], [-1., 0.]],
                               atol=1e-15, rtol=0)
    assert result.objective == pytest.approx(np.sqrt(10) + 2, abs=1e-14)


def test_identical_rows_reduce_effective_count_and_zero_sum_retains_center():
    X = np.repeat(np.array([[0., 1.]]), 4, axis=0)
    result = spherical_prototypes(X, np.arange(4), 4)
    np.testing.assert_array_equal(result.vectors, [[0., 1.]])
    assert result.flags['requested_K'] == 4 and result.flags['effective_K'] == 1
    assert result.objective == 4
    assert result.flags['empty_clusters'] == [False]
    assert result.flags['zero_sum_clusters'] == [False]
    opposite = np.array([[0., -1.], [0., 1.]])
    result = spherical_prototypes(opposite, np.array([1, 2]), 1)
    np.testing.assert_array_equal(result.vectors, opposite[:1])
    assert result.flags['mean_fallback']
    assert result.flags['zero_sum_clusters'] == [True]
    assert result.objective == 0 and result.iterations == 1


def test_close_but_distinct_rows_are_not_merged():
    X = unit([[1, 0], [1, 1e-8]])
    result = spherical_prototypes(X, np.array([1, 2]), 4)
    assert result.flags['effective_K'] == 2
    assert result.vectors.shape == (2, 2)
    # Distinct stored rows have an exact float64 dot-product tie here. Both
    # assign to index 0, leaving index 1 empty even though it was initialized.
    assert result.flags['empty_clusters'] == [False, True]
    np.testing.assert_array_equal(result.vectors[1], X[1])


def test_prototype_objective_unit_norms_determinism_and_cap():
    rng = np.random.default_rng(20261029)
    X = unit(rng.standard_normal((19, 5)))
    ids = np.arange(19)
    limited = spherical_prototypes(X, ids, 4, max_updates=1)
    assert limited.iterations == 1 and limited.flags['cap_hit']
    result = spherical_prototypes(X, ids, 4)
    repeat = spherical_prototypes(X, ids, 4)
    np.testing.assert_array_equal(result.vectors, repeat.vectors)
    assert result.flags == repeat.flags and result.objective == repeat.objective
    assert 1 <= result.iterations <= 50
    assert not result.flags['cap_hit']
    assert np.isfinite(result.objective)
    np.testing.assert_allclose(np.linalg.norm(result.vectors, axis=1), 1, atol=1e-12, rtol=0)
    # The direct-distance objective does not reuse the scoring helper/reduction.
    objective = sum(1 - min(np.linalg.norm(x - p)**2 for p in result.vectors) / 2 for x in X)
    assert result.objective == pytest.approx(objective, abs=1e-10)
    assert np.all(np.diff(result.flags['objective_trace']) >= -1e-9 * len(X))
    assert result.objective >= limited.objective - 1e-9 * len(X)


def test_xor_linear_obstruction_and_nonlinear_separation():
    P = unit([[1, 1], [-1, -1]])
    N = unit([[1, -1], [-1, 1]])
    # Equal sums contradict the strict inequality of sums needed for ordering.
    np.testing.assert_array_equal(P.sum(axis=0), N.sum(axis=0))
    # Every positive-negative pair must have linear score difference >= 1.
    differences = np.array([p - n for p in P for n in N])
    feasibility = linprog(np.zeros(2), A_ub=-differences, b_ub=-np.ones(4),
                          bounds=[(None, None)] * 2, method='highs')
    assert feasibility.status == 2  # Infeasible, rather than a solver failure.
    centers = spherical_prototypes(P, np.array([1, 2]), 2).vectors
    scores = signed_scores(np.vstack([P, N]), centers, N, 1)
    np.testing.assert_allclose(scores, [1., 1., -1., -1.], atol=1e-15)


def test_positive_retrieval_drops_signed_winner():
    P = np.array([[1., 0.]])
    N = np.array([[.5, np.sqrt(3) / 2]])
    X = np.array([[1., 0.], [np.sqrt(3) / 2, -.5]])
    assert np.argmax(signed_scores(X, P, N, 0)) == 0
    np.testing.assert_allclose(signed_scores(X, P, N, 1), [.5, np.sqrt(3) / 2], atol=1e-15)
    assert np.argmax(signed_scores(X, P, N, 1)) == 1
    assert np.all(X @ N.T >= -1e-15)


@pytest.mark.parametrize('penalty', [-1, float('nan'), float('inf')])
def test_invalid_penalty_rejected(penalty):
    X, P, N = local_control()
    with pytest.raises(ValueError, match='penalty'):
        signed_scores(X, P, N, penalty)


@pytest.mark.parametrize('chunk', [0, -1, 1.5, True])
def test_invalid_chunk_rejected(chunk):
    X, P, N = local_control()
    with pytest.raises(ValueError, match='chunk_size'):
        signed_scores(X, P, N, 1, chunk_size=chunk)


@pytest.mark.parametrize('bad', [np.array([[np.nan, 0.]]), np.array([[2., 0.]]), np.array([1., 0.])])
def test_invalid_vectors_rejected(bad):
    good = np.array([[1., 0.]])
    with pytest.raises(ValueError):
        signed_scores(bad, good, good, 1)
    with pytest.raises(ValueError):
        ratio_scores(good, bad, good)
    with pytest.raises(ValueError):
        spherical_prototypes(bad, np.array([1]), 1)


def test_score_dimensions_must_match():
    X, P, N = local_control()
    for scorer in [lambda: signed_scores(X, np.eye(2), N, 1),
                   lambda: ratio_scores(X, P, np.eye(2))]:
        with pytest.raises(ValueError, match='dimension'):
            scorer()


@pytest.mark.parametrize('K,max_updates,ids', [(0, 50, [1, 2]), (True, 50, [1, 2]),
                                               (2, 0, [1, 2]), (2, 1.5, [1, 2]),
                                               (2, 50, [1, 1]), (2, 50, [1]),
                                               (2, 50, [1., 2.])])
def test_invalid_constructor_arguments(K, max_updates, ids):
    with pytest.raises(ValueError):
        spherical_prototypes(np.eye(2), np.array(ids), K, max_updates=max_updates)
