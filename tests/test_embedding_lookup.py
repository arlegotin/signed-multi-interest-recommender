"""Exact identity and allocation contracts for repeated embedding row lookup."""

from dataclasses import fields

import numpy as np
import pytest

from signed_study import embed
from signed_study.embed import EmbeddingCache, EmbeddingProbe
from signed_study.metrics import evaluate_user
from signed_study.models import signed_scores


def geometry(n):
    rows = np.zeros((n, 384), dtype=np.float32)
    rows[:, 0] = 1
    rows[:, 1] = np.arange(n)
    return rows


def test_unsigned_ids_cannot_wrap_outside_int64():
    with pytest.raises(ValueError, match="int64|range"):
        EmbeddingCache(np.array([1, 2**63], dtype=np.uint64), geometry(2), {})


@pytest.mark.parametrize("dtype,ids", [
    (np.int64, [-2**63, -1, 2**53, 2**53 + 1, 2**63 - 1]),
    (np.uint64, [0, 2**53, 2**53 + 1, 2**63 - 1]),
])
def test_large_integer_identity_and_order_are_exact(dtype, ids):
    cache = EmbeddingCache(np.array(ids, dtype=dtype), geometry(len(ids)), {})
    requested = [ids[-1], ids[0], ids[-2], ids[-1]]
    np.testing.assert_array_equal(cache.rows(iter(requested)), cache.vectors[[-1, 0, -2, -1]])
    assert cache.item_ids.tolist() == ids
    with pytest.raises(KeyError):
        cache.rows([2**63])


@pytest.mark.parametrize("ids", [[True], [np.bool_(False)], [1.0], ["1"], [None]])
def test_lookup_rejects_malformed_and_boolean_ids(ids):
    cache = EmbeddingCache(np.array([1, 2]), geometry(2), {})
    with pytest.raises(ValueError, match="integer"):
        cache.rows(ids)


def test_lookup_does_not_revisit_catalog_ids(monkeypatch):
    # Counts actual full-catalog traversal, independent of the index's container.
    visited = []
    original = enumerate

    def counting(values, *args):
        for pair in original(values, *args):
            visited.append(pair[0])
            yield pair

    monkeypatch.setattr(embed, "enumerate", counting, raising=False)
    cache = EmbeddingCache(np.arange(64), geometry(64), {})
    for _ in range(20):
        np.testing.assert_array_equal(cache.rows([5, 1, 5]), cache.vectors[[5, 1, 5]])
    assert len(visited) == 64


def test_owned_geometry_manifest_and_rows_are_isolated():
    ids, raw, manifest = np.array([1, 2, 3]), geometry(3), {"input": [1]}
    cache = EmbeddingProbe(ids, raw, manifest)
    expected_raw = raw.copy()
    expected = raw.astype(np.float64)
    expected /= np.linalg.norm(expected, axis=1, keepdims=True)
    ids[:] = 99
    raw[:] = 9
    manifest["input"].append(2)
    np.testing.assert_array_equal(cache.vectors, expected)
    np.testing.assert_array_equal(cache.raw_rows([3, 1, 3]), expected_raw[[2, 0, 2]])
    assert cache.manifest == {"input": [1]}
    for backing in (cache.item_ids, cache.vectors, cache.raw_vectors):
        with pytest.raises(ValueError):
            backing.setflags(write=True)
    selected = cache.rows([3, 1, 3])
    selected[:] = 0
    selected_raw = cache.raw_rows([1, 2])
    selected_raw[:] = 0
    np.testing.assert_array_equal(cache.vectors, expected)
    np.testing.assert_array_equal(cache.raw_vectors, expected_raw)
    assert cache.rows([]).shape == cache.raw_rows([]).shape == (0, 384)
    assert cache.rows([]).dtype == np.float64
    assert cache.raw_rows([]).dtype == np.float32
    assert {f.name for f in fields(EmbeddingCache)} == {"item_ids", "vectors", "manifest"}


def test_lookup_keeps_fixed_scores_and_metrics_exact():
    raw = geometry(7)
    cache = EmbeddingCache(np.arange(1, 8), raw, {})
    direct = raw.astype(np.float64)
    direct /= np.linalg.norm(direct, axis=1, keepdims=True)
    candidate_ids = [7, 3, 5, 6]
    actual = signed_scores(cache.rows(candidate_ids), cache.rows([1, 2]), cache.rows([4]), 2)
    expected = signed_scores(direct[[6, 2, 4, 5]], direct[[0, 1]], direct[[3]], 2)
    np.testing.assert_array_equal(actual, expected)
    labels = np.array([1, 0, 1, 0], dtype=np.int8)
    assert evaluate_user(actual, labels, candidate_ids) == evaluate_user(expected, labels, candidate_ids)
