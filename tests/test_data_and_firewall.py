"""Independent tiny local data, split, acquisition, and evidence-boundary checks."""

from dataclasses import fields, replace
import hashlib
import io
import json
from pathlib import Path
import urllib.error
import zipfile

import numpy as np
import pytest

from signed_study import data
from signed_study.data import (
    Dataset, Episode, History, Movie, Rating, make_episodes,
    episode_store_digest, load_confirmation_outcomes, load_dataset, load_development_outcomes, temporal_cutoff,
)
from signed_study.protocol import VerifiedProtocol


def dataset_for(ratings):
    return Dataset(tuple(ratings), {r.item_id: Movie(r.item_id, f"Movie {r.item_id} (2000)", ("Drama",))
                                  for r in ratings}, {"schema_version": 1, "dataset": "synthetic"})


def all_episodes(store):
    return (*store.development, *store.confirmation)


def test_integer_boundary_and_tied_timestamps(ninety_ratings, local_config):
    assert temporal_cutoff(ninety_ratings) == ninety_ratings[63].timestamp
    assert 7 * len(ninety_ratings) // 10 == 63
    # The tie begins before index 63; all five tied records must be in evaluation.
    tied = tuple(replace(r, timestamp=1061) if 61 <= i <= 65 else r
                 for i, r in enumerate(ninety_ratings))
    store = make_episodes(dataset_for(tied[::-1]), local_config)
    episode, = all_episodes(store)
    assert episode.history.cutoff == 1061
    history_timestamps = [r.timestamp for r in tied if r.timestamp < 1061]
    evaluation_timestamps = [r.timestamp for r in tied if r.timestamp >= 1061]
    assert max(history_timestamps) < min(evaluation_timestamps)
    history_item_ids = episode.history.positive_ids + episode.history.negative_ids
    candidate_item_ids = episode.candidate_ids
    assert not set(history_item_ids) & set(candidate_item_ids)
    assert tied[61].item_id in candidate_item_ids
    assert tied[62].item_id not in candidate_item_ids  # neutral
    assert tied[63].item_id in candidate_item_ids


def test_neutral_rows_set_boundary_but_never_become_labels(ninety_ratings, local_config):
    episode, = all_episodes(make_episodes(dataset_for(ninety_ratings), local_config))
    assert episode.history.cutoff == 1063
    assert len(episode.history.positive_ids) == 21
    assert len(episode.history.negative_ids) == 21
    assert len(episode.candidate_ids) == 18
    assert episode.candidate_ids[:2] == (199, 202)
    assert {f.name for f in fields(Episode)} == {"history", "candidate_ids"}
    assert {f.name for f in fields(History)} == {
        "user_id", "cutoff", "role", "positive_ids", "negative_ids"}


def test_caps_choose_newest_with_item_id_tiebreak(local_config):
    # Boundary is index 350. History has 200 positive and 150 negative records.
    ratings = [Rating(1, i + 1, 5 if i < 200 else 1, 1000 + i // 3) for i in range(350)]
    ratings += [Rating(1, i + 1, 5 if i % 2 else 1, 3000 + i) for i in range(350, 500)]
    store = make_episodes(dataset_for(ratings[::-1]), local_config)
    episode, = all_episodes(store)
    assert episode.history.positive_ids == tuple(range(73, 201))
    assert episode.history.negative_ids == tuple(range(287, 351))
    assert episode.candidate_ids == tuple(range(351, 501))
    assert store.audit["caps"] == {"positive_users_truncated": 1, "negative_users_truncated": 1}


def test_funnel_hash_roles_and_burst_statistics(ninety_ratings, local_config):
    # Literal hash-split identities were checked independently of production.
    ratings = tuple(replace(r, user_id=u) for u in (1, 2, 3, 4, 5) for r in ninety_ratings)
    store = make_episodes(dataset_for(ratings), local_config)
    assert {e.history.user_id for e in store.development} == {1, 3}
    assert {e.history.user_id for e in store.confirmation} == {2, 4, 5}
    assert store.audit["funnel"] == {
        "raw_users": 5, "history_positive_at_least_10": 5,
        "history_negative_at_least_5": 5, "evaluation_positive_at_least_5": 5,
        "evaluation_negative_at_least_5": 5, "eligible_users": 5,
        "development_users": 2, "confirmation_users": 3,
    }
    assert store.audit["medians"] == {
        "history_records": 63.0, "evaluation_records": 27.0,
        "history_positive": 21.0, "history_negative": 21.0,
        "evaluation_positive": 9.0, "evaluation_negative": 9.0,
        "split_gap_seconds": 1.0,
    }
    assert store.audit["bursts"] == {"record_span_under_day_users": 5, "same_utc_date_boundary_users": 5}
    assert store.audit["counts"]["neutral_ratings"] == 150
    # Artifacts bind IDs and outcomes, without putting labels in episode JSON.
    artifact = json.loads(store.paths["episodes"].read_text())
    assert artifact["schema_version"] == 1
    assert "input_sha256" in artifact and "output_sha256" in artifact
    assert "labels" not in json.dumps(artifact) and "outcomes" not in json.dumps(artifact)
    assert json.loads(store.paths["audit"].read_text())["funnel"] == store.audit["funnel"]


@pytest.mark.parametrize("failed_class,expected_key", [
    ("history_positive", "history_positive_at_least_10"),
    ("history_negative", "history_negative_at_least_5"),
    ("evaluation_positive", "evaluation_positive_at_least_5"),
    ("evaluation_negative", "evaluation_negative_at_least_5"),
])
def test_each_eligibility_threshold_is_enforced(ninety_ratings, local_config, failed_class, expected_key):
    history, sign = failed_class.split("_")
    indices = [i for i, r in enumerate(ninety_ratings)
               if (i < 63) == (history == "history") and (r.rating >= 4 if sign == "positive" else r.rating <= 2)]
    keep = 9 if failed_class == "history_positive" else 4
    remove_sign = set(indices[keep:])
    ratings = tuple(replace(r, rating=3) if i in remove_sign else r for i, r in enumerate(ninety_ratings))
    store = make_episodes(dataset_for(ratings), local_config)
    assert not all_episodes(store)
    assert store.audit["funnel"][expected_key] == 0


def test_outcomes_align_and_development_cannot_read_confirmation(ninety_ratings, local_config):
    ratings = tuple(replace(r, user_id=u) for u in (1, 2) for r in ninety_ratings)
    store = make_episodes(dataset_for(ratings), local_config)
    np.testing.assert_array_equal(load_development_outcomes(store, 1), [1, 0] * 9)
    with pytest.raises(PermissionError):
        load_development_outcomes(store, 2)
    with pytest.raises(PermissionError):
        load_confirmation_outcomes(store, 2, None)
    with pytest.raises(PermissionError):
        load_confirmation_outcomes(store, 2, VerifiedProtocol("0" * 64, {}))
    assert store.paths["development_outcomes"] != store.paths["confirmation_outcomes"]
    permit = VerifiedProtocol("a" * 64, {"data_access": {
        "confirmation_user_ids": [2], "episodes_sha256": episode_store_digest(store)}})
    np.testing.assert_array_equal(load_confirmation_outcomes(store, 2, permit), [1, 0] * 9)
    for access in [
        {"confirmation_user_ids": [3], "episodes_sha256": episode_store_digest(store)},
        {"confirmation_user_ids": [2], "episodes_sha256": "0" * 64},
        {"confirmation_user_ids": [True], "episodes_sha256": episode_store_digest(store)},
    ]:
        with pytest.raises(PermissionError):
            load_confirmation_outcomes(store, 2, VerifiedProtocol("a" * 64, {"data_access": access}))


def test_episode_digest_ignores_paths_and_binds_selected_ids(ninety_ratings, local_config):
    store = make_episodes(dataset_for(ninety_ratings), local_config)
    assert episode_store_digest(replace(store, paths={})) == episode_store_digest(store)
    episode, = all_episodes(store)
    modified = replace(episode, candidate_ids=episode.candidate_ids[::-1])
    changed = replace(store, development=(modified,))
    assert episode_store_digest(changed) != episode_store_digest(store)


def test_outcome_tampering_is_rejected(ninety_ratings, local_config):
    store = make_episodes(dataset_for(ninety_ratings), local_config)
    np.save(store.paths["development_outcomes"] / "1.npy", np.zeros(18, dtype=np.int8), allow_pickle=False)
    with pytest.raises(ValueError, match="hash|digest|integrity"):
        load_development_outcomes(store, 1)


def test_reconstruction_preserves_previous_outcome_integrity(ninety_ratings, local_config):
    dataset = dataset_for(ninety_ratings)
    store = make_episodes(dataset, local_config)
    destination = store.paths["development_outcomes"] / "1.npy"
    np.save(destination, np.array([1, 0] * 9, dtype=np.float64), allow_pickle=False)
    before = destination.read_bytes()
    with pytest.raises(ValueError, match="integrity|schema"):
        make_episodes(dataset, local_config)
    assert destination.read_bytes() == before


def test_official_cohort_mismatch_saves_audit_before_any_outcomes(ninety_ratings, local_config):
    dataset = replace(dataset_for(ninety_ratings), manifest={"schema_version": 1, "dataset": "ml-1m"})
    with pytest.raises(ValueError, match="cohort|count|discrepancy"):
        make_episodes(dataset, local_config)
    audit = json.loads((local_config.paths["results"] / "cohort_audit.json").read_text())
    assert audit["status"] == "discrepancy"
    assert audit["discrepancies"]["funnel.eligible_users"] == {"expected": 2194, "observed": 1}
    assert not (local_config.paths["data"] / "derived").exists()


@pytest.mark.parametrize("mutation,match", [
    ("duplicate_pair", "(?i)duplicate"), ("missing_join", "metadata|join"),
    ("rating_range", "rating"), ("bad_timestamp", "timestamp"),
    ("nonfinite", "rating|integer|finite"),
])
def test_invalid_rows_fail_before_splitting(ninety_ratings, local_config, mutation, match):
    dataset = dataset_for(ninety_ratings)
    if mutation == "duplicate_pair":
        dataset = replace(dataset, ratings=dataset.ratings + (dataset.ratings[0],))
    elif mutation == "missing_join":
        dataset = replace(dataset, metadata={k: v for k, v in dataset.metadata.items() if k != 10})
    else:
        update = {"rating_range": {"rating": 6}, "bad_timestamp": {"timestamp": -1},
                  "nonfinite": {"rating": float("nan")}}[mutation]
        dataset = replace(dataset, ratings=(replace(dataset.ratings[0], **update), *dataset.ratings[1:]))
    with pytest.raises(ValueError, match=match):
        make_episodes(dataset, local_config)


GENRES = ("unknown", "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime",
          "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery",
          "Romance", "Sci-Fi", "Thriller", "War", "Western")


def archive_bytes(dataset_name, *, duplicate_metadata=False, malformed=False):
    target = io.BytesIO()
    with zipfile.ZipFile(target, "w") as archive:
        if dataset_name == "ml-1m":
            archive.writestr("ml-1m/README", "Local fixture license and acknowledgment")
            archive.writestr("ml-1m/ratings.dat", "1::42::5::1000\n1::999::1::1001\n")
            movies = "42::Amélie (2001)::Drama|Comedy\n999::Other (2000)::Drama\n"
            if duplicate_metadata:
                movies += "42::Duplicate (2000)::Drama\n"
            if malformed:
                movies = "42::Too::many::fields\n"
            archive.writestr("ml-1m/movies.dat", movies.encode("latin-1"))
            archive.writestr("ml-1m/users.dat", b"unreadable demographics\xff\x00")
        else:
            archive.writestr("ml-100k/README", "Local fixture license and acknowledgment")
            archive.writestr("ml-100k/u.data", "1\t42\t5\t1000\n1 999 1 1001\n")
            archive.writestr("ml-100k/u.genre", "\n".join(f"{g}|{i}" for i, g in enumerate(GENRES)) + "\n\n")
            rows = []
            for item, title, active in [(42, "Amélie (2001)", {5, 8}), (999, "Other (2000)", {8})]:
                rows.append("|".join([str(item), title, "01-Jan-2001", "", "http://unused",
                                      *(str(int(i in active)) for i in range(19))]))
            if duplicate_metadata:
                rows.append(rows[0])
            if malformed:
                rows[0] = "42|wrong|schema"
            archive.writestr("ml-100k/u.item", ("\n".join(rows) + "\n").encode("latin-1"))
    return target.getvalue()


@pytest.mark.parametrize("dataset_name", ["ml-1m", "ml-100k"])
def test_both_archive_schemas_latin1_noncontiguous_ids_and_readme(tmp_path, dataset_name):
    content = archive_bytes(dataset_name)
    archive = tmp_path / f"{dataset_name}.zip"
    archive.write_bytes(content)
    dataset = data._read_archive(archive, dataset_name, hashlib.sha256(content).hexdigest())
    assert dataset.metadata[42].title == "Amélie (2001)"
    assert dataset.metadata[42].genres == ("Comedy", "Drama")
    assert sorted(dataset.metadata) == [42, 999]
    assert [r.item_id for r in dataset.ratings] == [42, 999]
    assert Path(dataset.manifest["readme_path"]).read_text() == "Local fixture license and acknowledgment"
    assert dataset.manifest["files"]["README"]["sha256"] == hashlib.sha256(
        b"Local fixture license and acknowledgment").hexdigest()


@pytest.mark.parametrize("dataset_name", ["ml-1m", "ml-100k"])
@pytest.mark.parametrize("kind,match", [("duplicate_metadata", "(?i)duplicate"), ("malformed", "schema|fields")])
def test_archive_metadata_anomalies_stop(tmp_path, dataset_name, kind, match):
    content = archive_bytes(dataset_name, **{kind: True})
    archive = tmp_path / "fixture.zip"
    archive.write_bytes(content)
    with pytest.raises(ValueError, match=match):
        data._read_archive(archive, dataset_name, hashlib.sha256(content).hexdigest())


def test_archive_hash_mismatch_stops_without_network(tmp_path, monkeypatch):
    archive = tmp_path / "fixture.zip"
    archive.write_bytes(archive_bytes("ml-1m"))
    monkeypatch.setattr(data.urllib.request, "urlopen", lambda *a, **k: pytest.fail("network must not run"))
    with pytest.raises(ValueError, match="SHA256|digest|hash"):
        data._read_archive(archive, "ml-1m", "0" * 64)


def test_fallback_requires_two_failures_and_official_page_check(local_config, monkeypatch):
    content = archive_bytes("ml-100k")
    sources = dict(data.SOURCES)
    sources["ml-100k"] = replace(sources["ml-100k"], sha256=hashlib.sha256(content).hexdigest(), size=len(content))
    monkeypatch.setattr(data, "SOURCES", sources)
    calls = []

    def official_transport(request, **kwargs):
        url = request.full_url
        calls.append(url)
        if url.endswith("ml-1m.zip"):
            raise urllib.error.URLError("offline primary")
        return io.BytesIO(content if url.endswith("ml-100k.zip") else b"official dataset page")

    monkeypatch.setattr(data.urllib.request, "urlopen", official_transport)
    path, name, records = data._acquire_archive(local_config)
    assert name == "ml-100k" and path.read_bytes() == content
    assert calls == ["https://files.grouplens.org/datasets/movielens/ml-1m.zip"] * 2 + [
        "https://grouplens.org/datasets/movielens/1m/",
        "https://files.grouplens.org/datasets/movielens/ml-100k.zip"]
    assert len(records) == 4
    # Resume reuses the validated fallback and does not create more primary attempts.
    assert data._acquire_archive(local_config)[1] == "ml-100k"
    assert len(calls) == 4


def test_primary_digest_discrepancy_does_not_trigger_access_fallback(local_config, monkeypatch):
    calls = []

    def corrupted_transport(request, **kwargs):
        calls.append(request.full_url)
        return io.BytesIO(b"corrupted official response")

    monkeypatch.setattr(data.urllib.request, "urlopen", corrupted_transport)
    with pytest.raises(ValueError, match="SHA256|digest|hash|size"):
        load_dataset(local_config)
    assert len(calls) == 1


def test_local_io_failure_is_not_an_access_fallback(local_config, monkeypatch):
    real_open = Path.open

    def local_disk_failure(path, *args, **kwargs):
        if path.name.endswith(".part"):
            raise OSError("local disk full")
        return real_open(path, *args, **kwargs)

    calls = []

    def transport(request, **kwargs):
        calls.append(request.full_url)
        return io.BytesIO(archive_bytes("ml-1m"))

    monkeypatch.setattr(Path, "open", local_disk_failure)
    monkeypatch.setattr(data.urllib.request, "urlopen", transport)
    with pytest.raises(OSError, match="disk full"):
        data._acquire_archive(local_config)
    assert len(calls) == 1


@pytest.mark.parametrize("statuses,page_status", [
    (["started", "started"], None),
    (["unavailable", "retrieved"], "retrieved"),
    (["unavailable", "unavailable"], "started"),
])
def test_incomplete_or_integrity_failure_evidence_cannot_authorize_fallback(
        local_config, monkeypatch, statuses, page_status):
    records = [{"kind": "primary", "status": status} for status in statuses]
    if page_status is not None:
        records.append({"kind": "official_page", "status": page_status})
    path = local_config.paths["results"] / "data_acquisition.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"schema_version": 1, "attempts": records}))
    monkeypatch.setattr(data.urllib.request, "urlopen", lambda *a, **k: pytest.fail("fallback must stay closed"))
    with pytest.raises(ValueError, match="interrupted|unavailable|discrepancy|incomplete"):
        data._acquire_archive(local_config)


def test_unexpected_official_counts_save_discrepancy_and_stop(local_config, monkeypatch):
    content = archive_bytes("ml-1m")
    sources = dict(data.SOURCES)
    sources["ml-1m"] = replace(sources["ml-1m"], sha256=hashlib.sha256(content).hexdigest(), size=len(content))
    monkeypatch.setattr(data, "SOURCES", sources)
    raw = local_config.paths["data"] / "raw"
    raw.mkdir(parents=True)
    (raw / "ml-1m.zip").write_bytes(content)
    with pytest.raises(ValueError, match="count|discrepancy"):
        load_dataset(local_config)
    manifest = json.loads((local_config.paths["results"] / "data_manifest.json").read_text())
    assert manifest["status"] == "discrepancy"
    assert manifest["discrepancies"]["raw_ratings"] == {"expected": 1000209, "observed": 2}


def test_empty_cutoff_rejected():
    with pytest.raises(ValueError, match="empty|ratings"):
        temporal_cutoff(())
