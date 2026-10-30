"""Audited MovieLens acquisition, exact temporal episodes, and isolated outcomes.

Dataset access belongs to the data auditor. Models receive only History/Episode
records. VerifiedProtocol is an application boundary issued by the later live
protocol verifier, not an unforgeable operating-system capability.
"""

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
import urllib.error
import urllib.request
import zipfile

import numpy as np

from .protocol import StudyConfig, VerifiedProtocol, canonical_sha256


@dataclass(frozen=True, slots=True)
class Rating:
    user_id: int
    item_id: int
    rating: int
    timestamp: int


@dataclass(frozen=True, slots=True)
class Movie:
    item_id: int
    title: str
    genres: tuple[str, ...]


@dataclass(frozen=True)
class History:
    user_id: int
    cutoff: int
    role: str
    positive_ids: tuple[int, ...]
    negative_ids: tuple[int, ...]


@dataclass(frozen=True)
class Episode:
    history: History
    candidate_ids: tuple[int, ...]


@dataclass(frozen=True)
class Dataset:
    ratings: tuple[Rating, ...]
    metadata: Mapping[int, Movie]
    manifest: dict


@dataclass(frozen=True)
class EpisodeStore:
    development: tuple[Episode, ...]
    confirmation: tuple[Episode, ...]
    audit: dict
    paths: Mapping[str, Path]


@dataclass(frozen=True)
class _Source:
    url: str
    sha256: str
    size: int


SOURCES = {
    "ml-1m": _Source(
        "https://files.grouplens.org/datasets/movielens/ml-1m.zip",
        "a6898adb50b9ca05aa231689da44c217cb524e7ebd39d264c56e2832f2c54e20", 5917549),
    "ml-100k": _Source(
        "https://files.grouplens.org/datasets/movielens/ml-100k.zip",
        "50d2a982c66986937beb9ffb3aa76efe955bf3d5c6b761f4e3a7cd717c6a3229", 4924029),
}
_OFFICIAL_PAGE = "https://grouplens.org/datasets/movielens/1m/"
_EXPECTED_COUNTS = {
    "ml-1m": {"raw_ratings": 1000209, "raw_users": 6040, "metadata_movies": 3883,
              "rated_movies": 3706, "positive_ratings": 575281, "negative_ratings": 163731,
              "neutral_ratings": 261197},
    "ml-100k": {"raw_ratings": 100000, "raw_users": 943, "metadata_movies": 1682,
                "rated_movies": 1682},
}
_EXPECTED_AUDIT = {
    "ml-1m": {
        "funnel.eligible_users": 2194, "funnel.development_users": 418,
        "funnel.confirmation_users": 1776, "medians.history_records": 168,
        "medians.evaluation_records": 74, "medians.history_positive": 91.5,
        "medians.history_negative": 23, "medians.evaluation_positive": 33,
        "medians.evaluation_negative": 14, "medians.split_gap_seconds": 33,
        "caps.positive_users_truncated": 730, "caps.negative_users_truncated": 347,
        "bursts.record_span_under_day_users": 976, "bursts.same_utc_date_boundary_users": 2154,
    },
    "ml-100k": {"funnel.eligible_users": 287, "funnel.development_users": 62,
                "funnel.confirmation_users": 225},
}


def _sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True, ensure_ascii=False, allow_nan=False,
                                    separators=(",", ":")), encoding="utf-8")
    temporary.replace(path)


def _verified_archive(path: Path, name: str) -> None:
    source = SOURCES[name]
    if path.stat().st_size != source.size or _sha256(path) != source.sha256:
        raise ValueError(f"{name} archive size/SHA256 discrepancy; inspect README/schema and document resolution")


def _acquire_archive(config: StudyConfig) -> tuple[Path, str, list[dict]]:
    """Bounded access fallback, with persisted attempts across interrupted/resumed calls.

    The stage owner applies the cumulative setup/time admission before calling.
    Integrity/schema discrepancies never authorize an access fallback.
    """
    raw = config.paths["data"] / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    evidence = config.paths["results"] / "data_acquisition.json"
    payload = json.loads(evidence.read_text()) if evidence.exists() else {"schema_version": 1, "attempts": []}
    records = payload["attempts"]

    def persist():
        payload["output_sha256"] = canonical_sha256({"attempts": records})
        payload["input_sha256"] = canonical_sha256({name: asdict(source) for name, source in SOURCES.items()})
        _write_json(evidence, payload)

    def request(url, kind, destination=None):
        record = {"kind": kind, "url": url, "started_at": datetime.now(timezone.utc).isoformat(),
                  "status": "started"}
        records.append(record)
        persist()  # An interrupted retrieval still consumes an attempt.
        temporary = destination.with_suffix(".zip.part") if destination else None
        try:
            with urllib.request.urlopen(urllib.request.Request(url), timeout=30) as response:
                if temporary:
                    with temporary.open("wb") as handle:
                        size = 0
                        while block := response.read(1024 * 1024):
                            size += len(block)
                            if size > 10 * 1024 * 1024:
                                raise ValueError("Archive exceeds the allowed source size")
                            handle.write(block)
                else:
                    response.read(1024 * 1024)
            if temporary:
                temporary.replace(destination)
            record["status"] = "retrieved"
        except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
            record.update(status="unavailable", error=str(error))
        finally:
            if temporary:
                temporary.unlink(missing_ok=True)
            record["ended_at"] = datetime.now(timezone.utc).isoformat()
            persist()
        return record["status"] == "retrieved"

    primary = raw / "ml-1m.zip"
    if primary.exists():
        _verified_archive(primary, "ml-1m")
        return primary, "ml-1m", records
    primary_records = [r for r in records if r["kind"] == "primary"]
    if len(primary_records) > 2 or any(r["status"] != "unavailable" for r in primary_records):
        raise ValueError("Incomplete/interrupted or integrity discrepancy in primary retrieval evidence")
    while sum(r["kind"] == "primary" for r in records) < 2:
        if request(SOURCES["ml-1m"].url, "primary", primary):
            _verified_archive(primary, "ml-1m")
            return primary, "ml-1m", records
    if not any(r["kind"] == "official_page" for r in records):
        request(_OFFICIAL_PAGE, "official_page")
    page_records = [r for r in records if r["kind"] == "official_page"]
    if len(page_records) != 1 or page_records[0]["status"] not in ("retrieved", "unavailable"):
        raise ValueError("Incomplete/interrupted official-page check; fallback remains closed")
    fallback = raw / "ml-100k.zip"
    if not fallback.exists():
        if any(r["kind"] == "fallback" for r in records) or not request(
                SOURCES["ml-100k"].url, "fallback", fallback):
            raise OSError("Official 1M and bounded 100K access fallback unavailable")
    _verified_archive(fallback, "ml-100k")
    return fallback, "ml-100k", records


def _read_archive(path: Path, name: str, expected_sha256: str) -> Dataset:
    if _sha256(path) != expected_sha256:
        raise ValueError("Archive SHA256 discrepancy; inspect README/schema and document resolution")
    if name not in SOURCES:
        raise ValueError("Unsupported dataset schema")
    members = ("ratings.dat", "movies.dat", "README") if name == "ml-1m" else (
        "u.data", "u.item", "u.genre", "README")
    with zipfile.ZipFile(path) as archive:
        if any(archive.namelist().count(f"{name}/{member}") != 1 for member in members):
            raise ValueError("Missing or duplicate required archive schema members")
        # Read only named inputs: never extract arbitrary paths or demographics.
        contents = {member: archive.read(f"{name}/{member}") for member in members}
    readme = path.parent / name / "README"
    readme.parent.mkdir(parents=True, exist_ok=True)
    readme.write_bytes(contents["README"])
    metadata = {}
    genre_ids = {}
    if name == "ml-100k":
        for line in contents["u.genre"].decode("latin-1").splitlines():
            if not line.strip():
                continue
            parts = line.split("|")
            if len(parts) != 2 or not parts[0] or not parts[1].isdigit():
                raise ValueError("Invalid u.genre schema")
            index = int(parts[1])
            if index in genre_ids or parts[0] in genre_ids.values():
                raise ValueError("Duplicate u.genre schema entry")
            genre_ids[index] = parts[0]
        if set(genre_ids) != set(range(19)):
            raise ValueError("Expected 19 indexed u.genre fields")
    movie_member = "movies.dat" if name == "ml-1m" else "u.item"
    for line in contents[movie_member].decode("latin-1").splitlines():
        parts = line.split("::" if name == "ml-1m" else "|")
        if len(parts) != (3 if name == "ml-1m" else 24):
            raise ValueError(f"Invalid {movie_member} schema fields")
        item_id = int(parts[0])
        if item_id in metadata:
            raise ValueError("Duplicate metadata item ID")
        if name == "ml-1m":
            genres = parts[2].split("|")
        else:
            if any(flag not in ("0", "1") for flag in parts[5:]):
                raise ValueError("Invalid genre indicator schema")
            genres = [genre_ids[i] for i, flag in enumerate(parts[5:]) if flag == "1"]
        metadata[item_id] = Movie(item_id, parts[1], tuple(sorted(genres)))
    rating_member = "ratings.dat" if name == "ml-1m" else "u.data"
    ratings = []
    for line in contents[rating_member].decode("ascii").splitlines():
        parts = line.split("::") if name == "ml-1m" else line.split()
        if len(parts) != 4:
            raise ValueError(f"Invalid {rating_member} schema fields")
        ratings.append(Rating(*map(int, parts)))
    manifest = {
        "schema_version": 1, "dataset": name, "archive_sha256": expected_sha256,
        "archive_bytes": path.stat().st_size, "source_url": SOURCES[name].url,
        "readme_path": str(readme), "files": {
            member: {"sha256": hashlib.sha256(value).hexdigest(), "bytes": len(value)}
            for member, value in contents.items()},
    }
    dataset = Dataset(tuple(ratings), MappingProxyType(metadata), manifest)
    _validate(dataset)
    return dataset


def _validate(dataset: Dataset) -> None:
    if not dataset.ratings or not dataset.metadata:
        raise ValueError("Dataset ratings and metadata must be nonempty")
    for item_id, movie in dataset.metadata.items():
        if type(item_id) is not int or item_id <= 0 or movie.item_id != item_id:
            raise ValueError("Metadata IDs must be positive integers and match their mapping")
        if not movie.title.strip() or not movie.genres or any(not g.strip() for g in movie.genres):
            raise ValueError("Metadata title and genres must be nonempty")
    pairs = set()
    for row in dataset.ratings:
        if any(type(value) is not int for value in (row.user_id, row.item_id, row.rating, row.timestamp)):
            raise ValueError("Rating IDs, rating, and timestamp must be finite integers")
        if row.user_id <= 0 or row.item_id <= 0:
            raise ValueError("Rating IDs must be positive integers")
        if not 1 <= row.rating <= 5:
            raise ValueError("Invalid rating range")
        if not 0 <= row.timestamp <= 253402300799:
            raise ValueError("Invalid Unix timestamp")
        if (row.user_id, row.item_id) in pairs:
            raise ValueError("Duplicate user-item pair")
        pairs.add((row.user_id, row.item_id))
        if row.item_id not in dataset.metadata:
            raise ValueError("Missing rating-to-metadata join")


def _counts(dataset: Dataset) -> dict:
    return {"raw_ratings": len(dataset.ratings), "raw_users": len({r.user_id for r in dataset.ratings}),
            "metadata_movies": len(dataset.metadata), "rated_movies": len({r.item_id for r in dataset.ratings}),
            "positive_ratings": sum(r.rating >= 4 for r in dataset.ratings),
            "negative_ratings": sum(r.rating <= 2 for r in dataset.ratings),
            "neutral_ratings": sum(r.rating == 3 for r in dataset.ratings)}


def _discrepancies(actual, expected):
    return {key: {"expected": value, "observed": actual[key]} for key, value in expected.items()
            if actual[key] != value}


def load_dataset(config: StudyConfig) -> Dataset:
    """Acquire and parse the pinned official archive, stopping on discrepancies."""
    path, name, attempts = _acquire_archive(config)
    dataset = _read_archive(path, name, SOURCES[name].sha256)
    manifest = dataset.manifest
    manifest["access_fallback"] = name == "ml-100k"
    manifest["acquisition_attempts_sha256"] = canonical_sha256({"attempts": attempts})
    manifest["counts"] = _counts(dataset)
    manifest["discrepancies"] = _discrepancies(manifest["counts"], _EXPECTED_COUNTS[name])
    manifest["status"] = "discrepancy" if manifest["discrepancies"] else "validated"
    manifest["input_sha256"] = SOURCES[name].sha256
    manifest["output_sha256"] = canonical_sha256({"files": manifest["files"], "counts": manifest["counts"]})
    _write_json(config.paths["results"] / "data_manifest.json", manifest)
    if manifest["discrepancies"]:
        raise ValueError("Official dataset count discrepancy; inspect README/schema and document resolution")
    return dataset


def temporal_cutoff(ratings: Sequence[Rating]) -> int:
    """Integer 70% boundary over all rows, including neutral and tied ratings."""
    if not ratings:
        raise ValueError("Cannot split empty ratings")
    ordered = sorted(ratings, key=lambda row: (row.timestamp, row.item_id))
    return ordered[(7 * len(ordered)) // 10].timestamp


def _episode_records(store: EpisodeStore) -> dict:
    return {"schema_version": 1, **{
        role: [asdict(episode) for episode in sorted(getattr(store, role), key=lambda e: e.history.user_id)]
        for role in ("development", "confirmation")}}


def episode_store_digest(store: EpisodeStore) -> str:
    """Bind stable model-facing records, excluding paths, audits, and labels."""
    return canonical_sha256(_episode_records(store))


def make_episodes(dataset: Dataset, config: StudyConfig) -> EpisodeStore:
    """Audit exact splits before emitting separately stored, hashed outcomes."""
    _validate(dataset)
    input_digest = canonical_sha256({
        "ratings": [[r.user_id, r.item_id, r.rating, r.timestamp] for r in dataset.ratings],
        "metadata": [asdict(dataset.metadata[key]) for key in sorted(dataset.metadata)],
    })
    users = defaultdict(list)
    for row in dataset.ratings:
        users[row.user_id].append(row)
    roles = {"development": [], "confirmation": []}
    labels = {"development": {}, "confirmation": {}}
    funnel = dict.fromkeys(("raw_users", "history_positive_at_least_10", "history_negative_at_least_5",
                           "evaluation_positive_at_least_5", "evaluation_negative_at_least_5",
                           "eligible_users", "development_users", "confirmation_users"), 0)
    funnel["raw_users"] = len(users)
    measurements = {key: [] for key in ("history_records", "evaluation_records", "history_positive",
                                       "history_negative", "evaluation_positive", "evaluation_negative",
                                       "split_gap_seconds")}
    caps = {"positive_users_truncated": 0, "negative_users_truncated": 0}
    bursts = {"record_span_under_day_users": 0, "same_utc_date_boundary_users": 0}
    threshold_keys = tuple(funnel)[1:5]
    for user_id, rows in users.items():
        rows.sort(key=lambda r: (r.timestamp, r.item_id))
        cutoff = temporal_cutoff(rows)
        earlier = [r for r in rows if r.timestamp < cutoff]
        later = [r for r in rows if r.timestamp >= cutoff]
        hp, hn = [r for r in earlier if r.rating >= 4], [r for r in earlier if r.rating <= 2]
        ep, en = [r for r in later if r.rating >= 4], [r for r in later if r.rating <= 2]
        for key, count, threshold in zip(threshold_keys, (len(hp), len(hn), len(ep), len(en)), config.eligibility):
            if count < threshold:
                break
            funnel[key] += 1
        else:
            digest = hashlib.sha256((config.hash_prefix + str(user_id)).encode("utf-8")).hexdigest()
            role = "development" if int(digest[:8], 16) % config.hash_modulus == 0 else "confirmation"
            funnel["eligible_users"] += 1
            funnel[role + "_users"] += 1
            candidates = [r for r in later if r.rating != 3]
            history = History(user_id, cutoff, role, tuple(r.item_id for r in hp[-config.support_caps[0]:]),
                              tuple(r.item_id for r in hn[-config.support_caps[1]:]))
            roles[role].append(Episode(history, tuple(r.item_id for r in candidates)))
            labels[role][user_id] = np.array([int(r.rating >= 4) for r in candidates], dtype=np.int8)
            values = (len(earlier), len(later), len(hp), len(hn), len(ep), len(en),
                      later[0].timestamp - earlier[-1].timestamp)
            for key, value in zip(measurements, values):
                measurements[key].append(value)
            caps["positive_users_truncated"] += len(hp) > config.support_caps[0]
            caps["negative_users_truncated"] += len(hn) > config.support_caps[1]
            bursts["record_span_under_day_users"] += rows[-1].timestamp - rows[0].timestamp < 86400
            bursts["same_utc_date_boundary_users"] += earlier[-1].timestamp // 86400 == later[0].timestamp // 86400
    for episodes in roles.values():
        episodes.sort(key=lambda e: hashlib.sha256((config.hash_prefix + str(e.history.user_id)).encode()).hexdigest())
    audit = {"schema_version": 1, "dataset": dataset.manifest["dataset"], "input_sha256": input_digest,
             "counts": _counts(dataset), "funnel": funnel, "caps": caps, "bursts": bursts,
             "medians": {key: float(np.median(value)) if value else None for key, value in measurements.items()}}
    expected = _EXPECTED_AUDIT.get(dataset.manifest["dataset"], {})
    observed = {key: audit[key.split(".")[0]][key.split(".")[1]] for key in expected}
    audit["discrepancies"] = _discrepancies(observed, expected)
    audit["status"] = "discrepancy" if audit["discrepancies"] else "validated"
    paths = {"audit": config.paths["results"] / "cohort_audit.json"}
    if audit["discrepancies"]:
        audit["output_sha256"] = canonical_sha256({"funnel": funnel, "medians": audit["medians"],
                                                    "caps": caps, "bursts": bursts})
        _write_json(paths["audit"], audit)
        raise ValueError("Official cohort count discrepancy; document resolution before progression")
    store = EpisodeStore(tuple(roles["development"]), tuple(roles["confirmation"]), audit, paths)
    episodes_digest = episode_store_digest(store)
    root = config.paths["data"] / "derived" / episodes_digest
    paths["episodes"] = root / "episodes.json"
    outcomes = {}
    for role in roles:
        directory = root / "outcomes" / role
        paths[role + "_outcomes"] = directory
        directory.mkdir(parents=True, exist_ok=True)
        outcomes[role] = {}
        for user_id, values in labels[role].items():
            destination = directory / f"{user_id}.npy"
            # Never overwrite previously emitted outcomes on a resumed construction.
            if destination.exists():
                previous = np.load(destination, allow_pickle=False)
                if previous.dtype != values.dtype or not np.array_equal(previous, values):
                    raise ValueError("Existing outcome integrity discrepancy; preserve and document resolution")
            else:
                with destination.with_suffix(".npy.tmp").open("wb") as handle:
                    np.save(handle, values, allow_pickle=False)
                destination.with_suffix(".npy.tmp").replace(destination)
            outcomes[role][str(user_id)] = _sha256(destination)
    records = _episode_records(store)
    _write_json(paths["episodes"], {**records, "input_sha256": input_digest, "output_sha256": episodes_digest})
    audit["episodes_sha256"] = episodes_digest
    audit["outputs"] = {"episodes_file_sha256": _sha256(paths["episodes"]), "outcome_sha256": outcomes}
    audit["output_sha256"] = canonical_sha256(audit["outputs"])
    _write_json(paths["audit"], audit)
    return EpisodeStore(store.development, store.confirmation, audit, MappingProxyType(paths))


def _load_outcomes(store: EpisodeStore, user_id: int, role: str) -> np.ndarray:
    if type(user_id) is not int:
        raise PermissionError("Outcome access requires an integer user ID")
    episode = next((e for e in getattr(store, role) if e.history.user_id == user_id), None)
    if episode is None:
        raise PermissionError(f"User is outside the {role} outcome partition")
    if store.audit["episodes_sha256"] != episode_store_digest(store):
        raise ValueError("Episode store digest integrity discrepancy")
    path = store.paths[role + "_outcomes"] / f"{user_id}.npy"
    expected = store.audit["outputs"]["outcome_sha256"][role][str(user_id)]
    if _sha256(path) != expected:
        raise ValueError("Outcome hash integrity discrepancy")
    values = np.load(path, allow_pickle=False)
    if values.dtype != np.int8 or values.shape != (len(episode.candidate_ids),) or not np.isin(values, (0, 1)).all():
        raise ValueError("Invalid outcome schema or candidate alignment")
    values.flags.writeable = False
    return values


def load_development_outcomes(store: EpisodeStore, user_id: int) -> np.ndarray:
    return _load_outcomes(store, user_id, "development")


def load_episode_store(audit_path: Path, episodes_path: Path) -> EpisodeStore:
    """Reload validated model-facing records without opening either outcome set."""
    audit = json.loads(Path(audit_path).read_text())
    records = json.loads(Path(episodes_path).read_text())
    if audit.get('status') != 'validated' or audit.get('schema_version') != 1:
        raise ValueError('Episode audit is not validated')
    if _sha256(Path(episodes_path)) != audit['outputs']['episodes_file_sha256']:
        raise ValueError('Episode file integrity discrepancy')
    if canonical_sha256(audit['outputs']) != audit['output_sha256']:
        raise ValueError('Episode audit output integrity discrepancy')
    roles = {}
    for role in ('development', 'confirmation'):
        episodes = []
        for record in records[role]:
            h = record['history']
            if h['role'] != role:
                raise ValueError('Episode role mismatch')
            history = History(h['user_id'], h['cutoff'], role, tuple(h['positive_ids']), tuple(h['negative_ids']))
            ids = (*history.positive_ids, *history.negative_ids, *record['candidate_ids'], history.user_id)
            if any(type(i) is not int for i in ids):
                raise ValueError('Episode IDs must be integers')
            episodes.append(Episode(history, tuple(record['candidate_ids'])))
        if len({e.history.user_id for e in episodes}) != len(episodes):
            raise ValueError('Duplicate episode user')
        roles[role] = tuple(episodes)
    paths = {'audit': Path(audit_path), 'episodes': Path(episodes_path), **{
        role+'_outcomes': Path(episodes_path).parent/'outcomes'/role for role in roles}}
    store = EpisodeStore(roles['development'], roles['confirmation'], audit, MappingProxyType(paths))
    if episode_store_digest(store) != audit['episodes_sha256'] or records['output_sha256'] != audit['episodes_sha256']:
        raise ValueError('Episode store digest discrepancy')
    if {e.history.user_id for e in store.development} & {e.history.user_id for e in store.confirmation}:
        raise ValueError('Overlapping outcome roles')
    return store


def load_confirmation_outcomes(store: EpisodeStore, user_id: int, permit: VerifiedProtocol) -> np.ndarray:
    """Require the verifier's locked confirmation IDs and exact episode digest."""
    if not isinstance(permit, VerifiedProtocol) or not isinstance(permit.payload, dict):
        raise PermissionError("Confirmation outcomes require a verified protocol permit")
    access = permit.payload.get("data_access")
    if not isinstance(access, dict):
        raise PermissionError("Protocol permit lacks locked data access")
    locked_ids = access.get("confirmation_user_ids")
    if (not isinstance(locked_ids, (list, tuple)) or any(type(u) is not int for u in locked_ids)
            or len(set(locked_ids)) != len(locked_ids)
            or user_id not in locked_ids or access.get("episodes_sha256") != episode_store_digest(store)):
        raise PermissionError("Protocol permit does not match the locked confirmation episode store")
    return _load_outcomes(store, user_id, "confirmation")
