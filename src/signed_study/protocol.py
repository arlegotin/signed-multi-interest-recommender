"""Validated fixed study settings and canonical protocol identities.

Acquisition resolves model/environment identities. Stage code later enforces the
pre-freeze and budget conditions on recorded, blueprint-authorized adjustments.
"""

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any


_FIXED_SETTINGS = {
    "schema_version": 1,
    "dataset": "ml-1m",
    "model_id": "sentence-transformers/all-MiniLM-L6-v2",
    "model_revision": None,
    "environment_hash": None,
    "text_template": "Movie: {title}. Genres: {genres}.",
    "split_numerator": 7,
    "split_denominator": 10,
    "eligibility": [10, 5, 5, 5],
    "support_caps": [128, 64],
    "hash_prefix": "signed-study:",
    "hash_modulus": 5,
    "K_grid": [1, 2, 4],
    "lambda_grid": [0, 0.25, 0.5, 1, 2, 4, 8],
    "C_grid": [0.01, 0.1, 1, 10, 100],
    "g_grid": [0.25, 1, 4, 16],
    "delta": 0.01,
    "bootstrap_resamples": 10000,
    "bootstrap_seed": 20261029,
    "bootstrap_batch_size": 200,
    "contamination_seeds": [1101, 1102, 1103],
    "encoder": {
        "device": "cpu", "inference_dtype": "float32", "storage_dtype": "float32",
        "batch_size": 32, "max_sequence_length": 256,
        "embedding_dimension": 384, "trust_remote_code": False,
    },
    "numerics": {
        "dtype": "float64", "norm_epsilon": 1e-12, "unit_norm_tolerance": 1e-12,
        "distance_clip_tolerance": 1e-10, "oracle_absolute_tolerance": 1e-10,
        "score_chunk_size": 512, "score_chunk_sizes_check": [1, 7, 512],
        "score_tolerance_factor": 1e-10, "repeat_encoding_tolerance": 1e-5,
        "metric_ties": "exact", "ranking_ties": "ascending_item_id",
    },
    "prototype": {
        "max_updates": 50, "allowed_repair_max_updates": 100,
        "improvement_tolerance_per_row": 1e-10,
        "decrease_tolerance_per_row": 1e-9, "cap_hit_inspection_fraction": 0.05,
    },
    "linear_logistic": {
        "penalty": "l2", "solver": "liblinear", "fit_intercept": True,
        "class_weight": "balanced", "tol": 1e-8, "max_iter": 2000,
        "allowed_repair_max_iter": 4000, "random_state": 0,
    },
    "rbf_svm": {
        "kernel": "rbf", "class_weight": "balanced", "tol": 1e-6,
        "max_iter": 100000, "allowed_repair_max_iter": 200000,
        "cache_size_mb": 64, "probability": False,
        "zero_distance_median_fallback": 1,
    },
    "selection": {
        "initial_entries": 78, "max_additions": 14, "max_entries": 92,
        "tie_tolerance": 1e-6, "expanded_lambda": 16,
        "expanded_C": [0.001, 1000], "expanded_g": [0.0625, 64],
    },
    "calibration": {
        "encoder_probe_items": 128, "repeat_encoding_items": 32,
        "development_users": 32,
        "primary_scale_pairs": [[418, 1776], [256, 1776], [418, 1500],
                                [256, 1500], [418, 1000], [256, 1000]],
        "fallback_scale_pair": [62, 225],
    },
    "contamination": {
        "max_users": 200, "swap_fraction": 0.1,
        "minimum_extra_seconds": 1200,
    },
    "timing": {
        "users": 32, "warmups": 1, "repetitions": 5,
        "blas_threads": 1, "allowed_blas_threads": [1, 4],
        "inference_workers": 1, "amortization_requests": 1000,
    },
    "resources": {
        "experimental_seconds": 14400, "setup_seconds": 2700,
        "report_reserve_seconds": 1800, "repair_seconds": 1200,
        "repair_cycles": 2, "evaluation_limit": 100000,
        "disk_growth_bytes": 6442450944, "initial_free_disk_bytes": 12884901888,
        "available_memory_headroom_bytes": 4294967296,
        "rss_bytes": 8589934592, "rss_ram_fraction": 0.25,
        "rss_sample_seconds": 0.1, "safety_multiplier": 1.5, "workers": 1,
        "stage_seconds": {"G0": 600, "G1": 600, "G2": 1200, "G3": 2400,
                          "G4": 4800, "G5": 1200, "G6": 1800, "repair": 1200},
    },
    "paths": {"cache": ".cache", "data": "data", "results": "results"},
    "pre_freeze_repairs": [],
}


@dataclass(frozen=True)
class StudyConfig:
    """Validated scientific settings, resource limits, and local output paths.

    Obtain configurations through :func:`load_config`. Nested settings are
    immutable mappings, and JSON lists become tuples.
    """

    schema_version: int
    dataset: str
    model_id: str
    model_revision: str | None
    environment_hash: str | None
    text_template: str
    split_numerator: int
    split_denominator: int
    eligibility: tuple[int, ...]
    support_caps: tuple[int, int]
    hash_prefix: str
    hash_modulus: int
    K_grid: tuple[int, ...]
    lambda_grid: tuple[float, ...]
    C_grid: tuple[float, ...]
    g_grid: tuple[float, ...]
    delta: float
    bootstrap_resamples: int
    bootstrap_seed: int
    bootstrap_batch_size: int
    contamination_seeds: tuple[int, ...]
    encoder: Mapping[str, Any]
    numerics: Mapping[str, Any]
    prototype: Mapping[str, Any]
    linear_logistic: Mapping[str, Any]
    rbf_svm: Mapping[str, Any]
    selection: Mapping[str, Any]
    calibration: Mapping[str, Any]
    contamination: Mapping[str, Any]
    timing: Mapping[str, Any]
    resources: Mapping[str, Any]
    paths: Mapping[str, Path]
    pre_freeze_repairs: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True)
class VerifiedProtocol:
    """Lock identity and payload issued to application code by the lock verifier.

    This record does not itself verify a lock. Only the later verification
    boundary may issue it after checking live inputs and prerequisites.
    """

    lock_hash: str
    payload: dict


def canonical_sha256(value: dict) -> str:
    """Hash compact sorted UTF-8 JSON, rejecting nonfinite JSON numbers."""
    if not isinstance(value, dict):
        raise ValueError("Canonical JSON requires a dictionary")
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError(f"JSON numbers must be finite: {value}")


def _check_finite(value):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("JSON numbers must be finite")
    if isinstance(value, dict):
        for child in value.values():
            _check_finite(child)
    elif isinstance(value, list):
        for child in value:
            _check_finite(child)


_VARIABLE_FIELDS = {
    "model_revision", "environment_hash", "paths", "pre_freeze_repairs",
    "prototype.max_updates", "linear_logistic.max_iter", "rbf_svm.max_iter",
    "encoder.batch_size", "numerics.score_chunk_size", "bootstrap_batch_size",
    "timing.blas_threads",
}


def _validate_fixed(actual, expected, path=""):
    if path in _VARIABLE_FIELDS:
        return
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            raise ValueError(f"{path or 'configuration'} must be an object")
        missing = expected.keys() - actual.keys()
        unknown = actual.keys() - expected.keys()
        if missing or unknown:
            raise ValueError(f"{path or 'configuration'} keys: missing {sorted(missing)}, unknown {sorted(unknown)}")
        for key in expected:
            _validate_fixed(actual[key], expected[key], f"{path}.{key}" if path else key)
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError(f"{path} must equal {expected!r}")
        for index, (item, required) in enumerate(zip(actual, expected)):
            _validate_fixed(item, required, f"{path}[{index}]")
    else:
        # In JSON, booleans and numeric 1/0 are distinct protocol values.
        correct_type = type(actual) is type(expected)
        if isinstance(expected, float):
            correct_type = type(actual) in (int, float)
        if not correct_type or actual != expected:
            raise ValueError(f"{path} must equal {expected!r}")


def _validate_repairs(settings):
    repairs = settings["pre_freeze_repairs"]
    if not isinstance(repairs, list) or len(repairs) > 2:
        raise ValueError("pre_freeze_repairs must contain at most two repair records")
    kinds = set()
    for repair in repairs:
        if not isinstance(repair, dict) or set(repair) != {"kind", "stage", "reason"}:
            raise ValueError("Each repair requires kind, stage, and reason")
        kind = repair["kind"]
        if not isinstance(kind, str) or kind not in {"prototype_iterations", "solver_iterations"} or kind in kinds:
            raise ValueError("repair kind must be a unique authorized global iteration repair")
        if repair["stage"] not in ("G1", "G2", "G3"):
            raise ValueError("repair stage must precede confirmation and freeze")
        if not isinstance(repair["reason"], str) or not repair["reason"].strip():
            raise ValueError("repair reason must reference the calibration/convergence evidence")
        kinds.add(kind)
    for group, field, normal, repaired, kind in (
        ("prototype", "max_updates", 50, 100, "prototype_iterations"),
        ("linear_logistic", "max_iter", 2000, 4000, "solver_iterations"),
        ("rbf_svm", "max_iter", 100000, 200000, "solver_iterations"),
    ):
        value = settings[group][field]
        if type(value) is not int or value not in (normal, repaired):
            raise ValueError(f"{group}.{field} has an unauthorized iteration limit")
        if value == repaired and kind not in kinds:
            raise ValueError(f"{group}.{field} increase requires an explicit pre-freeze repair record")


def _validate_variable_settings(settings):
    for key, length in (("model_revision", 40), ("environment_hash", 64)):
        value = settings[key]
        if value is not None and (
            not isinstance(value, str) or re.fullmatch(f"[0-9a-f]{{{length}}}", value) is None
        ):
            raise ValueError(f"{key} must be null or an immutable {length}-character lowercase hex identity")
    paths = settings["paths"]
    if not isinstance(paths, dict) or set(paths) != {"cache", "data", "results"}:
        raise ValueError("paths requires cache, data, and results")
    for key, root in (("cache", ".cache"), ("data", "data"), ("results", "results")):
        value = paths[key]
        if not isinstance(value, str):
            raise ValueError(f"paths.{key} must be a local relative path under {root}")
        path = Path(value)
        if path.is_absolute() or not path.parts or ".." in path.parts or path.parts[0] != root:
            raise ValueError(f"paths.{key} must be a local relative path under {root}")
    for group, field, maximum in (
        ("encoder", "batch_size", 32), ("numerics", "score_chunk_size", 512),
        (None, "bootstrap_batch_size", 200),
    ):
        value = settings[group][field] if group else settings[field]
        if type(value) is not int or not 1 <= value <= maximum:
            raise ValueError(f"{group + '.' if group else ''}{field} must be an integer from 1 to {maximum}")
    threads = settings["timing"]["blas_threads"]
    if type(threads) is not int or threads not in (1, 4):
        raise ValueError("timing.blas_threads must be 1 or 4")
    _validate_repairs(settings)


def _freeze(value):
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(child) for key, child in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(child) for child in value)
    return value


def load_config(path: Path) -> StudyConfig:
    """Load the supported, fixed study schema without acquiring any resources.

    Stage code must additionally enforce admission, recorded calibration choices,
    immutable resolved acquisition identities, and freeze/outcome prerequisites.
    Paths are relative to the repository working directory, as are CLI commands.
    """
    with Path(path).open(encoding="utf-8") as handle:
        settings = json.load(handle, parse_constant=_reject_constant, object_pairs_hook=_unique_keys)
    _check_finite(settings)
    _validate_fixed(settings, _FIXED_SETTINGS)
    _validate_variable_settings(settings)
    values = {key: _freeze(value) for key, value in settings.items()}
    values["paths"] = MappingProxyType({key: Path(value) for key, value in settings["paths"].items()})
    return StudyConfig(**values)
