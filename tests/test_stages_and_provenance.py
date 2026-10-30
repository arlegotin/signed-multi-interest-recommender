"""Protocol contracts used by later stage/provenance tests."""

import json
from pathlib import Path

import pytest

from signed_study.protocol import StudyConfig, VerifiedProtocol, canonical_sha256, load_config


def test_fixed_protocol_and_canonical_hash(study_config_path):
    cfg = load_config(study_config_path)
    assert isinstance(cfg, StudyConfig)
    assert cfg.split_numerator == 7 and cfg.split_denominator == 10
    assert cfg.support_caps == (128, 64)
    assert cfg.delta == 0.01 and cfg.bootstrap_seed == 20261029
    assert cfg.bootstrap_resamples == 10000
    assert cfg.model_revision is None and cfg.environment_hash is None
    assert canonical_sha256({"a": 1, "b": 2}) == canonical_sha256({"b": 2, "a": 1})
    assert canonical_sha256({"a": 1, "b": 2}) == (
        "43258cff783fe7036d8a43033f830adfc60ec037382473548ac742b888292777"
    )


def write_settings(path, settings):
    path.write_text(json.dumps(settings), encoding="utf-8")


@pytest.mark.parametrize("caps", [[0, 64], [129, 64], [128, 65], [128], [128.0, 64], [True, 64]])
def test_rejects_invalid_caps(study_config_path, study_settings, caps):
    study_settings["support_caps"] = caps
    write_settings(study_config_path, study_settings)
    with pytest.raises(ValueError, match="support_caps"):
        load_config(study_config_path)


@pytest.mark.parametrize("field,value", [
    ("delta", 0.005), ("schema_version", 2), ("schema_version", True),
    ("split_numerator", 6), ("model_id", "another/model"),
    ("eligibility", [10, 5, 4, 5]), ("bootstrap_resamples", 9999),
])
def test_rejects_changed_fixed_protocol(study_config_path, study_settings, field, value):
    study_settings[field] = value
    write_settings(study_config_path, study_settings)
    with pytest.raises(ValueError, match=field):
        load_config(study_config_path)


@pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity", "1e9999"])
def test_rejects_nonfinite_json(study_config_path, literal):
    text = study_config_path.read_text(encoding="utf-8")
    study_config_path.write_text(text.replace('"delta": 0.01', f'"delta": {literal}'), encoding="utf-8")
    with pytest.raises(ValueError, match="finite"):
        load_config(study_config_path)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_canonical_hash_rejects_nonfinite_values(value):
    with pytest.raises(ValueError):
        canonical_sha256({"nested": [value]})


def test_rejects_duplicate_and_unknown_keys(study_config_path, study_settings):
    text = study_config_path.read_text(encoding="utf-8")
    study_config_path.write_text(text.replace('"schema_version": 1', '"schema_version": 1, "schema_version": 1'), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        load_config(study_config_path)
    study_settings["unrecognized"] = 1
    write_settings(study_config_path, study_settings)
    with pytest.raises(ValueError, match="unrecognized"):
        load_config(study_config_path)


def test_rejects_missing_settings(study_config_path, study_settings):
    del study_settings["bootstrap_seed"]
    write_settings(study_config_path, study_settings)
    with pytest.raises(ValueError, match="bootstrap_seed"):
        load_config(study_config_path)


@pytest.mark.parametrize("group,field,value", [
    ("encoder", "device", "mps"), ("encoder", "inference_dtype", "float16"),
    ("numerics", "dtype", "float32"), ("linear_logistic", "tol", 1e-4),
    ("resources", "evaluation_limit", 100001), ("prototype", "max_updates", 75),
])
def test_rejects_changed_nested_settings(study_config_path, study_settings, group, field, value):
    study_settings[group][field] = value
    write_settings(study_config_path, study_settings)
    with pytest.raises(ValueError, match=field):
        load_config(study_config_path)


@pytest.mark.parametrize("path", ["../outside", "/tmp/results", ".", "data/../results"])
def test_rejects_unsafe_output_paths(study_config_path, study_settings, path):
    study_settings["paths"]["results"] = path
    write_settings(study_config_path, study_settings)
    with pytest.raises(ValueError, match="paths.results"):
        load_config(study_config_path)


def test_resolved_acquisition_outputs_are_validated(study_config_path, study_settings):
    study_settings["model_revision"] = "a" * 40
    study_settings["environment_hash"] = "b" * 64
    write_settings(study_config_path, study_settings)
    cfg = load_config(study_config_path)
    assert cfg.model_revision == "a" * 40 and cfg.environment_hash == "b" * 64
    study_settings["model_revision"] = "main"
    write_settings(study_config_path, study_settings)
    with pytest.raises(ValueError, match="model_revision"):
        load_config(study_config_path)


@pytest.mark.parametrize("group,field,value,kind", [
    ("prototype", "max_updates", 100, "prototype_iterations"),
    ("linear_logistic", "max_iter", 4000, "solver_iterations"),
    ("rbf_svm", "max_iter", 200000, "solver_iterations"),
])
def test_authorized_iteration_repairs_require_a_record(study_config_path, study_settings, group, field, value, kind):
    study_settings[group][field] = value
    write_settings(study_config_path, study_settings)
    with pytest.raises(ValueError, match="repair"):
        load_config(study_config_path)
    study_settings["pre_freeze_repairs"] = [{"kind": kind, "stage": "G2", "reason": "Calibration evidence; budget admitted."}]
    write_settings(study_config_path, study_settings)
    assert getattr(load_config(study_config_path), group)[field] == value


def test_loaded_settings_cannot_be_mutated(study_config_path):
    cfg = load_config(study_config_path)
    with pytest.raises(TypeError):
        cfg.resources["evaluation_limit"] = 1
    with pytest.raises(TypeError):
        cfg.resources["stage_seconds"]["G1"] = 1


def test_repository_config_matches_blueprint_contract(study_settings):
    path = Path(__file__).resolve().parents[1] / "config" / "study.json"
    assert json.loads(path.read_text(encoding="utf-8")) == study_settings
    cfg = load_config(path)
    assert cfg.paths["results"] == Path("results")


def test_verified_protocol_record_preserves_lock_identity():
    protocol = VerifiedProtocol(lock_hash="a" * 64, payload={"schema_version": 1})
    assert protocol.lock_hash == "a" * 64 and protocol.payload == {"schema_version": 1}
