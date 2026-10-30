"""Local, synthetic protocol fixtures; never acquire study data or models."""

import json

import pytest


@pytest.fixture
def study_settings():
    """Blueprint settings, independent of the production configuration loader."""
    return {
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


@pytest.fixture
def study_config_path(tmp_path, study_settings):
    path = tmp_path / "study.json"
    path.write_text(json.dumps(study_settings, allow_nan=False), encoding="utf-8")
    return path


@pytest.fixture
def ninety_ratings():
    """Ninety records; all three classes contribute to the split boundary."""
    from signed_study.data import Rating

    return tuple(Rating(1, 10 + 3 * i, (5, 1, 3)[i % 3], 1000 + i) for i in range(90))


@pytest.fixture
def local_config(tmp_path, monkeypatch, study_config_path):
    from signed_study.protocol import load_config

    monkeypatch.chdir(tmp_path)
    return load_config(study_config_path)
