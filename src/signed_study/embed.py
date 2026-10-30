"""Frozen local CPU text encoding and a shared, provenance-checked vector cache.

Acquisition and inference are explicit, budgeted caller operations. Importing this
module loads neither torch nor Hugging Face. No helper admits study stages or
changes StudyConfig; G0 must persist resolve_encoder's result and hand its pinned
revision to a newly constructed config before G2 calls the probe/full encoder.
"""

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import re
import shutil
import tempfile

import numpy as np

from .data import Movie
from .protocol import StudyConfig, canonical_sha256


_TEMPLATE = "Movie: {title}. Genres: {genres}."
_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_STANDARD_FILES = {
    "modules.json", "config.json", "config_sentence_transformers.json",
    "sentence_bert_config.json", "tokenizer.json", "tokenizer_config.json",
    "special_tokens_map.json", "vocab.txt", "1_Pooling/config.json", "README.md",
}
_REQUIRED_FILES = {"modules.json", "config.json", "tokenizer_config.json", "1_Pooling/config.json", "README.md"}
_WEIGHTS = {"model.safetensors", "pytorch_model.bin"}
_ARCHITECTURES = {
    ("", "sentence_transformers.models.Transformer"),
    ("1_Pooling", "sentence_transformers.models.Pooling"),
    ("2_Normalize", "sentence_transformers.models.Normalize"),
}
_INPUT_FIELDS = {
    "model_id", "model_revision", "model_files", "model_license", "encoder_provenance_sha256",
    "dataset_manifest", "metadata_sha256", "template", "item_ids", "shape", "dtype",
    "encoder", "runtime_versions", "normalization",
}


def _json_copy(value):
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False, ensure_ascii=False))


def _sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _cache_root(config):
    root = config.paths["cache"].resolve()
    if not root.is_relative_to(Path.cwd().resolve()):
        raise ValueError("Encoder cache must be project-local")
    return root


def _pinned_revision(revision):
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Encoder revision must be a resolved immutable 40-character commit")


def format_movie(title: str, genres: list[str]) -> str:
    """The fixed title/lexicographically sorted genre template, without prompts."""
    if not isinstance(title, str) or not title or any(not isinstance(g, str) or not g for g in genres):
        raise ValueError("Movie title and genres must be nonempty strings")
    return _TEMPLATE.format(title=title, genres=", ".join(sorted(genres)))


def _verify_model(config, provenance, *, require_config_revision=True):
    required = {"schema_version", "model_id", "model_revision", "license", "model_path", "files", "weight_file", "input_sha256", "output_sha256"}
    if not isinstance(provenance, dict) or set(provenance) != required or provenance["schema_version"] != 1:
        raise ValueError("Invalid encoder acquisition provenance")
    _pinned_revision(provenance["model_revision"])
    if config.model_id != _MODEL or provenance["model_id"] != config.model_id:
        raise ValueError("Encoder model identity mismatch")
    if provenance["input_sha256"] != canonical_sha256({"model_id": config.model_id, "model_revision": provenance["model_revision"]}):
        raise ValueError("Encoder acquisition input provenance digest mismatch")
    if require_config_revision or config.model_revision is not None:
        _pinned_revision(config.model_revision)
        if provenance["model_revision"] != config.model_revision:
            raise ValueError("Encoder revision provenance mismatch")
    if not isinstance(provenance["license"], str) or not provenance["license"]:
        raise ValueError("Encoder license evidence is required")
    digest = canonical_sha256({k: v for k, v in provenance.items() if k not in {"model_path", "output_sha256"}})
    if digest != provenance["output_sha256"]:
        raise ValueError("Encoder acquisition provenance digest mismatch")
    files = provenance["files"]
    weight = provenance["weight_file"]
    if (not isinstance(files, dict) or not _REQUIRED_FILES <= files.keys()
            or weight not in _WEIGHTS or set(files) & _WEIGHTS != {weight}
            or set(files) - (_STANDARD_FILES | _WEIGHTS)
            or not ({"tokenizer.json", "vocab.txt"} & files.keys())):
        raise ValueError("Incomplete or unsupported encoder model files")
    root = _cache_root(config)
    model = Path(provenance["model_path"]).resolve()
    if not model.is_relative_to(root):
        raise ValueError("Encoder model files must be project-local")
    actual_files = {path.relative_to(model).as_posix() for path in model.rglob("*") if path.is_file()}
    if actual_files != set(files):
        raise ValueError("Encoder model files differ from the complete acquisition manifest")
    for name, expected in files.items():
        path = model / name
        if not path.resolve().is_relative_to(root) or not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"Encoder model file hash mismatch: {name}")
    modules = json.loads((model / "modules.json").read_text())
    architecture = [(module.get("path"), module.get("type")) for module in modules]
    if (len(architecture) not in (2, 3) or len(set(architecture)) != len(architecture)
            or not set(architecture) <= _ARCHITECTURES
            or not _ARCHITECTURES.difference({("2_Normalize", "sentence_transformers.models.Normalize")}) <= set(architecture)
            or architecture[:2] != [("", "sentence_transformers.models.Transformer"),
                                    ("1_Pooling", "sentence_transformers.models.Pooling")]):
        raise ValueError("Unsupported encoder architecture; remote custom code is disabled")
    return model


def resolve_encoder(config: StudyConfig, *, api=None, download=None) -> dict:
    """Acquire only standard model/tokenizer files and one weight format.

    The caller owns G0 setup admission and persists this returned JSON record.
    The revision is validated before returning; use dataclasses.replace(config,
    model_revision=result['model_revision']) explicitly, never mutate config.
    api.model_info and download are injectable network boundaries for tiny tests.
    """
    if config.model_id != _MODEL:
        raise ValueError("Unsupported encoder model")
    if config.model_revision is not None:
        _pinned_revision(config.model_revision)
    if api is None or download is None:
        from huggingface_hub import HfApi, snapshot_download
        api = HfApi() if api is None else api
        download = snapshot_download if download is None else download
    info = api.model_info(config.model_id, revision=config.model_revision, token=False, timeout=30)
    _pinned_revision(info.sha)
    if config.model_revision is not None and config.model_revision != info.sha:
        raise ValueError("Resolved encoder revision differs from requested revision")
    names = {sibling.rfilename for sibling in info.siblings}
    weight = "model.safetensors" if "model.safetensors" in names else "pytorch_model.bin"
    selected = (names & _STANDARD_FILES) | ({weight} & names)
    if (not _REQUIRED_FILES <= selected or weight not in selected
            or not ({"tokenizer.json", "vocab.txt"} & selected)):
        raise ValueError("Incomplete standard encoder architecture/tokenizer files")
    card = info.card_data
    license_name = card.get("license") if card is not None else None
    if not isinstance(license_name, str) or not license_name:
        raise ValueError("Encoder license evidence is required")
    root = _cache_root(config)
    model = Path(download(repo_id=config.model_id, revision=info.sha, token=False,
                          cache_dir=str(root / "huggingface"), allow_patterns=sorted(selected),
                          max_workers=1)).resolve()
    if not model.is_relative_to(root):
        raise ValueError("Downloaded encoder cache must be project-local")
    provenance = {"schema_version": 1, "model_id": config.model_id, "model_revision": info.sha,
                  "license": license_name, "model_path": str(model), "weight_file": weight,
                  "files": {name: _sha256(model / name) for name in sorted(selected)}}
    provenance["input_sha256"] = canonical_sha256({"model_id": config.model_id, "model_revision": info.sha})
    provenance["output_sha256"] = canonical_sha256({k: v for k, v in provenance.items() if k != "model_path"})
    _verify_model(config, provenance, require_config_revision=False)
    return provenance


def _metadata_rows(metadata):
    if not isinstance(metadata, Mapping) or not metadata or any(type(i) is not int for i in metadata):
        raise ValueError("Metadata requires a nonempty integer-ID mapping")
    rows = []
    for item_id in sorted(metadata):
        movie = metadata[item_id]
        if not isinstance(movie, Movie) or type(movie.item_id) is not int or movie.item_id != item_id:
            raise ValueError("Metadata Movie IDs must match their keys")
        rows.append({"item_id": item_id, "title": movie.title, "genres": sorted(movie.genres)})
        format_movie(movie.title, movie.genres)
    return rows


def expected_embedding_manifest(metadata, config, *, encoder_provenance, dataset_manifest) -> dict:
    """Build all required cache inputs without loading an encoder or encoding.

    Validates pinned configuration, live model-file hashes, and acquisition
    provenance. Call with the exact Movie mapping and audited dataset manifest.
    Scoring policies, selected parameters, outcomes and protocol-lock changes are
    deliberately absent: they do not change this shared representation.
    """
    _verify_model(config, encoder_provenance)
    if config.text_template != _TEMPLATE:
        raise ValueError("Unsupported encoder template")
    if (not isinstance(dataset_manifest, dict) or dataset_manifest.get("schema_version") != 1
            or dataset_manifest.get("dataset") not in {"ml-1m", "ml-100k"}
            or not re.fullmatch(r"[0-9a-f]{64}", dataset_manifest.get("archive_sha256", ""))):
        raise ValueError("Audited dataset acquisition provenance is required")
    settings = dict(config.encoder)
    if (settings["device"] != "cpu" or settings["inference_dtype"] != "float32"
            or settings["storage_dtype"] != "float32" or settings["trust_remote_code"] is not False
            or settings["max_sequence_length"] != 256 or settings["embedding_dimension"] != 384
            or type(settings["batch_size"]) is not int or settings["batch_size"] <= 0):
        raise ValueError("Unsupported fixed encoder settings")
    rows = _metadata_rows(metadata)
    inputs = {"model_id": config.model_id, "model_revision": config.model_revision,
              "model_files": dict(encoder_provenance["files"]), "model_license": encoder_provenance["license"],
              "encoder_provenance_sha256": encoder_provenance["output_sha256"],
              "dataset_manifest": _json_copy(dataset_manifest), "metadata_sha256": canonical_sha256({"movies": rows}),
              "template": config.text_template, "item_ids": [row["item_id"] for row in rows],
              "shape": [len(rows), 384], "dtype": "float32", "encoder": settings,
              "runtime_versions": {name: version(name) for name in (
                  "numpy", "torch", "sentence-transformers", "transformers", "huggingface-hub", "tokenizers", "safetensors")},
              "normalization": {"dtype": "float64", "norm_epsilon": config.numerics["norm_epsilon"]}}
    return {"schema_version": 1, "inputs": inputs, "input_sha256": canonical_sha256(inputs)}


def _normalized(vectors):
    value = np.array(vectors, dtype=np.float64, order="C", copy=True)
    if value.ndim != 2 or value.shape[1] != 384 or not np.all(np.isfinite(value)):
        raise ValueError("Embedding vectors must be finite 384-dimensional rows")
    norms = np.linalg.norm(value, axis=1)
    if np.any(norms <= 1e-12) or not np.all(np.isfinite(norms)):
        raise ValueError("Embedding vectors must have nonzero finite norms")
    value /= norms[:, None]
    return value


@dataclass(frozen=True)
class EmbeddingCache:
    """Sorted integer item IDs and read-only unit float64 geometry rows."""

    item_ids: np.ndarray
    vectors: np.ndarray
    manifest: dict

    def __post_init__(self):
        ids = np.asarray(self.item_ids)
        if (ids.ndim != 1 or ids.dtype.kind not in "iu" or len(ids) == 0
                or np.any(ids[1:] <= ids[:-1])):
            raise ValueError("Embedding item IDs must be unique sorted integers")
        vectors = _normalized(self.vectors)
        if len(ids) != len(vectors):
            raise ValueError("Embedding vector and item-ID shape mismatch")
        # Backing with immutable bytes also prevents setflags(write=True).
        object.__setattr__(self, "item_ids", np.frombuffer(ids.astype(np.int64).tobytes(), dtype=np.int64))
        object.__setattr__(self, "vectors", np.frombuffer(vectors.tobytes(), dtype=np.float64).reshape(vectors.shape))
        object.__setattr__(self, "manifest", _json_copy(self.manifest))

    def rows(self, item_ids) -> np.ndarray:
        requested = list(item_ids)
        if any(type(item) is not int and not isinstance(item, np.integer) for item in requested):
            raise ValueError("Embedding row lookup requires integer IDs")
        index = {int(item): row for row, item in enumerate(self.item_ids)}
        return self.vectors[[index[int(item)] for item in requested]]


class _LocalEncoder:
    def __init__(self, config, provenance):
        import torch
        from sentence_transformers import SentenceTransformer

        self.torch = torch
        model = _verify_model(config, provenance)
        self.model = SentenceTransformer(str(model), device="cpu", cache_folder=str(_cache_root(config) / "sentence_transformers"),
                                         local_files_only=True, trust_remote_code=False, token=False, backend="torch",
                                         model_kwargs={"dtype": torch.float32},
                                         config_kwargs={"trust_remote_code": False},
                                         processor_kwargs={"trust_remote_code": False})
        self.model.to(device="cpu", dtype=torch.float32)
        self.model.max_seq_length = 256
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        if self.model.get_sentence_embedding_dimension() != 384:
            raise ValueError("Encoder must produce 384-dimensional vectors")

    def token_lengths(self, texts):
        result = self.model.tokenizer(texts, add_special_tokens=True, truncation=False, padding=False)
        return [len(ids) for ids in result["input_ids"]]

    def encode(self, texts, *, batch_size):
        with self.torch.no_grad():
            return self.model.encode(texts, batch_size=batch_size, show_progress_bar=False,
                                     precision="float32", convert_to_numpy=True, device="cpu",
                                     normalize_embeddings=False, prompt="")


def load_encoder(config, encoder_provenance):
    """Load the verified local model for reuse across caller-timed G2 probes.

    Construction is lazy and performs no downloads. The caller owns admission,
    records model-load time, and keeps this object only within the admitted stage.
    """
    return _LocalEncoder(config, encoder_provenance)


def _encode(metadata, config, expected, encoder_provenance, encoder, checkpoint):
    encoder = encoder if encoder is not None else load_encoder(config, encoder_provenance)
    item_ids = expected["inputs"]["item_ids"]
    texts = [format_movie(metadata[item].title, metadata[item].genres) for item in item_ids]
    size = config.encoder["batch_size"]
    arrays, lengths = [], []
    for start in range(0, len(texts), size):
        if checkpoint:
            checkpoint()
        batch = texts[start:start + size]
        tokens = encoder.token_lengths(batch)
        if len(tokens) != len(batch) or any(type(n) is not int or n <= 0 for n in tokens):
            raise ValueError("Encoder must account for actual tokenizer lengths")
        lengths.extend(tokens)
        array = np.asarray(encoder.encode(batch, batch_size=size))
        if array.dtype != np.float32 or array.shape != (len(batch), 384):
            raise ValueError("Encoder vectors must be float32 with shape (items, 384)")
        _normalized(array)
        arrays.append(array)
        if checkpoint:
            checkpoint()
    raw = np.ascontiguousarray(np.concatenate(arrays), dtype=np.float32)
    accounting = {"items": len(item_ids), "tokens_before_truncation": sum(lengths),
                  "tokens_after_truncation": sum(min(n, 256) for n in lengths),
                  "truncated_items": sum(n > 256 for n in lengths),
                  "discarded_tokens": sum(max(n - 256, 0) for n in lengths), "maximum_tokens": max(lengths)}
    return raw, accounting


def check_repeatability(first, second, tolerance=1e-5) -> dict:
    """Check maximum absolute difference on aligned repeated float64 unit rows.

    Alignment/subset selection belongs to the caller. Never renormalizes the
    supplied repeats, which could hide a broken repeat. Raises above tolerance.
    """
    first, second = np.asarray(first, dtype=np.float64), np.asarray(second, dtype=np.float64)
    if (first.shape != second.shape or first.ndim != 2 or not first.size
            or not np.all(np.isfinite(first)) or not np.all(np.isfinite(second))
            or tolerance != 1e-5):
        raise ValueError("Invalid repeat encoding comparison")
    difference = float(np.max(np.abs(first - second)))
    accepted = difference <= tolerance
    if not accepted:
        raise ValueError(f"repeat encoding difference {difference} exceeds {tolerance}")
    return {"accepted": True, "maximum_absolute_difference": difference, "tolerance": tolerance,
            "items": len(first)}


def encode_probe(metadata, config, *, encoder_provenance, dataset_manifest, item_ids,
                 encoder=None, checkpoint=None) -> EmbeddingCache:
    """Encode a caller-selected subset without writing/reusing the full cache.

    G2 selects its 128 probe/32 repeat IDs and times each call under Ledger.work.
    Passing the same encoder object excludes model loading from a repeat timing.
    ID order in the returned cache is sorted; use rows(ids) for explicit alignment.
    """
    ids = list(item_ids)
    if not ids or any(type(i) is not int for i in ids) or len(set(ids)) != len(ids):
        raise ValueError("Probe requires unique integer item IDs")
    subset = {item: metadata[item] for item in ids}
    expected = expected_embedding_manifest(subset, config, encoder_provenance=encoder_provenance, dataset_manifest=dataset_manifest)
    raw, accounting = _encode(subset, config, expected, encoder_provenance, encoder, checkpoint)
    manifest = {**expected, "accounting": accounting, "probe": True,
                "array_sha256": hashlib.sha256(raw.tobytes()).hexdigest()}
    manifest["output_sha256"] = canonical_sha256(manifest)
    return EmbeddingCache(np.array(expected["inputs"]["item_ids"], dtype=np.int64), raw, manifest)


def load_embedding_cache(path, expected_manifest: dict) -> EmbeddingCache:
    """Load only a fully committed, exact-provenance cache; stale/partial fails.

    expected_manifest is expected_embedding_manifest(...) or a complete previously
    saved manifest. Every declared expected field must match. Artifact hashes
    always verify the actual .npy bytes, even when output hashes were not expected.
    """
    path = Path(path)
    try:
        manifest = json.loads((path / "manifest.json").read_text())
        if (expected_manifest.get("schema_version") != 1 or set(expected_manifest.get("inputs", {})) != _INPUT_FIELDS
                or expected_manifest.get("input_sha256") != canonical_sha256(expected_manifest["inputs"])
                or manifest.get("schema_version") != 1
                or manifest.get("input_sha256") != canonical_sha256(manifest["inputs"])
                or any(manifest.get(key) != value for key, value in expected_manifest.items())):
            raise ValueError("Embedding cache provenance mismatch")
        if set(manifest) != {"schema_version", "inputs", "input_sha256", "artifacts", "accounting", "output_sha256"}:
            raise ValueError("Incomplete embedding cache manifest")
        if manifest["output_sha256"] != canonical_sha256({k: v for k, v in manifest.items() if k != "output_sha256"}):
            raise ValueError("Embedding cache manifest digest mismatch")
        if set(manifest["artifacts"]) != {"vectors.npy", "item_ids.npy"}:
            raise ValueError("Incomplete embedding cache artifacts")
        for name, digest in manifest["artifacts"].items():
            if _sha256(path / name) != digest:
                raise ValueError(f"Embedding cache artifact hash mismatch: {name}")
        raw = np.load(path / "vectors.npy", allow_pickle=False)
        ids = np.load(path / "item_ids.npy", allow_pickle=False)
        inputs = manifest["inputs"]
        if (raw.dtype != np.dtype("float32") or list(raw.shape) != inputs["shape"]
                or inputs["dtype"] != "float32" or not raw.flags.c_contiguous
                or ids.dtype != np.dtype("int64") or ids.tolist() != inputs["item_ids"]):
            raise ValueError("Embedding cache array shape/dtype/item-ID provenance mismatch")
        return EmbeddingCache(ids, raw, manifest)
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError(f"Incomplete or invalid embedding cache: {path}") from error


def encode_items(metadata, config, *, encoder_provenance, dataset_manifest,
                 encoder=None, cache_path=None, checkpoint=None) -> EmbeddingCache:
    """Encode/reuse the shared sorted catalog after caller G2 admission.

    encoder injection supplies token_lengths(texts) and encode(texts, batch_size=)
    returning float32 384-column rows; production uses frozen local CPU inference.
    checkpoint is typically ledger.checkpoint. No downloads or configuration
    resolution happen here. An existing stale/partial final cache raises; callers
    must inspect it explicitly rather than silently overwrite provenance.
    """
    expected = expected_embedding_manifest(metadata, config, encoder_provenance=encoder_provenance, dataset_manifest=dataset_manifest)
    path = Path(cache_path) if cache_path is not None else _cache_root(config) / "embeddings"
    if not path.resolve().is_relative_to(_cache_root(config)):
        raise ValueError("Embedding cache must be project-local")
    if path.exists():
        return load_embedding_cache(path, expected)
    raw, accounting = _encode(metadata, config, expected, encoder_provenance, encoder, checkpoint)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{path.name}.tmp-", dir=path.parent))
    try:
        np.save(temporary / "vectors.npy", raw, allow_pickle=False)
        np.save(temporary / "item_ids.npy", np.array(expected["inputs"]["item_ids"], dtype=np.int64), allow_pickle=False)
        manifest = {**expected, "accounting": accounting,
                    "artifacts": {name: _sha256(temporary / name) for name in ("vectors.npy", "item_ids.npy")}}
        manifest["output_sha256"] = canonical_sha256(manifest)
        (temporary / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        temporary.rename(path)  # Only complete directories become visible to cache readers.
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return load_embedding_cache(path, expected)
