# Signed multi-interest recommendation study

This package implements the fixed CPU study in [the blueprint](docs/blueprint.md).
The current bootstrap provides validated protocol configuration and command help.
Stage execution will be added in subsequent tasks; help performs no downloads,
loads no encoder, and accesses no outcomes.

Run from the repository root using native Python 3.11 or 3.12:

```sh
/opt/homebrew/bin/python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install --no-cache-dir -e '.[test]'
python -m pytest -q
python -m signed_study --help
```

The direct dependencies resolve from public CPU-capable releases on the target
machine. Exact versions are recorded during setup; the validated environment is
locked after the G1 correctness gate. Initial `model_revision` and
`environment_hash` in [config/study.json](config/study.json) are explicit null
acquisition outputs. Resolve their immutable identities before G2; later stages
must require exact provenance matches on resume.

`load_config(Path("config/study.json"))` validates the scientific contract,
resource limits, local paths, and recorded permitted iteration repairs. Returned
settings are immutable. Only the prescribed global repair iteration limits are
accepted, with a `pre_freeze_repairs` record containing `kind`
(`prototype_iterations` or `solver_iterations`), `stage` (`G1`, `G2`, or `G3`),
and an evidence-bearing `reason`. The owning stage must enforce calibration,
pre-freeze, and cumulative repair-budget conditions. Calibration may reduce
encoder batches or scoring chunks and select one or four BLAS threads; the
owning stage records those choices. Bootstrap records setup/test command start
and completion, elapsed seconds, and project disk usage in ignored
`results/bootstrap_accounting.jsonl`; G1 validation time is retained for the
permanent resource ledger. Source-editing idle time is separate.

The intended stage commands and their prerequisites are discoverable through
`python -m signed_study --help` and each command's `--help`:

```sh
python -m signed_study inspect --config config/study.json
python -m signed_study acquire --config config/study.json
python -m signed_study audit --config config/study.json
python -m signed_study calibrate --config config/study.json
python -m signed_study smoke --config config/study.json --development-users 8
python -m signed_study develop --config config/study.json
python -m signed_study freeze --config config/study.json
python -m signed_study confirm --lock results/protocol.lock.json
python -m signed_study diagnose --lock results/protocol.lock.json
python -m signed_study report --results results
```

Global parameters use development outcomes. Confirmation outcomes require a
verified immutable freeze. CPU encoder inference uses float32; geometry,
fitting, distances, and metrics use float64. No experiments or model/data
downloads are part of this bootstrap task.

Keep the environment and Hugging Face caches under `.venv/` and `.cache/`.
Pass the local cache path explicitly when acquisition/encoding is implemented.
Raw MovieLens data, derived per-item content, outcomes, scores, and local run
artifacts belong under ignored `data/` and `results/`; do not redistribute
them. MovieLens use must follow its bundled license and acknowledgment terms.
