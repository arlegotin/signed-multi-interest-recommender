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
```

After acquisition/audit, record a current full correctness result before G2:

```sh
python - <<'PY'
from pathlib import Path
from signed_study.protocol import load_config
from signed_study.budget import Ledger
from signed_study.stage_evidence import record_correctness
config = load_config(Path('config/study.json'))
ledger = Ledger(config.paths['results'] / 'budget_ledger.jsonl', config)
try:
    result = record_correctness(config, ledger)
finally:
    ledger.close()
raise SystemExit(0 if result['status'] == 'completed' else 2)
PY
python -m signed_study calibrate --help
python -m signed_study calibrate --config config/study.json
python -m signed_study smoke --config config/study.json --development-users 8
python -m signed_study develop --config config/study.json
python -m signed_study freeze --config config/study.json
python -m signed_study confirm --lock results/protocol.lock.json
python -m signed_study diagnose --lock results/protocol.lock.json
python -m signed_study report --results results
```

`acquire`, `audit`, `calibrate`, and the eight-user `smoke` command are
implemented. Their results report `status`, hashed `artifacts`, `attempt_ids`,
and `stop_reason`. Completed stages reuse matching evidence and verify the
complete embedding cache before further development work. Unmet prerequisites
return a nonzero CLI status. Later commands remain discoverable placeholders.

Acquisition saves the immutable model revision in
`results/encoder_acquisition.json`, the explicit configuration in
`results/resolved_config.json`, and provisional native package pins in
`requirements.lock.txt`. The lock excludes the local `signed-study` distribution;
reconstruct it from this checkout with `pip install -e .`. The current G1 result
binds source/test bytes, configuration, environment evidence, and the package
lock. Record it again after any bound input changes; an older successful pytest
entry alone cannot open G2. `record_correctness` owns its accounting, so do not
wrap it or any implemented stage with another ledger command wrapper.

Calibration records probe/repeat/full-cache costs, development-only fitting
attempts and numerical checks, and the measured forecast in
`results/calibration.json`. `results/calibration_config.json` and the ordered
cohort IDs in its stage result supply the policy/schedule for later work.
`repair_required` preserves failed diagnostics and requires a recorded global
pre-freeze correction; it never retries a single method with different support
or tolerances. All actual fit/scoring attempts have durable local journals;
batch reservations remain distinct from executed evaluations.

Global parameters use development outcomes. Confirmation outcomes require a
verified immutable freeze. CPU encoder inference uses float32; geometry,
fitting, distances, and metrics use float64. Synthetic tests use injected
acquisition and encoder boundaries and perform no model/data downloads.

Keep the environment and Hugging Face caches under `.venv/` and `.cache/`.
Pass the local cache path explicitly when acquisition/encoding is implemented.
Raw MovieLens data, derived per-item content, outcomes, scores, and local run
artifacts belong under ignored `data/` and `results/`; do not redistribute
them. MovieLens use must follow its bundled license and acknowledgment terms.
