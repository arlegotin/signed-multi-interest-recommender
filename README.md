# Signed multi-interest recommendation study

This package implements the bounded CPU study in [the blueprint](docs/blueprint.md).
All ordered stage commands and saved-evidence reports are implemented. Synthetic
test results do not imply that a real experiment has occurred.

Recreate the native Python 3.11/3.12 environment from the repository root, using
the platform/Python in saved environment evidence and the exact package pins:

```sh
/opt/homebrew/bin/python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install --no-cache-dir -r requirements.lock.txt
python -m pip install --no-deps -e '.[test]'
python -m signed_study --help
```

Help loads no encoder or outcomes. Keep caches under `.cache/`, including a local
Hugging Face cache (`HF_HOME="$PWD/.cache/huggingface"`). Acquisition validates the
fixed model's immutable revision/local files/license, records native package
pins and writes `results/resolved_config.json`. Initial null revision/environment
fields in `config/study.json` are acquisition outputs, not guessed identities.
Preserve the original invocation config once its receipts are bound; do not
copy resolved fields back into it. Stages recover the acquisition config, then
`results/calibration_config.json` for measured policy choices.

Execute in order, from the root with `.venv` active:

```sh
python -m signed_study inspect --config config/study.json
python -m signed_study acquire --config config/study.json
python -m signed_study audit --config config/study.json
```

Inspection records hardware, RAM, power, process-tree RSS, disk, packages and
blockers. Non-target inspection remains useful evidence but does not open heavy
target stages. Acquisition uses MovieLens 1M and the fixed MiniLM encoder. The
100K fallback is permitted only for verified 1M access failure under the
blueprint's access-only rule, with explicit separate scope; disappointing
results, incompatible encoder or inadequate resources do not authorize it.

Before calibration, record **current full-suite** G1 evidence:

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
python -m signed_study calibrate --config config/study.json
python -m signed_study smoke --config config/study.json --development-users 8
python -m signed_study develop --config config/study.json
python -m signed_study freeze --config config/study.json
python -m signed_study confirm --lock results/protocol.lock.json
python -m signed_study diagnose --lock results/protocol.lock.json
python -m signed_study report --results results
```

Every command owns its ledger accounting; never nest it in `Ledger.run_command`.
For independent development/test commands use that wrapper once with stage G1.
`record_correctness` owns its full pytest command and ledger evidence, binding
source/test bytes, config, environment and package lock. Record it again after
bound inputs change; an older successful pytest entry cannot open G2.
Bootstrap accounting was already imported once; never import it again or reset
clocks. Effective stage allocations/reallocations come from the permanent ledger,
including failures, not config defaults. The 240 experimental minutes, 45 setup
minutes and protected 30 reporting minutes remain separate.

Stages return status, hashed artifacts, attempt IDs and stop reasons. Unmet
prerequisites exit nonzero. Matching completed work is verified and reused;
missing/corrupt cache or receipts fail closed. Calibration records forecasts,
complete-cache validation, policy and cohort IDs. Development selects global
settings from bounded grids (78 initial, at most 14 additions). Durable journals
retain every started/failed evaluation. Reservations are distinct from actual
executed user/configuration attempts. Failed methods are not silently retried.
Global pre-freeze repairs require recorded evidence and obey the two-cycle,
20-minute repair cap within total time. Freeze requires clean committed source,
current G1 proof and complete prerequisite provenance.

Confirmation opens outcomes only after verifying the immutable canonical lock,
sidecar, ledger anchor and current identities. Interrupted work retains exposure
and failures; terminal partial results are not silently rerun. A verified
implementation-only post-exposure correction is exploratory on the same users:

```sh
python -m signed_study confirm --lock results/protocol.lock.json \
  --correction-reason 'Concrete implementation defect and correction'
```

This invalidates confirmatory interpretation, requires current corrected
implementation evidence, and consumes the authorized repair admission when run.
Diagnostics use frozen selections, saved original geometry and bounded
secondary/unadjusted analyses. Optional contamination can be skipped under its
predeclared applicability/resource rule. Numerical float32 checks do not replace
canonical float64 accuracy inference. Encoder inference is CPU float32;
geometry, fitting, distances and metrics are float64. Synthetic tests use no
study downloads or acquired labels.

Reporting reads saved evidence only: no acquisition, encoding, fitting, scoring,
confirmation resume, bootstrap inference or outcome-array access.
`report --results results` works for early stops and non-target evidence. It
writes `report.md`, `costs.csv`, observed `comparisons.csv`, applicable paired
figures and `artifact-inventory.json`. Its ten sections distinguish primary B*,
complete-suite and mechanism claims, retaining missing/invalid baselines,
analysis-pending work and failure costs. Rerun that command to regenerate.
The library API `signed_study.report.render_report(Path('results'))` admits no
work; the CLI owns G6. Equal saved evidence yields equal scientific tables and
figures. Repeated CLI admissions extend the ledger, so cost snapshots carry an
as-of ledger identity and need not remain byte-identical.

Historical report validation checks the immutable freeze/publication references,
journal hash chain/run identity and all current ledger statuses without requiring
the original installed model, cache or current source HEAD. It issues no
outcome-access permit. New confirmation/diagnostic work still requires strict
current verification. Preserve the ledger, lock/sidecar, exposure journal/marker
and published artifacts at their bound paths; hashes do not authorize relocation
or edits. Retain source commit/config/environment pins for local reproduction.
Prior-art links are inherited from blueprint planning evidence, not freshly
reviewed literature.

Raw MovieLens archives, derived text, outcomes, vectors, per-item scores and local
runs stay in ignored `data/`, `.cache/` and `results/`. Regenerate them locally
through the ordered commands under original provenance, rather than copying
them into an export. Public reporting exports contain aggregates and inventory
hashes only. Do not redistribute MovieLens raw/derived content; follow its bundled
usage license and acknowledgment terms. Encoder acquisition retains Apache 2.0
license evidence; respect model/dependency licenses. Observed-rating AUC does not
establish online satisfaction, missing-is-dislike, universal embedding failure
or a first-ever contribution.
