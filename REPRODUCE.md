# Reproduction and evidence

## Inspect this release

A fresh clone includes aggregate results, not the private native execution
bundle. Verify the public CSVs, coverage, paired comparisons and file digests:

```sh
python3 scripts/verify_public_results.py
```

This standard-library check needs Python 3.11+ and no downloads. It checks the
published evidence's integrity and internal consistency; it does not reproduce
model predictions, recompute confidence intervals or independently authenticate
private evidence from its hashes.

With the study dependencies installed, redraw the accuracy–cost figure from
public aggregates alone:

```sh
python scripts/plot_public_results.py
```

The plotted AUCs are descriptive means. The separate paired-comparison figure
shows intervals for differences, not uncertainty bars for individual AUCs.
See [the outcome](docs/study-outcome.md) for interpretation and scope.

## Install and test

The package supports Python 3.11–3.12. The recorded experiment used native
Apple arm64 Python 3.12.9. `requirements.lock.txt` pins that native environment;
it is not a guarantee that identical wheels exist on every platform.

For ordinary development, install in a new virtual environment:

```sh
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[test]'
python -m pytest -q
python -m signed_study --help
```

Tests use synthetic fixtures and do not acquire MovieLens or model assets.
The help command also performs no acquisition. The current release adds
packaging and public evidence checks; the historical study's 462-test proof
belongs to its frozen scientific source, not the current release metadata.

## Reproduce the scientific run

The experiment froze scientific source at commit
`5a0595d29869678c9b424ae0a9df42bfcee7f8f9`, before confirmation exposure.
Use that exact source, the native package pins, a clean workspace and newly
acquired inputs for an experimental rerun. The current release's packaging
metadata differs from that commit. A fresh run creates its own receipts and
protocol lock; its byte identities and measured timings need not match the
original run. Do not substitute a fresh lock for the original confirmation's
provenance.

The complete original ledger, per-user records, cache and receipts remain
local and are excluded from the public release. Commands below that resume
calibration or render saved native evidence require that original bundle;
they are not fresh-clone reproduction commands. For a new run, use `calibrate`
instead of `resume-calibration` and follow the stages in order. Review the
[dataset and model terms](THIRD_PARTY_NOTICES.md) before acquisition.
Never overwrite historical correctness or stage receipts to test a release.

## Native execution and saved-evidence runbook

The original G2 forecast stop remains preserved as historical evidence. On
31 October the user authorized results-first continuation: elapsed allocations,
forecasts and ordinary material-repair quotas became advisory. The scientific
question, thresholds, seeds, full cohorts and non-time constraints stayed fixed.
Saved calibration's 1056 evaluations were adopted without repeated fitting or
encoding. The completed campaign counted 62822 evaluations in total.

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
pins and writes `results/resolved_config.json`. Native execution uses
`config/study-pinned.json`, which pins the first observed immutable model
snapshot. Its nullable environment field is an acquisition output, not a
guessed identity. `config/study.json` remains the unresolved template.
Preserve the original invocation config once its receipts are bound; do not
copy resolved fields back into it. Stages recover the acquisition config, then
`results/calibration_config.json` for measured policy choices.

Use the report recipe below for completed native evidence. The ordered stage
recipe describes a fresh or unfinished approved run; experimental stages require
their exact frozen scientific source. Execute from the root with `.venv` active:

```sh
python -m signed_study inspect --config config/study-pinned.json
python -m signed_study acquire --config config/study-pinned.json
python -m signed_study audit --config config/study-pinned.json
```

Inspection records hardware, RAM, power, process-tree RSS, disk, packages and
blockers. Non-target inspection remains useful evidence but does not open heavy
target stages. Acquisition uses MovieLens 1M and the fixed MiniLM encoder. The
100K fallback is permitted only for verified 1M access failure under the
blueprint's access-only rule, with explicit separate scope; disappointing
results, incompatible encoder or inadequate resources do not authorize it.

Before calibration, record **current full-suite** G1 evidence.

For the authorized continuation, first adopt its durable policy, then run the
G1 recipe below. Adoption uses the existing ledger and cannot occur after freeze
or confirmation exposure. Existing ledgers remain bounded unless explicitly amended.

```sh
python -m signed_study results-first --config config/study-pinned.json \
  --reason '31 October user authorization: continue to scientific results; elapsed limits advisory'
```

```sh
python - <<'PY'
from pathlib import Path
from signed_study.protocol import load_config
from signed_study.budget import Ledger
from signed_study.stage_evidence import record_correctness
config = load_config(Path('config/study-pinned.json'))
ledger = Ledger(config.paths['results'] / 'budget_ledger.jsonl', config)
try:
    result = record_correctness(config, ledger)
finally:
    ledger.close()
raise SystemExit(0 if result['status'] == 'completed' else 2)
PY
python -m signed_study resume-calibration --config config/study-pinned.json
python -m signed_study smoke --config config/study-pinned.json --development-users 8
python -m signed_study develop --config config/study-pinned.json
python -m signed_study freeze --config config/study-pinned.json
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
clocks. Effective stage allocations/reallocations and execution policy come from
the permanent ledger, including failures. Original 240 experimental minutes,
45 setup minutes, stage allocations, 1.5 forecast multiplier and 30 reporting
minutes are historical planning estimates and advisory under `results_first`.
All elapsed usage and repair cycles remain counted. Evaluation, memory and disk
limits remain enforced, with no paid or remote computation.

`resume-calibration` verifies the original failed terminal publication, original
G1 and unchanged scientific implementation/settings, completed workload journal,
numerical checks and full cache. It writes `calibration-resumed-<id>.json` and a
new successful calibration stage bound to current G1, with zero new evaluations.
It preserves `results/calibration.json` at its original path and hash, the failed
attempt and the archived old correctness bytes. Downstream stages resolve the
calibration artifact from that completed stage. For a new study without saved
measurements, use `calibrate` in place of `resume-calibration`.

If an earlier bounded run completed calibration with a smaller cohort, ordinary
reuse becomes stale after results-first adoption. `resume-calibration` explicitly
requalifies unchanged scientific measurements and publishes the full audited
cohort without fitting again. It preserves the old calibration and policy files
and the old receipt at its original path. Re-run `smoke` and `develop` in order:
only this explicit supersession permits replacement evidence, saved to new
versioned receipt files. Hash-checked `stages/{calibrate,smoke,develop}.active.json`
references select the effective receipts; original receipts and outputs remain
readable. Freeze and reporting use those verified effective references. No
receipt can be superseded after freeze or confirmation exposure.

When calibration instead requires a permitted numerical repair or a scientific
implementation correction, record the repair, obtain fresh G1 and invoke
`calibrate` with the repaired configuration. A qualified correction performs
new, counted measurements in a versioned calibration publication; it preserves
the prior publication, probe, policy and failed attempt. Administrative policy
requalification never substitutes for required scientific remeasurement.

Stages return status, hashed artifacts, attempt IDs and stop reasons. Unmet
prerequisites exit nonzero. Matching completed work is verified and reused;
missing/corrupt cache or receipts fail closed. Calibration records forecasts,
complete-cache validation, policy and cohort IDs. Development selects global
settings from bounded grids (78 initial, at most 14 additions). Durable journals
retain every started/failed evaluation. Reservations are distinct from actual
executed user/configuration attempts. Failed methods are not silently retried.
Global pre-freeze repairs require recorded evidence and review; ordinary material
cycles and their elapsed costs are advisory under `results_first`. Scientific
solver/prototype iteration and repair rules remain fixed. Freeze requires clean committed source,
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
writes `report.md`, `costs.csv`, `actual-work.csv`, observed `comparisons.csv`, applicable paired
figures and `artifact-inventory.json`. Its ten sections distinguish primary B*,
complete-suite and mechanism claims, retaining missing/invalid baselines,
analysis-pending work and failure costs.
The library API `signed_study.report.render_report(Path('results'))` admits no
work; the CLI owns G6. Equal saved evidence yields equal scientific tables and
figures. Repeated CLI admissions extend the ledger, so cost snapshots carry an
as-of ledger identity and need not remain byte-identical.

Regenerate the active result from the existing local evidence, from the repository
root with the exact pinned environment:

```sh
.venv/bin/python -m signed_study report --results results
.venv/bin/python scripts/render_results_first_study.py --results results
```

The generic CLI writes the ten-section report, aggregate metrics/comparisons/
counts/costs, paired interval and difference figures, and artifact inventory.
The tracked saved-evidence packaging script verifies the bound native summary,
adds operational qualifications, regenerates the committed aggregate CSVs and
readable figure, and publishes a versioned reproduction receipt selected by
`results/results-first-reproduction.json`. It needs no workflow scratch files. The native
scientific exports were checked against the bound aggregate summary and then
regenerated to verify identical scientific CSV/PNG hashes. Reporting extends the
ledger, so cost snapshots carry their as-of identity and can change.

Preserve resolved configuration, ledger, effective stage receipts, acquisition,
audit, environment, calibration publications, immutable lock/sidecar and exposure
journal/marker at their bound paths. The native summary and operating-conditions
receipts are in `results/operator_logs/task2-scientific-summary.json` and
`task2-timing-operating-conditions.json`. The summary verifies exact complete
coverage and separates measured model work, broader pipeline overhead and
advisory forecasts. No model, encoding, new inference or outcome-array access
is part of reporting.

Timing used the fixed 3883-item catalog, 32 users, one warmup and five timed
repeats. The configured policy was one worker/one BLAS thread; detected libomp
had one thread. Accelerate's actual BLAS thread count was not independently
verified. The pre-timing power/CPU snapshot includes background contention;
performance claims stay within those observed conditions and catalog.

The historical G2-only `scripts/render_study_outcome.py` wrapper must not run
against the active completed study. Read the preserved original narrative and
exports under `results/history/2026-10-31-original-time-stop/`; its 20-file hash
index remains unchanged. The active generic renderer conservatively flags an
absolute/relative inspection-path mismatch. The tracked packaging script verifies
the same file and exact hash against its completed G0 publication, resolves the
qualification in the active report, and records it in the inventory; see the
outcome for the specific qualification. The calibration actual-work row retains
its original terminal forecast failure even though all 1056 measurements
completed and were qualified by the new successful adoption receipt.

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
