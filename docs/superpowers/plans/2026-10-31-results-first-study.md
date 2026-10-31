# Results-first Study Continuation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace elapsed-time stopping rules and deliver the pending native scientific comparison.

**Architecture:** Append an explicit execution-policy amendment without changing acquisition identities or erasing historical accounting. Adopt the verified saved calibration under current correctness evidence, then execute the remaining existing scientific pipeline.

**Tech Stack:** Existing native Python 3.12 virtual environment, NumPy, scikit-learn, torch, pytest, current Git branch.

**Spec:** `docs/superpowers/specs/2026-10-31-results-first-study-design.md`

## Global Constraints

- Work on the current `main` branch and make meaningful commits; no push or merge.
- Elapsed-time totals, stage allocations, forecasting safety multipliers and the reporting reserve are advisory, and cannot reject, shrink or stop the study.
- Preserve the existing permanent ledger, original resource hash, all historical events and counters; no resets or reallocation.
- Keep scientific numerical iteration/endpoint rules, memory and disk protection, one-worker/thread policy, the 100,000-evaluation bound, and all scientific gates.
- Use the full primary cohort (418 development / 1776 confirmation), determined before outcomes by the original digest order.
- No model, dataset, metric, threshold, seed, baseline or holdout changes follow from this amendment. No paid or remote computation is authorized.
- No confirmation outcomes before current correctness, development readiness, immutable freeze and verification.
- Preserve original saved calibration publications and early-stop evidence; do not duplicate the 1056 completed calibration evaluations.
- Use `.venv/bin/python`; never nest ledger-owned stage commands inside another admission.
- Implementers do not spawn subagents; the controller supplies independent review.

## Review Focus

- Existing ledgers must reopen with the exact original resource hash and history after policy adoption.
- Expired setup/stage/repair/global clocks must not block results-first execution; memory/disk/evaluation violations still block.
- A saved failed forecast may be adopted only after its original publication and completed numerical workload are verified; corruption fails closed.
- Confirmation/freeze/diagnostic consumers must use the adopted calibration artifact, and outcome isolation must remain strict.
- Post-freeze policy changes or source repairs must retain existing exposure/invalidation rules.

## File structure

`budget.py` owns durable execution policy and resource admission; `calibration.py`,
`development.py`, `protocol_lock.py` and `diagnostics.py` own their corresponding
time-dependent forecasts/applicability checks. `stages.py` owns saved-calibration
adoption and prerequisite paths. `__main__.py`/`run.py` expose commands. Reporting
must distinguish historical stop evidence from current completed results.
Tests belong beside the existing budget, calibration, stage, lock and report
tests. Blueprint and README document the new authoritative rule and resume recipe.

### Task 1: Implement and verify results-first policy and saved calibration adoption

**Files:** Modify `src/signed_study/{budget,calibration,development,stages,protocol_lock,diagnostics,report,__main__,run}.py` as required; `docs/blueprint.md`, `README.md`; add meaningful regressions in the relevant existing tests or a focused continuation test module.

**Interfaces:**
- Add `Ledger.adopt_results_first(reason: str) -> str`, returning the durable policy event ID; repeated adoption is idempotent. Snapshot exposes execution policy; legacy snapshots remain bounded by default.
- Add CLI `results-first --config PATH --reason TEXT` for explicit pre-freeze adoption.
- Add CLI `resume-calibration --config PATH` to verify/reuse saved measurements and publish the completed calibration stage without model work. Use a new calibration artifact path, retaining old path/hash bindings.
- Existing smoke/develop/freeze/confirm/diagnose/report interfaces remain usable with that stage receipt.

- [ ] Write and observe failing behavioral regressions for advisory time vs hard non-time resources, durable/idempotent policy and pre-freeze restriction, preserved history/hash/counters, saved-publication corruption and numerical rejection, path-independent calibration consumption, and full-cohort admission despite the historical forecast.
- [ ] Implement the minimal policy event and checks. Validate replayed policy events rather than silently trusting malformed records. Ordinary material repair cycles stay counted but the former quota is advisory in results-first mode; fixed scientific solver/prototype repair rules remain enforced.
- [ ] Update every direct time gate, including endpoint/selected/freeze forecasts and optional contamination, retaining finite nonnegative measurements and evaluation/resource/scientific checks. Keep bounded legacy behavior.
- [ ] Implement saved-calibration adoption. Verify the failed parent publication and workload against original ledger outputs, original scientific settings and numerical/cache checks; bind new inputs to current G1. Preserve old files and zero new evaluations.
- [ ] Rewrite blueprint time-ceiling and G2 stop rules and README resume instructions. Mark original timing numbers as historical estimates and preserve the first stop outcome as historical evidence.
- [ ] Run focused regressions with real accounting. Activate the authorized durable results-first policy on the native ledger, then run `record_correctness` once for the current full suite. Record exact results and all failures. Do not perform real calibration adoption or open confirmation outcomes yet.
- [ ] Commit the change and self-review; write the required report for independent controller review.

### Task 2: Resume the native study and deliver scientific results

**Files:** Existing ignored `results/`, `.cache/`, `data/`; update `docs/study-outcome.md`, `docs/study-decisions.md`, `README.md` with aggregate findings and provenance. Correct reporting code only if necessary, with tests/review and current proof before any exposure.

**Interfaces:** Consume Task 1's explicit policy, current correctness proof and `resume-calibration` command, then existing ordered stage commands under `config/study-pinned.json`.

- [ ] Preserve old report/aggregate exports and closing receipts under a versioned historical directory with a hash index; do not move any original publication still referenced by a receipt.
- [ ] Execute `resume-calibration`, verify zero duplicate evaluations and full-cohort selection, then smoke and development, including required endpoint coverage, numerical/precision checks and selected forecasts.
- [ ] Diagnose ordinary implementation defects with evidence and regression tests; log repairs, commit/review them and refresh current correctness before continuing. Do not turn a time forecast into another early stop.
- [ ] Verify readiness and clean committed source, then freeze once. Check cohort identities, metric/threshold/seeds, complete comparison schedule and immutable lock before confirmation.
- [ ] Execute confirmation, applicable diagnostics and the unchanged paired bootstrap analysis. Preserve every failure and missing-result bound; no retuning or fresh holdout after exposure.
- [ ] Render and verify the actual result tables/figures/report. Audit the historical overhead explanation using saved measurements; distinguish actual timings from forecasts and primary, complete-suite and mechanism claims.
- [ ] Update and commit the readable aggregate outcome/decision history/README; independently review the operational evidence and scientific claims, preserve the workflow audit trail, and leave a clean current branch.
