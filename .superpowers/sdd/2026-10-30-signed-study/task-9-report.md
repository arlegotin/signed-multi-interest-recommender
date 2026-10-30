# Task 9 implementation report

Status: DONE. Implements development tuning, the complete bounded endpoint
expansion, selected family/comparator/control inputs, refreshed downstream costs,
precision planning and history thresholds. No real data or model acquisition,
real tuning, confirmation scoring, outcome exposure or protocol freeze occurred.
All runtime validation below uses tiny synthetic saved episodes and vectors.

Commit subject: `feat: implement bounded development tuning and comparator selection`.

## Implementation and decisions

- `run.py` exposes `select_family(rows) -> ModelConfig`,
  `select_comparator(family_choices) -> ModelConfig`,
  `endpoint_requests(family_choices, config) -> list[ModelConfig]`, and
  `develop(config, ledger) -> dict`. Implementations belong to `metrics.py`,
  `models.py` and the focused `development.py`; `stages.py` remains the thin
  provenance/admission wrapper. CLI `develop` now dispatches the implemented gate.
- Within-family selection first finds the maximum valid complete-cohort macro
  AUC, then takes the smallest `(K, lambda, C, g, configuration ID)` among entries
  within 1e-6 of that maximum. Missing parameters contribute zero. It does not
  use a pairwise approximate comparator. B* uses actual selected AUC, exact ties
  and lexicographic required-family ID; SignedMax, controls and unknown families
  cannot enter that comparison.
- The seven declared families retain all 78 initial entries. Centroid,
  PositiveMax and RocchioDirection are selected views of the prescribed existing
  entries; no extra fitting campaign is introduced. Each declared equivalent
  entry still consumes its evaluation slot, even if its scores are equal.
- Endpoint requests follow blueprint 6.3 exactly: numeric `(C,g)` order for the
  complete RBF row/column including the intersection, then LinearLogistic,
  MeanNegative, AllExemplarMargin and SignedMax. Lower/upper outer values are
  `.001/1000` and `.0625/64`; lambda expands from 8 to 16 at the selected K.
  The maximum is 10 RBF plus four other entries, 14 additions and 92 total.
  Requests are computed once from initial choices; final selection cannot
  initiate another expansion.
- The controller ruled that new endpoint costs must be measured in a separately
  admitted, outcome-free pilot rather than extrapolated using solver iteration
  counts. The pilot profiles the **entire** eligible request list on the existing
  history-based calibration sampler restricted to fixed planned development IDs.
  Official cohorts have at least 62 users and supply the prescribed 32-user
  sample; tiny synthetic fixtures profile their entire smaller cohort. Initial
  family timings estimate admission of the bounded profiling work. Its actual
  parameter-specific timings price admission of the complete full-cohort
  expansion. Missing/failed timing coverage or admission failure records every
  requested ID as unavailable. Pilot attempts have `metrics: null`, load no
  labels, and remain ineligible for tuning. Once fully admitted, the complete
  expansion reruns all users explicitly, charging those attempts separately.
- The existing `workload.measure_users` loop now accepts explicit unique
  `declared_entries` and `timing_only`; its original defaults remain unchanged.
  It persists started/completed/failed/interrupted attempts, score arrays,
  diagnostics and metrics before selection. It saves actual positive-prototype
  fingerprints and measured diagnostics, and uses the resolved config for every
  fit. Labels are read once per user per admitted evaluation batch, never per
  configuration. The pilot does not read labels; confirmation labels are never
  loaded. No duplicate persistence/evaluation framework was added.
- Aggregation requires exactly one successful result for every prescribed user.
  Partial and failed configurations have no selectable macro AUC, and record
  failed shares. Missing required families stop development and retain a
  full-family/fixed-cohort repair-required decision. The controller confirmed
  no automatic repair/rerun API is required here: a later validated uniform
  pre-freeze correction may rerun the complete affected family with preserved
  attempts and repair accounting. This implementation never silently retries
  a solver or substitutes a simpler baseline.
- Initial and final boundary winners include adjacent interior values and AUC
  gains. Unavailable expansion coverage, any incomplete configuration, or a
  persistent expanded outer winner disqualifies a complete-suite tuning claim.
  A feasible qualified primary comparison can still be prepared without claiming
  a complete-suite result.
- Final history statistics use only capped histories and cached float64 rows:
  `H = 1 - ||mean(P)||²`, `O = mean_p max_n p·n`, negative-mean norm and negative
  count. Development medians of H and O are saved. Precision uses paired selected
  S*/B* development differences and the fixed confirmation cohort size; delta
  remains `.01`, with explicit modest-effect imprecision metadata.
- The concrete 13-slot comparison schedule is repriced after selection using
  actual canonical-chunk measurements and exact chosen solver parameters.
  Forecasts include confirmation, 32-user catalog timing with one warmup/five
  repetitions, and up to 64 users across the 11 geometric slots. Counts must
  stay within prior slot allowances and remaining stage/total/evaluation budgets,
  including the reporting reserve. New expensive endpoints are not priced as
  their old solver settings. This remains an empirical forecast; normal ledger
  admission and monitoring still apply to later execution.
- Planning, selection, history computations, forecasts and artifact writes have
  separate G3 admissions around the separately admitted fitting batches. Live
  attempt IDs are shared with the public stage wrapper, including interruptions.

## Artifact schemas and Task 10 freeze handoff

Completed public stages write `results/stages/develop.json` using the existing
schema-versioned, input/output-hashed stage record. It binds successful ledger
attempts and SHA256 file references for `development`, every batch summary and
every durable JSONL journal. Prerequisites are current verified G1, acquisition,
audit, calibration, verified complete cache and the eight-user smoke record.
Resume verifies these and saved artifacts without new fitting reservations;
changed or missing evidence fails closed. Failed required-family runs preserve
their unique development summary without creating a completed stage record.

The referenced `results/development-<uuid>.json` has schema version 1 and canonical
`input_sha256`/`output_sha256`. Its inputs bind the resolved config and calibration
identity. It contains:

- `development_user_ids`, `confirmation_user_ids`: the calibrated fixed IDs.
- `configurations`: only the initial and fully admitted expansion entries, each
  `{family, parameters, id}`. Parameter keys are `K`, `lambda`, `C`, `g` as
  applicable; IDs bind canonical family/parameter JSON.
- `attempts`, `pilot_attempts`, `batches`, `attempt_ids`, `actual_evaluations`:
  evaluation and profiling attempts are separate. Batch records contain kind,
  summary/journal paths, ledger ID and actual evaluation count. Batch summaries
  retain `user_measurements`, canonical chunk costs and diagnostics. Journals
  retain durable per-attempt hashes; pilot scores never supply tuning metrics.
- `configuration_results`: configuration record, completion/status, failed
  share, macro AUC or null, per-user AUCs and actual per-user prototype hashes.
  JSON object user-ID keys are serialized as strings.
- `families`: ten named selected family records. Each record is
  `{family, parameters, id, macro_auc, source_configuration_id, fit_family,
  prototype_identity, development_prototype_identities}`.
  For the three named views, `family`/`id` name the view;
  `source_configuration_id` refers to its actual declared source evaluation and
  `fit_family` supplies the executable adapter. Other entries reference themselves.
  `prototype_identity` hashes constructor kind, global policy and requested K;
  `development_prototype_identities` maps each user to a SHA256 fingerprint of
  the actual positive support IDs and fitted positive vectors.
- `S_star`, `B_star`: the same selected-record schema, with no confirmation data.
- `matched_controls`: keys `matched_MeanNegative`, `matched_PositiveMax`,
  `matched_K1_SignedMax`, each containing `{family, parameters, id, fit_family,
  prototype_identity, development_prototype_identities}`. Parameters respectively
  equal S*'s `(K,lambda)`, `(K,0)` and `(1,lambda)`. The first two share S*'s
  actual development prototype hashes; the K=1 control uses the declared K=1
  SignedMax prototype hashes. Named slots remain distinct even when parameter
  IDs coincide with another slot.
- `comparison_schedule`: the ten selected slots plus three matched slots,
  retaining `name`, `role`, `geometric`, executable `fit_family`, actual parameters,
  canonical IDs and the selected/control records above. Downstream code must fit
  by **fit_family and parameters**, not by alias name or an assumed default.
- `endpoint_coverage`: complete ordered requested records, admitted flag, all
  unavailable IDs/reason and expansion batch count. `initial_boundary_winners`,
  `boundary_winners`, `incomplete_configuration_ids`,
  `missing_required_families`, `repair_decisions`, `complete_suite_eligible` and
  explicit stop status explain the permitted interpretation.
- `planned_precision`, `history_statistics`, `history_thresholds`, and
  `selected_forecast`. The last includes concrete schedule, G4/G5 seconds and
  evaluations, timing/numerical user IDs, actual catalog cardinality, canonical
  chunk, safety multiplier, feasibility and stop reason.
- `confirmation_outcomes_accessed: false`.

Freeze must verify the completed stage and references, require a feasible selected
forecast, preserve coverage qualifications, use the resolved calibration policy,
bind the fixed episode/cache/source/prototype identities, and freeze these actual
selected/control parameters and diagnostic user schedules. Development metadata
does not authorize confirmation outcome access; verified lock creation and access
remain Task 10. No `protocol.lock.json` was created or modified here.

## TDD and verification evidence

All commands used `.venv/bin/python` from the repository root, launched by
`Ledger.run_command(stage='G1')` against `results/budget_ledger.jsonl`, with
authorized host resource sampling and project-local pytest basetemps. No bootstrap
ledger, clock reset, double wrapping, or stage reallocation was used.

1. Selection RED:
   `.venv/bin/python -m pytest tests/test_stages_and_provenance.py -q -k selection_ties --basetemp=.cache/task9-pytest`.
   **1 failed, 104 deselected in 1.63s**, missing `run.select_family` assertion.
   The first implementation check exposed an RBF recursive-request bug; it was
   corrected by restricting expansion to the original RBF grid. The test's second
   endpoint fixture also resets the geometric initial choices between cases.
2. Development RED:
   `.venv/bin/python -m pytest tests/test_development.py -q --basetemp=.cache/task9-pytest`.
   **3 failed in 1.81s**, missing development module/aggregation/execution and
   unexpected `declared_entries` argument. These demonstrate absent production
   features, not missing dependencies or tooling. An intermediate combined run
   was **1 failed, 3 passed** because a synthetic aggregation fixture supplied
   only AUC instead of the existing complete metric schema; the fixture now uses
   real `evaluate_user` output.
3. Initial development GREEN: the focused stage/development command was
   **7 passed, 104 deselected in 6.53s**, including actual fit/output persistence,
   complete-denial accounting, failed required-family stop and prerequisite gates.
4. Requested stage/metric suites:
   `.venv/bin/python -m pytest tests/test_development.py tests/test_stages_and_provenance.py tests/test_metrics_and_baselines.py -q --basetemp=.cache/task9-pytest`.
   **176 passed in 11.72s**, exit 0, pristine.
5. Strengthened focused GREEN was **10 passed, 104 deselected in 15.56s** using
   the two stage/development files and `-k 'development or selection_ties or
   declared_aliases or partial_configuration or expansion_denial or
   failed_required_family or public_develop or real_variable or endpoint_profile
   or stage_verifies'`. This covers independent maximum-anchored chain/tie order,
   all simplicity coordinates, exact between-family differences, both RBF outer
   corners, deduplication/order/no recursive expansion, actual varying-score
   selection, matched parameters/prototype hashes, no pilot labels, full denied
   coverage, profile failure, smoke, immutable resume and tamper rejection.
6. First full pre-commit check:
   `.venv/bin/python -m pytest -q --basetemp=.cache/task9-full-pytest`.
   **290 passed in 35.85s**, exit 0, pristine.
7. Subsequent self-review found the interruption attempt-reference gap. A new
   `test_develop_interruption_retains_live_attempt_references` RED command used
   `tests/test_development.py -q -k interruption_retains --basetemp=.cache/task9-pytest`:
   **1 failed, 9 deselected in 3.63s**; the failure identified two newly admitted
   development attempts absent from the public failure record. After sharing the
   live attempt list, identical GREEN was **1 passed, 9 deselected in 3.46s**.
8. Because that new finding changed orchestration after the first full check, a
   second complete pre-commit check was justified and run with the same full
   command: **291 passed in 40.53s**, exit 0, pristine. This is the final code state.

The synthetic integration tests use real model fits, saved development labels,
real score/metric calculations and complete selection. Failure injections affect
the failed adapter or admission boundary while successful fits and aggregation
remain real. No real-data or semantic-effect result is inferred from these tests.

## Permanent G1 accounting

| Attempt | Check | Status | Workload seconds |
|---|---|---|---:|
| 22dba12e-60dc-4ca2-ad51-aa4cc55a2248 | selection RED | failed | 2.571172 |
| 3856796a-7561-4a9d-8468-1f74c6699264 | recursive expansion check | failed | 1.736147 |
| 6ae27863-f2c2-4cf8-8a8f-22fe342fdd19 | development RED | failed | 2.732888 |
| 63a6c2f9-89d6-403a-80ce-891ac47598df | metric-fixture schema check | failed | 4.595148 |
| e9ff4dd4-79a0-43f1-bd8b-62ce037ba37a | seven focused checks | completed | 7.563935 |
| f51e96b7-ecfd-426e-90a4-b295a28d6989 | stage/metric suites | completed | 12.767126 |
| 4ab2db1b-e8f8-48ef-9f8e-90ae2190aa6b | variable-score/profile checks | completed | 9.972770 |
| 70d95c2d-bc72-4994-a1a5-d6aa45f07349 | public stage/resume | completed | 7.836412 |
| 261dd608-8c06-4b03-82af-50c300b8d878 | ten focused checks | completed | 16.519477 |
| dc45602f-d34b-4b2f-a7db-1d7e741b42c2 | first full suite | completed | 36.878099 |
| cf0afe2a-8bab-411d-91e8-d5d3947b1d5e | interruption RED | failed | 5.374539 |
| 598d72a1-1208-4ec2-8e84-c87637e8561f | interruption GREEN | completed | 4.548164 |
| 9ba622b8-b660-448a-a28a-c45061830f08 | final full suite | completed | 42.157360 |

These test workloads total **155.253238 seconds**; admission/resource-sampling
overhead is additionally retained in the permanent ledger. An accounted ledger
summary after the final suite reported cumulative G1 use **471.758618 seconds**
including earlier tasks and that bookkeeping command. Cumulative production
attempted evaluations remain **0**; synthetic tests own isolated fixture ledgers.
No reallocation or repair cycle was needed. The final full-suite attempt recorded
233,095,168 bytes peak process-tree RSS. Final diff checks and commit commands are
also permanently G1-accounted; their exact records remain in that same ledger.

## Files and self-review

Changed `run.py`, `models.py`, `metrics.py`, `workload.py`, `stages.py`,
`__main__.py`, and `tests/test_stages_and_provenance.py`; added responsibility-based
`development.py`, `tests/test_development.py` and this report. Existing tests remain.

Read the complete changed source and interfaces against the brief, endpoint rules,
Task 8 fix contracts and controller pilot ruling. Checked max-anchored tolerance,
required-family filtering, fixed cohort completeness, 14/92 ceilings, pilot
ineligibility and explicit rerun charging, immutable artifact references, matching
prototype identities, all thirteen named slots, concrete canonical-chunk costs,
precision/strata inputs, coverage qualification and interruption evidence. The
interruption issue found in self-review was fixed with the explicit RED/GREEN
regression and final whole-suite rerun above. No unresolved implementation concern.
Real endpoint costs, feasibility and scientific outcomes remain execution work.
