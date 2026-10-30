# Signed preference study: implementation specification

- Date: 2026-10-30.
- Status: Builder checked; awaiting design approval. Implementation has not started.
- Source: [POC execution brief](../../blueprint.md), committed as `33347fe`.
- Source SHA256: `dffa43017f2d1fe3518f30ff80c0ad28769f45376930f815128f04c305fee062`.

## 1. Intent and authority

Build a small, local, reproducible diagnostic of whether a nearest-negative
penalty improves a frozen-content multi-interest ranker. Success means reaching
an evidence-supported decision, including a valid negative result, inconclusive
result, or documented early stop. It does not require the proposed method to win.

This specification translates the blueprint into software contracts and
acceptance criteria. The blueprint remains authoritative for scientific
definitions, equations, literature, and execution limits. Section references
below point into that file. No protocol change or new literature claim is
introduced here. A conflict must be resolved explicitly before confirmation.

The present work produces this specification and an implementation plan only.
Installing dependencies, implementing the package, downloading study resources,
and running experiments belong to subsequent implementation work. None of the
proposed commands or target-machine measurements are represented as completed.

### Goals

- Give every method identical frozen item vectors and capped earlier feedback.
- Separate positive prototype count from negative aggregation using the required
  baseline families and matched controls.
- Keep confirmation outcomes out of global model and protocol selection.
- Make every gate, attempted run, resource decision, failure, and conclusion
  reconstructible from local artifacts.
- Deliver a standalone report proportional to the stages actually reached.

### Non-goals and failure conditions

Do not add representation learning, richer metadata, additional research domains,
unobserved negative sampling, collaborative learning, ANN retrieval, active
feedback, diversification, a service, or a general experiment framework. The
study cannot establish online satisfaction, future viewing behavior, commercial
value, or a new scoring family. A positive claim produced by leakage, weak or
missing comparators, post-confirmation tuning, or concealed failures is a failure
of this project even if the resulting software runs.

## 2. Approach and implementation choices

The blueprint already selects the scientific approach. The appropriate design is
a direct Python package with explicit stage functions, not another research
design exercise. Alternative implementation shapes are:

| Approach | Tradeoff | Disposition |
|---|---|---|
| Small package, pure numerical functions, thin command dispatcher | Slight initial structure; contracts, tests, and replay remain easy to inspect | Proposed approach, consistent with blueprint §11 |
| Notebook or one large script | Fast exploration; outcome access and resumability become harder to audit | Does not meet the evidence-boundary requirements as cleanly |
| General experiment scheduler or database | More extensibility; more infrastructure and hidden state than this bounded study needs | Excluded by the blueprint |

These are implementation proposals, not additional user-selected research
decisions. Use the existing repository root instead of creating the illustrative
`signed_preference_poc/` outer directory in §11.2. Stay on the current branch, as
requested. Keep standard-library JSON/JSONL/CSV plus NumPy arrays; use no service.

Use Python 3.11 or 3.12, NumPy, SciPy, scikit-learn, torch,
sentence-transformers, huggingface-hub, psutil, matplotlib, and pytest. Resolve
compatible versions on the target Mac and record exact versions; do not invent
pins in advance. Use CPU throughout, float32 encoder inference, and float64
geometry, fitting, distances, and metrics. Start with one worker and one BLAS
thread. Calibration may select the blueprint's four-thread alternative.

## 3. Scientific contract

For unit candidate vector `x`, positive prototypes `P_u(K)`, and negative support
vectors `N_u`, score:

```text
SignedMax(u, x; K, lambda) = max(x @ P_u(K).T) - lambda * max(x @ N_u.T)
K in {1, 2, 4}; lambda in {0, .25, .5, 1, 2, 4, 8}
```

Choose `S*` and the strongest required comparator `B*` using development users
only, then freeze them. The primary estimand is the equal-user mean of
`AUC_u(S*) - AUC_u(B*)` on confirmation users. Practical threshold: `delta=.01`.
It is conditional on this cohort, catalog, representation, feedback caps, and
later observed extreme rating submissions.

For a valid paired 95% interval `[L,H]`, report:

| Condition | Interpretation |
|---|---|
| `L > .01` | Supports the specified practical advantage over frozen `B*` |
| `H < .01` | Rules out the specified practical advantage at this precision |
| `H < 0` | Evidence of harm relative to `B*` |
| `L <= .01 <= H` | Practical question remains unresolved |
| Interval entirely inside `[-.01,.01]` | Supports a small-difference conclusion; does not prove equality |

These descriptions may overlap; the report must retain applicable qualifications.
Report all selected comparators, including any with a higher confirmation mean
than `S*`, without replacing `B*`. Missing, invalid, or inadequately tuned required
families prevent claims about the intended complete suite (§2.3).

Required outcome labels are supported contribution, informative negative result,
practical infeasibility, inconclusive evidence, and invalid experiment (§12.1).
A timeout or a nonsignificant result alone is not a scientific negative result.

## 4. Data and representation

### Acquisition and episode construction

Acquire MovieLens 1M from the official URL in §4.1; verify archive SHA256
`a6898adb50b9ca05aa231689da44c217cb524e7ebd39d264c56e2832f2c54e20`.
Retain its README locally. Decode movie metadata as latin-1 and validate unique
rating pairs, metadata IDs, timestamps, ratings, and complete joins. Do not read
demographics. A digest/schema/count discrepancy blocks splitting until explained;
do not silently deduplicate or filter it away.

For each user:

1. Sort all ratings, including neutral three-star ratings, by `(timestamp,item_id)`.
2. Compute `j=(7*n)//10`, then `t_cut=timestamp[j]` with zero-based indexing.
3. History has timestamps strictly below `t_cut`; evaluation has timestamps at or
   above it. Equal timestamps cannot straddle the split.
4. Positive means rating at least four; negative means at most two. Neutral rows
   influence the boundary but supply no labels or candidate items.
5. Require history counts at least `(10 positive,5 negative)` and evaluation counts
   at least `(5 positive,5 negative)`.
6. Independently keep the newest 128 positive and 64 negative history records,
   breaking history ties by item ID. Score every signed evaluation candidate.
7. Assign development when
   `int(sha256(('signed-study:'+str(user_id)).encode()).hexdigest()[:8],16)%5==0`.
   Remaining eligible users are confirmation users.

Reproduce the full audit in §4.4, including rating bursts, cap truncation, and
filtering funnel. Planning expectations are 1,000,209 ratings, 3,883 metadata
items, and 2,194 eligible users split into 418 development and 1,776 confirmation.
These are acquisition assertions to verify, not measurements made in this task.

Only after two failed 1M retrieval attempts and one official-page check within
the acquisition allowance may the declared MovieLens 100K access fallback be
used. Follow its separate parser and SHA256 in §4.6; expected counts are 1,682
metadata items and 62/225 development/confirmation users. All catalog operations
must use the active manifest cardinality. Do not switch datasets after an
unfavorable outcome, shrink the fallback as confirmation, or substitute features
if the encoder is unavailable.

### Frozen encoder

Use only `sentence-transformers/all-MiniLM-L6-v2`, with an immutable revision
resolved during acquisition, remote custom code disabled, and local model file
hashes/license recorded. Encode exactly:

```text
Movie: {original title including year}. Genres: {sorted genres joined by ", "}.
```

Preserve title wording/year, normalize whitespace, and sort genres. Use evaluation
mode, no gradients, explicit CPU, batch size 32 initially, maximum sequence length
256, and record token lengths/truncation. No ratings or generated text enter the
template. Encode in item-ID order; store float32 outputs, then cast and normalize
in float64. Reject nonfinite rows or norms at most `1e-12`; normalized norms must
be within `1e-12` of one.

During G2, select 128 probe items by metadata item-ID hash and test repeated
encoding of 32 items with maximum component difference at most `1e-5`. Admit the
full cache only after forecasting its cost. The cache manifest binds archive,
item-ID map, template, model revision/files, versions, dtype, shape, and array
hashes. Resolve revision before G2 and pin the validated environment after G1;
complete G2 cannot be reused under incompatible provenance.

## 5. Modules and evidence boundaries

```text
pyproject.toml                 # package, entry point, direct dependencies
requirements.lock.txt          # exact validated target environment
config/study.json              # fixed protocol and acquisition-resolved values
src/signed_study/
  __init__.py
  __main__.py                  # command parsing and dispatch only
  data.py                      # acquisition, validation, episodes, outcome access
  embed.py                     # template, model provenance, embedding cache
  models.py                    # prototypes, geometric methods, sklearn adapters
  metrics.py                   # metrics, selection summaries, paired inference
  run.py                       # stages, attempts, resource ledger, protocol lock
  report.py                    # saved evidence to tables, plots, report
tests/
  test_geometry.py
  test_data_and_firewall.py
  test_metrics_and_baselines.py
  test_stages_and_provenance.py
```

Use immutable records or equivalently explicit structures:

| Contract | Inputs and outputs | Boundary |
|---|---|---|
| `load_dataset(config)` | Ratings, metadata, validated manifest | Only auditor reads raw data during audit |
| `make_episodes(ratings,config)` | History records, candidate IDs, separate outcome records, cohort audit | Raw labels are not returned in a model-facing episode |
| `encode_items(metadata,config)` | Item-ID map, shared vectors, cache manifest | No ratings argument |
| `fit_user(method,history,embeddings,parameters)` | Fitted state and diagnostics | No evaluation labels or outcome handle |
| `score_user(model,candidate_ids,embeddings)` | Finite aligned scores | No evaluation labels |
| `evaluate_user(scores,labels,item_ids)` | AUC, NDCG@10, negative fraction, tie diagnostics | Outcome access belongs to evaluator |
| `run_stage(stage,config,ledger)` | Immutable attempts and artifact manifests | Checks prerequisites and provenance before work |
| `render_report(results_directory)` | Tables, figures, report | Consumes saved evidence; does not refit |

The auditor may access confirmation outcomes to determine eligibility and the
predeclared audit counts. Development code receives only development outcomes;
model construction may use confirmation histories but never their evaluation
labels. Store confirmation outcome arrays separately from model-facing episodes
in ignored local storage. The stage dispatcher must refuse their evaluation
loader before a valid lock. This is a testable program boundary, not a claim that
a researcher with filesystem access cannot read their own dataset.

Tests must show that permuting/replacing confirmation evaluation labels leaves
fit state and scores byte-identical; development cannot open the confirmation
outcome path; and a lock mismatch is rejected before outcomes are loaded.

## 6. Models and selection

### Prototypes and score semantics

Implement the deterministic spherical Lloyd constructor exactly as §5.1:
ascending item-ID inputs; retain duplicates for updates; distinct-row
initialization; first center nearest the raw mean or smallest-ID fallback;
farthest-first remaining centers; maximum-dot assignments with prototype-index
ties; normalized sums; retain previous centers for empty/zero-sum clusters.

Stop on unchanged assignments or improvement at most `1e-10*n`, reject objective
decrease exceeding `1e-9*n`, and cap at 50 updates. Log effective K, objective,
iterations, empty/zero-sum flags, and cap hits. Only the single global pre-freeze
increase to 100 updates allowed by §5.1 may alter this rule.

Use matrix products in candidate chunks of 512 initially. Do not clip cosine
scores, normalize final scores, or update support from outcomes. Empty negatives
give positive-only scores; empty positives raise an ineligible-history error.
Sort descending by stored score, then ascending integer item ID. Reconstruct
distances with dot clipping only as specified in §5.3; record excess over `1e-10`.
The distance ratio returns `.5`, `1`, or `0` for its explicit zero-distance cases.

### Required configurations

Let `A_K` be the positive prototype maximum, `A_all` the positive-exemplar maximum,
`B_max` the negative maximum, and `B_mean=x@mean(N)` with an unnormalized mean.

| Family | Configuration count | Contract |
|---|---:|---|
| MeanNegative | 21 | `A_K-lambda*B_mean` |
| SignedMax | 21 | `A_K-lambda*B_max` |
| AllExemplarMargin | 7 | `A_all-lambda*B_max` |
| PrototypeRatio | 3 | Nearest-negative / sum of nearest-class Euclidean distances |
| ExemplarRatio | 1 | Same ratio with all positive exemplars |
| LinearLogistic | 5 | Balanced L2 logistic, liblinear, intercept, `C=.01,.1,1,10,100` |
| RBFSVM | 20 | Balanced SVC, `C=.01,.1,1,10,100`; `g=.25,1,4,16`; `gamma=g/m2` |

The 78 entries also supply Centroid, PositiveMax, and RocchioDirection views
without additional fitting campaigns. Record all ten named families. Equivalent
computations may be reused while retaining declared configuration identities and
evaluation counts. Compute RBF `m2` from positive off-diagonal squared history
distances only, falling back to one when none is positive.

Preserve §6.2's exact solver settings: logistic tolerance `1e-8`, 2,000 iterations,
random state zero; SVC tolerance `1e-6`, 100,000 iterations, 64 MB cache, no
probabilities. Use class weights `h/(2*h_class)` and decision functions. Warnings,
nonfinite scores, iteration exhaustion, and nonzero SVC status are failed fits.
The single global pre-freeze increase to 4,000/200,000 iterations is allowed only
under the documented convergence/repair rules.

Select each family by macro development AUC. For differences at most `1e-6`,
prefer smaller K, lambda, C, g, then lexicographic configuration ID, as applicable.
Choose `B*` from the selected comparator families, excluding SignedMax and untuned
matched controls; exact between-family ties use lexicographic family ID.

Implement the one-time endpoint expansion algorithm in §6.3 in its exact request
order. Expand the RBF row/column including their corner first, then logistic,
MeanNegative, AllExemplarMargin, and SignedMax. Add at most 14 entries, giving at
most 92 total. Admission covers the complete requested expansion, never a partial
RBF corner. Log unadmitted coverage, endpoint/interior gains, and persistent outer
boundary winners. Do not equate bounded tuning with evidence of an optimum.

Confirmation includes all selected families and these frozen matched controls:
MeanNegative at `S*`'s K/lambda and prototypes; PositiveMax at its K; SignedMax at
K=1 and its lambda. Also compare independently tuned MeanNegative. Matched
max-versus-mean substitution changes both aggregation and user-specific penalty
scale. Log negative-mean norms; do not infer locality alone. Component selection
(`K>1` or `lambda>0`) is not proof of component benefit.

## 7. Measurement and diagnostics

Implement tie-aware rank AUC and compare it with independent pair enumeration
and sklearn on small fixtures. Weight users equally. NDCG@10 uses binary gains,
`k=min(10,candidate_count)`, logarithmic discounts, and item-ID tie breaking.
Use exact equality of stored float64 scores for AUC ties; log near ties separately.

Primary uncertainty uses 10,000 paired user bootstrap resamples, seed 20261029,
2.5/97.5 percentiles, batches at most 200, and a paired standard-error diagnostic.
Before freeze report `1.96*sd(dev_differences)/sqrt(planned_confirmation_count)`;
retain `.01` even when precision is limited. No rating-level resampling or
repeated deterministic runs may inflate the sample count.

Keep every scheduled confirmation user in the denominator. For missing user
differences use bounds `[-1,1]` when both scores are missing, `[AUC_S-1,AUC_S]`
when only the comparator is missing, and `[-AUC_B,1-AUC_B]` when only SignedMax is
missing. Known differences have equal bounds. Bootstrap lower/upper arrays with
identical user indices, taking the lower 2.5th and upper 97.5th percentiles as the
decision envelope. Also report sample-mean bounds and labeled complete cases.

Freeze history dispersion H, positive-negative overlap O, their development-median
thresholds, and negative-count bands `5–8,9–16,17–64` as specified in §8.4. As an
implementation convention, define the high side as `>=` the threshold and the
low side as `<`, and record this boundary in the lock. Empty strata show counts
and unavailable estimates; strata below 100 users are explicitly imprecise.
Their intervals are secondary and unadjusted. Do not select subgroup thresholds
from confirmation results.

G5 includes the prescribed float32 score-only check on the first 64 confirmation
IDs in digest order without refitting. A float32 deployment limitation need not
invalidate correct canonical float64 results. Contamination uses the first 200
confirmation IDs (or all available if fewer), seeds 1101–1103, identical cached
class-count-preserving swaps, and the four frozen methods in §7.4. It is conditional
and requires 20 minutes beyond the report reserve. Timing uses the same 32
development users, the entire active catalog, one warm-up, five timed repeats,
and deterministic method rotation. It measures cost, not catalog accuracy.

## 8. Stage control, resource accounting, and freeze

Expose the commands required by §11.4 through `python -m signed_study`:
`inspect`, `acquire`, `audit`, `calibrate`, `smoke`, `develop`, `freeze`, `confirm`,
`diagnose`, and `report`. Unit correctness is run through pytest. Keep the
dispatcher short; command help must state prerequisites and outcome access.

| Gate | Preconditions and output | Experimental allocation |
|---|---|---:|
| G0 | Inspect actual hardware, acquire manifests, audit counts, resolve model revision | 10 min |
| G1 | Independent correctness tests; no encoder required | 10 min |
| G2 | G1 passed; encoder probe, admitted full cache, 32-user calibration and forecast | 20 min |
| G3 | Cache/forecast valid; development-only smoke, grid, endpoints, precision and freeze | 40 min |
| G4 | Verified immutable lock; all selected families/controls, raw scores and paired analysis | 80 min |
| G5 | Locked applicable diagnostics and timing that fit remaining resources | 20 min |
| G6 | Evidence-based report, tables, figures, artifact inventory, even after early stop | 30 min protected |
| Shared repair allowance | At most two material cycles, counted cumulatively | 20 min |

These total 230 minutes, leaving ten within the 240-minute experimental ceiling.
Setup waiting has a separate 45-minute ceiling. Preserve the 30-minute report
reserve. Source editing is not experimental runtime; actual preprocessing,
calibration, tests, failed workloads, analysis, and packaging count as specified
by the blueprint. Record stage boundaries explicitly across process restarts.

Maintain an append-only JSONL ledger. Before starting work, record an attempt ID,
stage, inputs/configuration digest, planned cost, and evaluation reservation.
Record completion/failure, elapsed work, peak process-tree RSS, disk growth,
warnings, exceptions, and output hashes. Count every attempted declared
user/configuration evaluation, including equivalent entries, repairs, diagnostics,
and timing repetitions; never reset counts on resume or dataset fallback. An
interrupted attempt remains visible and must be reconciled conservatively before
another workload is admitted. Use a single process lock to prevent simultaneous
stage writers; this is not a job scheduler.

Hard ceilings are 100,000 evaluations, project disk growth 6 GiB, and process-tree
RSS `min(8 GiB,.25*installed_RAM)`. Require 12 GiB free initially and 4 GiB
estimated available-memory headroom before a heavy stage. Use project-local
`.venv` and `.cache`; never clear global caches. Sample RSS every 100 ms with
psutil and use OS peak facilities where available. Record the limits of sampling
and retain an interrupted workload as a failure rather than claiming an
unmeasured peak. Start no expensive stage on a non-target machine; unaffected
code, tests, and evidence preparation remain useful.

Admit projected work only if the 1.5 cost multiplier fits the remaining stage and
cumulative allocations while retaining the report reserve. If necessary, remove
optional work, reduce encoder batch/scoring chunks, then try primary sample pairs
in order `(418,1776),(256,1776),(418,1500),(256,1500),(418,1000),(256,1000)`.
Choose users by full digest order, never outcome. Preserve required baselines and
the metric. The 100K fallback uses all 62/225 users and no primary resize list.
Record practical infeasibility if no authorized scale fits.

Before confirmation, `freeze` requires validated prior stages and serializes:

- Dataset/model/cache/environment/configuration and code-commit hashes.
- Exact cohort, support/candidate IDs or digests, selected sample order, all
  evaluated configurations, family choices, `S*`, `B*`, and matched controls.
- Preprocessing/solver settings, tie rules, seeds, stratum boundaries, table
  schemas, metric implementation, timing protocol, resource snapshot, and deviations.

Hash canonical serialized bytes, store the lock digest in a separate sidecar,
and never rewrite the lock. Output artifacts and ledger events reference its
digest. Confirmation verifies live inputs, source/configuration hashes, and
prerequisites before opening outcomes. Use a clean committed source tree for the
freeze. Mutable logs reference a frozen snapshot rather than changing lock data.

Reuse completed artifacts only on exact input/provenance matches. Every retry
gets a new attempt ID; no overwrite of failure history. After confirmation
outcomes are exposed, an implementation-changing repair makes the corrected
same-user analysis exploratory (§8.5). Preserve both attempts and exposure
history. Do not invent a fresh holdout after observing results.

## 9. Artifacts and reporting

Keep raw archives, extracted data, per-item text/vectors, private outcome arrays,
raw candidate scores, local per-user metrics, model caches, and transient outputs
in ignored local directories. Ignore `data/`, `results/`, `.cache/`, and `.venv/`
by default. Use an explicit reviewed allowlist if aggregate report material is
later committed; the blueprint does not authorize publication or data contact.

Local outputs include environment/data/cohort/cache/calibration manifests,
resolved configuration, budget and run ledgers, the protocol lock, per-user
metrics and scores, failure/repair history, aggregate tables, figures, and
`results/report.md`. Figure/table regeneration reads saved evidence. Reporting
must also work before calibration or freeze, showing unavailable stages rather
than fabricated benchmark rows.

The report covers all ten content requirements in §12.2: question and outcome;
actual execution/hardware/gates; provenance/cohort/limitations; primary difference
and all comparators; matched controls; secondary strata and robustness; complete
cost accounting/amortization; correctness/failures/deviations; scoped prior-art
relationship; unresolved questions and at most one justified next step. Use the
blueprint's references as source material, preserving the missing-paper and
novelty limitations. Report paired difference and interval plots only when real
results exist. Include cold total cost and the 1,000-request amortization scenario.

## 10. Verification and completion criteria

The implementation plan must assign the following requirements to executable
checks, with meaningful failing fixtures before the associated implementation:

| Requirement | Evidence required before the gate advances |
|---|---|
| Data integrity and temporal split | Digest/schema failures; neutral and tied-timestamp fixtures; n=90 gives index 63; caps; disjoint support/candidates; hash split; full target audit |
| Geometry and constructor | Independent distance/loop oracle; max-min identity; lambda zero; K=J=1; rotations; duplicates; zero/empty cases; deterministic prototypes and objective checks |
| Canonical numerical behavior | `1e-10` oracle agreement up to lambda 16; chunk sizes 1/7/512 within `eta=1e-10*(1+lambda)`; ordering stable for gaps above `2*eta`; unit norms |
| Mechanism controls | Exact local-negative scores `.8,.8,.2,.2`; ratio separation; XOR obstruction; positive-only retrieval counterexample |
| Metrics and baselines | Perfect/reversed/tied/mixed AUC against two references; tied/short NDCG; linear and nonlinear fit fixtures; failed solver states surfaced |
| Evidence separation | Confirmation-label mutation invariance; development outcome-access rejection; no outcomes opened for a bad lock |
| Selection | 78 initial/at most 92 total entries; alias accounting; tie rules; complete endpoint corner; matched controls; failed configurations ineligible |
| Inference | Paired bootstrap reproducibility; bounded batches; missing-run confidence envelope; decision-boundary and denominator fixtures |
| Stages and resources | Invalid prerequisite rejection; cumulative restart accounting; reserved report budget; cap refusal; interrupted attempt; immutable artifact replay |
| Provenance and freeze | Changed model/data/config/source rejected; no overwrite; outcomes exposed only after verification; corrected exploratory labeling |
| Reporting | Saved-evidence regeneration for complete, partial, invalid, and early-stop cases; no invented metrics or claim beyond interval evidence |

Before target workloads, inspect actual architecture, chip, RAM, storage, macOS,
Python, BLAS/threading, package compatibility, and power state (§9.1). Do not infer
these from the machine name. Tests can establish correctness elsewhere; they
cannot establish M3 Max execution, runtime, memory, or comparative accuracy.

This documentation task is complete when a reviewed specification and a concrete,
sequenced implementation plan are committed on the current branch with the
blueprint intact. The subsequent study is complete when every reached gate has
valid preserved evidence and the appropriate scoped report exists; passing every
gate or producing a favorable effect is not required.

## 11. Choices deliberately left to the implementer

Choose small record types, internal helper names, canonical JSON implementation,
compressed score file layout, plot styling, and exact target-compatible dependency
pins. Choose the deterministic calibration sampler that covers the prescribed
history quartiles/long candidates and record the resulting 32 IDs before outcome
inspection. The history-only selection procedure may inspect candidate counts,
not evaluation labels or model performance. Choose the available-memory
measurement method and document it. None of these choices authorizes changes to
the scientific protocol, resource caps, fallback rules, or evidence boundaries.

## 12. Design review

A fresh research-software reviewer read this specification alongside the
blueprint and reported: "nothing important left" that would prevent planning.
This checks whether the written requirements are actionable; it is not executable
validation or approval of the design by the user.

The implementation proposals to review are the root package layout, separate
outcome-loading boundary, append-only attempt ledger with a single writer,
canonical lock plus digest sidecar, ignored local results with an aggregate
allowlist, and the `>=` high-stratum boundary convention. Scientific choices
remain those of the blueprint.
