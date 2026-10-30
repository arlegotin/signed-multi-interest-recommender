# Signed Preference Study Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement and execute the bounded frozen-content study, ending with the strongest supported conclusion even when the proposed ranker fails or execution stops early.

**Architecture:** A small Python package separates pure data/model/metric functions from stage execution. Separate outcome loading, immutable provenance, and a cumulative ledger enforce the confirmation and resource boundaries. Reports are regenerated from saved evidence rather than rerunning models.

**Tech Stack:** Python 3.11 or 3.12; NumPy, SciPy, scikit-learn, torch, sentence-transformers, huggingface-hub, psutil, matplotlib, pytest; JSON/JSONL/CSV/NPY storage; CPU execution.

**Spec:** [Approved implementation specification](../specs/2026-10-30-signed-study-design.md), derived from the authoritative [blueprint](../../blueprint.md). Read both before execution. This plan's commands and test expectations are requirements, not recorded execution results.

## Global Constraints

- Work in the existing repository root on the current branch; no worktree or new outer project directory.
- Use Python 3.11 or 3.12; resolve exact native target-compatible dependency versions and record them rather than inventing pins.
- Use CPU throughout, float32 encoder inference, and float64 geometry, fitting, distances, and metrics.
- Use only MovieLens 1M and `sentence-transformers/all-MiniLM-L6-v2`; the access-only 100K fallback follows blueprint §4.6.
- Exact split: `j=(7*n)//10`; eligibility `[10,5,5,5]`; support caps `[128,64]`; no unlabeled evaluation negatives.
- `K in {1,2,4}`; initial lambda grid `{0,.25,.5,1,2,4,8}`; 78 declared entries, at most 14 one-time additions and 92 total.
- Practical threshold `delta=.01`; 10,000 paired user bootstrap resamples; seed 20261029; batches at most 200.
- Experimental wall time 240 minutes; separate setup waiting 45 minutes; protect the final 30 experimental minutes for reporting.
- At most 100,000 cumulative attempted user/configuration evaluations, including repairs, equivalent declared entries, diagnostics, and timing repeats.
- Peak process-tree RSS `min(8 GiB,.25*installed_RAM)`; project disk growth 6 GiB; require 12 GiB free initially and 4 GiB available-memory headroom before heavy stages.
- At most two material repair cycles and 20 experimental repair minutes, included in the total budget.
- Use project-local `.venv` and `.cache`; keep raw/derived per-item content, outcomes, scores, and local runs out of git.
- Development outcomes select global parameters. Confirmation outcomes are opened only after verified freeze; a post-exposure implementation correction is exploratory on the same users.
- No new research directions, baseline removal, paid/remote compute, publication, or contact with third parties.

## Review Focus

1. A killed process must not erase a charged attempt, make its output appear complete, or reset caps on restart — Task 3.
2. A changed archive, reordered item map, or stale array cache must fail provenance checks before inference or outcome access — Tasks 2, 7, 10.
3. A required baseline with solver warnings or partially missing results must remain visible and must not quietly enter selection as a complete valid run — Tasks 5, 9, 10.
4. A threshold tie or an empty/small mechanism stratum must have deterministic membership and honest unavailable/imprecise output — Task 11.
5. A machine without the target hardware, encoder, or complete comparison must still produce a usable scoped early-stop report — Tasks 3, 12, 13.

---

## File map and shared conventions

| File | Responsibility |
|---|---|
| `pyproject.toml`, `requirements.lock.txt`, `.gitignore`, `README.md` | Installable package, validated environment, local artifact exclusions, operator commands |
| `config/study.json` | Fixed settings and explicit unresolved acquisition outputs |
| `src/signed_study/protocol.py` | `StudyConfig`, canonical digests, artifact provenance, verified protocol lock |
| `src/signed_study/data.py` | Dataset parsers/acquisition, episodes, audit, separated outcomes |
| `src/signed_study/budget.py` | Persistent attempts, admission, resource sampling, interruption reconciliation |
| `src/signed_study/embed.py` | Fixed text, pinned local model, shared embedding cache |
| `src/signed_study/models.py` | Deterministic prototypes, geometric scores, sklearn adapters, configuration registry |
| `src/signed_study/metrics.py` | Metrics, selection helpers, paired intervals, missing-run bounds |
| `src/signed_study/run.py` | Thin ordered stage functions and CLI-facing results |
| `src/signed_study/report.py` | Saved-evidence summaries, tables, plots, scoped report |
| `src/signed_study/__main__.py`, `__init__.py` | Command dispatcher and package identity |
| `tests/conftest.py` | Small synthetic local fixtures; no real downloads or confirmation data |
| `tests/test_geometry.py` | Independent mathematical oracles and constructor behavior |
| `tests/test_data_and_firewall.py` | Integrity, exact episodes, evidence separation |
| `tests/test_metrics_and_baselines.py` | Independent metrics, sklearn fits, inference |
| `tests/test_stages_and_provenance.py` | Budget, cache, selection, freeze, diagnostic/report integration |

All IDs are integers; parameter dictionaries use keys `K`, `lambda`, `C`, `g`.
Vectors are row-major NumPy arrays. Each owning module defines its records; do
not introduce a generic plugin/model framework. Artifact JSON has a schema
version and SHA256 input/output references. Hash canonical UTF-8 JSON with
sorted keys, compact separators, and nonfinite numbers rejected.
Run commands from the repository root with the project virtual environment
activated; bootstrap establishes pytest and source imports before the first red
check, so missing tooling cannot masquerade as a valid failing test.

Keep setup/test command start/end records in ignored
`results/bootstrap_accounting.jsonl` from the first computational check; Task 3
imports them once into the permanent ledger. Source-editing idle time is separate.
TDD tests use tiny synthetic cases; real audit/encoding/calibration happen only in
the execution tasks. Charge validation work to G1 and record any permitted
pre-confirmation stage reallocation instead of resetting its clock.

## Task 1: Validated protocol configuration and installable package

**Files:** Create `pyproject.toml`, `.gitignore`, `README.md`, `config/study.json`,
`src/signed_study/{__init__,__main__,protocol}.py`, `tests/conftest.py`, and
`tests/test_stages_and_provenance.py`.

**Interfaces:** Produce `StudyConfig`,
`load_config(path: Path) -> StudyConfig`,
`canonical_sha256(value: dict) -> str`, and
`VerifiedProtocol(lock_hash: str, payload: dict)` in `protocol.py`.
Only the verifier added in Task 10 issues a verified protocol to application code.
`StudyConfig` exposes validated scientific settings, resource settings, paths,
and nullable acquisition outputs `model_revision`/`environment_hash`.

- [ ] **Step 1: Establish the test environment.** Use an existing Python 3.11/3.12
  to create `.venv`, activate it, and install pytest with `--no-cache-dir`. Create
  minimal package/build metadata in `pyproject.toml`, including pytest
  `pythonpath = ["src"]`. Verify `python -m pytest --version` succeeds. Record
  setup waiting and project disk usage from this point; do not implement protocol
  behavior in this bootstrap step.
- [ ] **Step 2: Write configuration contract tests.** Define a `study_config_path`
  pytest fixture that writes the blueprint settings into `tmp_path`. Assert:

  ```python
  cfg = load_config(study_config_path)
  assert cfg.split_numerator == 7 and cfg.split_denominator == 10
  assert cfg.support_caps == (128, 64)
  assert cfg.delta == .01 and cfg.bootstrap_seed == 20261029
  assert cfg.bootstrap_resamples == 10_000
  assert canonical_sha256({"a": 1, "b": 2}) == canonical_sha256({"b": 2, "a": 1})
  ```

  Name this `test_fixed_protocol_and_canonical_hash`; add rejection tests for
  invalid caps, altered fixed threshold, nonfinite JSON, and unsupported schema.
- [ ] **Step 3: Run** `python -m pytest tests/test_stages_and_provenance.py -q`.
  Expected initial failure: package/API absent, not a downloaded-resource failure.
- [ ] **Step 4: Implement the configuration contract.** Include every §11.3 value,
  including solver limits, contamination seeds, grids, prototype tolerances,
  resource caps, and unresolved revision. Add installation metadata and command
  help only; help must not load an encoder or outcomes. Ignore `.venv/`, `.cache/`,
  `data/`, `results/`, bytecode, pytest cache, and packaging outputs.
- [ ] **Step 5: Install locally and rerun the tests.** Complete the declared direct
  dependencies and install with `python -m pip install --no-cache-dir -e .`; record setup waiting. Run the
  test command above and `python -m signed_study --help`; expect passes and help
  with no network/model side effects. Exact environment locking follows G1.
- [ ] **Step 6: Commit only these source/config/test/docs files.**
  `git commit -m "build: establish validated study configuration and package"`.

## Task 2: Exact data episodes, audit, and separate outcomes

**Files:** Create `src/signed_study/data.py`, `tests/test_data_and_firewall.py`;
modify `tests/conftest.py`.

**Interfaces:** Consume `StudyConfig` and `VerifiedProtocol`. Produce
`History(user_id,cutoff,role,positive_ids,negative_ids)`,
`Episode(history,candidate_ids)`, `Dataset(ratings,metadata,manifest)`, and
`EpisodeStore(development,confirmation,audit,paths)`.
Implement `load_dataset(config) -> Dataset`,
`make_episodes(dataset,config) -> EpisodeStore`,
`load_development_outcomes(store,user_id) -> np.ndarray`, and
`load_confirmation_outcomes(store,user_id,permit: VerifiedProtocol) -> np.ndarray`.
Outcome arrays align to episode candidate IDs; model-facing records have no labels.

- [ ] **Step 1: Write independent split/audit fixtures.** Define `ninety_ratings`
  as 90 increasing timestamps containing all three label classes. Assert in
  `test_integer_boundary_and_tied_timestamps`:

  ```python
  assert temporal_cutoff(ninety_ratings) == ninety_ratings[63].timestamp
  assert 7 * len(ninety_ratings) // 10 == 63
  assert max(history_timestamps) < min(evaluation_timestamps)
  assert not set(history_item_ids) & set(candidate_item_ids)
  ```

  Define `temporal_cutoff(ratings) -> int` in this task. Construct tied-boundary
  and neutral-rating cases independently, then cap/eligibility/hash-split cases.
  Assert `Episode` has no outcome labels, development cannot load confirmation
  outcomes, duplicate pairs/missing joins fail, latin-1 titles survive, and
  noncontiguous movie IDs map correctly. Test both schemas and an archive hash
  mismatch without network access using tiny local archives.
- [ ] **Step 2: Run** `python -m pytest tests/test_data_and_firewall.py -q`;
  expect missing data functions to fail.
- [ ] **Step 3: Implement parsers, validated acquisition, and episode stores.**
  Use the official URLs/digests from blueprint §4.1/§4.6, preserve READMEs, and
  enforce two 1M retrieval attempts plus an official-page check before access
  fallback. Store outcomes separately. Emit the full funnel/burst/cap audit;
  unexpected counts or schemas stop progression pending documented resolution.
- [ ] **Step 4: Run the same tests**; expect all passes with no real downloads.
  Check the history/evaluation split uses integer arithmetic and all neutral rows.
- [ ] **Step 5: Commit.**
  `git commit -m "feat: build audited temporal episodes with isolated outcomes"`.

## Task 3: Persistent budget admission and target-machine inspection

**Files:** Create `src/signed_study/budget.py`, `src/signed_study/run.py`;
modify `__main__.py` and `tests/test_stages_and_provenance.py`.

**Interfaces:** Consume `StudyConfig`. Produce
`Ledger(path: Path,config: StudyConfig)`,
`Ledger.admit(stage: str,projected_seconds: float,evaluations: int) -> str`,
`Ledger.finish(attempt_id: str,status: str,measurements: dict,outputs: dict) -> None`,
`Ledger.snapshot() -> dict`, and `inspect_environment(config) -> dict`.
Inject monotonic clock and resource sampler in tests; production uses actual
process-tree measurements. Output `environment.json` and append-only events.

- [ ] **Step 1: Write `test_restart_preserves_attempts_and_report_reserve`.** Use
  fake time/resources and a temporary ledger; assert:

  ```python
  attempt = ledger.admit("G3", projected_seconds=2, evaluations=78)
  reopened = Ledger(ledger.path, config)
  assert reopened.snapshot()["attempted_evaluations"] == 78
  assert attempt in reopened.snapshot()["unfinished_attempts"]
  ```

  Add assertions that 100,001 evaluations, insufficient disk/headroom, the 1.5
  forecast multiplier, or consuming protected report time cause refusal before
  work. Test torn final JSONL record recovery, one writer, conservative unfinished
  attempt reconciliation, fallback not resetting caps, and non-target inspection
  producing a blocker rather than a fabricated target measurement.
- [ ] **Step 2: Run** `python -m pytest tests/test_stages_and_provenance.py -q`;
  new ledger tests must fail before implementation.
- [ ] **Step 3: Implement admission, monitoring, and inspection.** Import bootstrap
  accounting exactly once. Distinguish setup waiting from experimental time;
  reconcile interrupted intervals conservatively and explicitly. Persist attempt
  starts before work; use atomic complete-artifact writes and a single-writer
  lock. Enforce elapsed/evaluation/disk/RSS caps during work and sample RSS at
  100 ms. Capture §9.1 hardware/OS/Python/BLAS/power data without assuming an M3.
  Wire `inspect --config`; later commands use this same ledger.
- [ ] **Step 4: Run the stage tests** and
  `python -m signed_study inspect --config config/study.json`.
  Expect valid inspection evidence or a clear target blocker; no experimental
  model workload is admitted by inspection itself.
- [ ] **Step 5: Commit.**
  `git commit -m "feat: enforce persistent study budgets and environment gates"`.

## Task 4: Deterministic prototypes and exact geometric scoring

**Files:** Create `src/signed_study/models.py`, `tests/test_geometry.py`.

**Interfaces:** Produce `PrototypeResult(vectors,objective,iterations,flags)`,
`spherical_prototypes(X: np.ndarray,item_ids: np.ndarray,K: int,*,max_updates=50) -> PrototypeResult`,
`signed_scores(X,P,N,penalty: float,*,chunk_size=512) -> np.ndarray`, and
`ratio_scores(X,P,N) -> np.ndarray`. Inputs are unit float64 rows. Keep a slow
loop/direct-distance oracle in tests, independent of production score helpers.

- [ ] **Step 1: Write `test_exact_local_negative_control`** using the §7.2 vectors:

  ```python
  P = np.array([[0., 0., 1.], [0., 0., -1.]])
  N = np.array([[1., 0., 0.], [-1., 0., 0.]])
  X = np.array([[0., .6, .8], [0., -.6, -.8], [.6, 0., .8], [-.6, 0., -.8]])
  np.testing.assert_allclose(signed_scores(X, P, N, 1), [.8, .8, .2, .2], atol=1e-15)
  ```

  Add oracle/max-min/rotation tests through lambda 16 with `1e-10` tolerance;
  lambda-zero and single-pair identities; duplicate/empty/zero-sum/identical
  vectors; exact ratio zero cases; deterministic prototype ties/objective/unit
  norms; cap-hit flags; XOR obstruction and retrieval counterexample. For chunks
  1/7/512 assert score errors within `eta=1e-10*(1+lambda)` and ordering stability
  only for reference gaps above `2*eta`.
- [ ] **Step 2: Run** `python -m pytest tests/test_geometry.py -q`;
  expect missing scorer/constructor failures.
- [ ] **Step 3: Implement blueprint §5 exactly.** Use distinct-row initialization
  and duplicate-weighted spherical updates. Preserve unnormalized negative means,
  raw negative cosines, deterministic item-ID ranks, retained empty/zero-sum
  centers, objective thresholds `1e-10*n`/`1e-9*n`, and norm tolerance `1e-12`.
  No encoder imports are needed for these pure numerical functions.
- [ ] **Step 4: Run the geometry tests**; expect every independent oracle and
  counterexample to pass, including the ratio baseline's local-negative success.
- [ ] **Step 5: Commit.**
  `git commit -m "feat: implement deterministic signed and nearest-class geometry"`.

## Task 5: Equally informed baselines and declared configuration registry

**Files:** Modify `models.py`; create `tests/test_metrics_and_baselines.py`.

**Interfaces:** Consume `History`, shared item-ID vectors, and Task 4 geometry.
Produce `ModelConfig(family: str,parameters: dict,id: str)`,
`UserModel(method,parameters,support_ids,state,diagnostics)`,
`initial_configurations(config) -> list[ModelConfig]`,
`fit_user(method,history,embeddings,parameters) -> UserModel`, and
`score_user(model,candidate_ids,embeddings,*,chunk_size=512) -> np.ndarray`.
`embeddings` accepts a vector view with `rows(item_ids) -> np.ndarray`; use a tiny
in-memory view in tests until Task 7 supplies `EmbeddingCache`. Failed fit
diagnostics must be returned/persisted by the caller.

- [ ] **Step 1: Write `test_registry_and_baseline_information_parity`.** Assert
  registry family counts `[21,21,7,3,1,5,20]` and total 78; extracted Centroid,
  PositiveMax, and Rocchio views do not add entries. Assert:

  ```python
  assert len(initial_configurations(config)) == 78
  np.testing.assert_allclose(mean_negative_scores, A_K - penalty * X @ N.mean(axis=0))
  assert all(model.support_ids == expected_support_ids for model in fitted_models)
  ```

  Build `fitted_models` from the same tiny support in the test. Check logistic
  separable ordering, RBF nonlinear ordering at a declared flexible setting,
  history-only positive-distance median with all-identical fallback `m2=1`, class
  weights, and surfaced solver warnings/status/nonfinite outputs. Mutate only
  candidate labels and assert byte-identical fit state/scores.
- [ ] **Step 2: Run** `python -m pytest tests/test_metrics_and_baselines.py -q`;
  expect absent registry/fit adapter failures.
- [ ] **Step 3: Implement every §6.1 family.** Use exact §6.2 sklearn settings,
  decision functions, contiguous float64 arrays, and per-user fits. Reuse
  equivalent score computation without dropping declared configuration records.
  Release estimator objects after saving outputs. Never replace failed SVC fits
  with a weaker method or tune a convergence repair per user.
- [ ] **Step 4: Run the baseline and geometry suites**; expect valid fixtures to
  pass and all injected solver failures to be recorded explicitly.
- [ ] **Step 5: Commit.**
  `git commit -m "feat: add complete bounded baseline suite and configuration registry"`.

## Task 6: Independent metrics, paired uncertainty, and missing-run decisions

**Files:** Create `src/signed_study/metrics.py`; modify
`tests/test_metrics_and_baselines.py`.

**Interfaces:** Produce `evaluate_user(scores,labels,item_ids) -> dict`,
`difference_bounds(auc_signed: float|None,auc_baseline: float|None) -> tuple[float,float]`,
`confidence_envelope(lower,upper,*,seed=20261029,resamples=10000,batch_size=200) -> dict`,
and `interpret_interval(low: float,high: float,delta=.01) -> list[str]`.
Metric keys are `auc`, `ndcg10`, `negative_at_10`, and `tie_fraction`.

- [ ] **Step 1: Write `test_auc_ties_and_missing_result_envelope`.** Compare AUC
  with both direct pair enumeration and sklearn; use perfect/reversed/all-tied
  cases `1/0/.5` and a mixed-tie case. Assert:

  ```python
  assert difference_bounds(None, None) == (-1., 1.)
  assert difference_bounds(.75, None) == (-.25, .75)
  assert difference_bounds(None, .25) == (-.25, .75)
  result = confidence_envelope(np.full(4, .02), np.full(4, .02))
  assert result["low"] == pytest.approx(.02)
  assert result["high"] == pytest.approx(.02)
  ```

  Add hand-calculated short/tied NDCG and negative-at-10 checks, reproducible paired
  resampling with batches at most 200, whole-cohort denominator checks, threshold
  equalities, and cases where missing results change the conclusion. Intervals
  must use user resamples shared across compared methods, not rating resamples.
- [ ] **Step 2: Run** `python -m pytest tests/test_metrics_and_baselines.py -q`;
  expect missing metric/inference functions to fail.
- [ ] **Step 3: Implement rank AUC, deterministic ranked metrics, and paired analysis.**
  Use exact stored-score ties, equal-user macro aggregation, sample-mean missing
  bounds, bootstrap confidence envelope, and complete-case diagnostic labels.
  Include paired standard error and pre-freeze half-width helpers. Reject invalid
  single-class metric inputs rather than silently admitting an ineligible user.
- [ ] **Step 4: Run the metrics and baseline suite**; expect agreement with both
  independent AUC references and deterministic bootstrap output.
- [ ] **Step 5: Commit.**
  `git commit -m "feat: implement paired metrics and conservative missing-run inference"`.

## Task 7: Frozen text encoder and provenance-checked shared cache

**Files:** Create `src/signed_study/embed.py`; modify
`tests/test_stages_and_provenance.py` and `tests/conftest.py`.

**Interfaces:** Produce `EmbeddingCache(item_ids,vectors,manifest)` with
`rows(item_ids) -> np.ndarray`, `format_movie(title: str,genres: list[str]) -> str`,
`resolve_encoder(config) -> dict`, `encode_items(metadata,config) -> EmbeddingCache`,
and `load_embedding_cache(path,expected_manifest: dict) -> EmbeddingCache`.
`vectors` are renormalized float64; stored encoder outputs are float32.

- [ ] **Step 1: Write `test_template_and_cache_provenance`** with a deterministic
  injected encoder so tests download nothing. Assert:

  ```python
  assert format_movie("Toy Story (1995)", ["Comedy", "Animation"]) == (
      "Movie: Toy Story (1995). Genres: Animation, Comedy."
  )
  assert cache.vectors.dtype == np.float64
  np.testing.assert_allclose(np.linalg.norm(cache.vectors, axis=1), 1., atol=1e-12)
  ```

  The fixture creates and reloads a small cache. Assert changed model revision,
  template, dataset digest, item-ID order, shape/dtype, or array bytes are rejected;
  zero/nonfinite vectors fail; incomplete temporary outputs are never reused;
  and unresolved revision prevents full encoding. Test repeatability acceptance
  at `1e-5` and rejection above it.
- [ ] **Step 2: Run** `python -m pytest tests/test_stages_and_provenance.py -q`;
  expect missing embedding/template/cache functions to fail.
- [ ] **Step 3: Implement pinned local CPU encoding and manifests.** Use no custom
  remote code, evaluation/no-grad mode, 384 output dimensions, 256 wordpieces,
  initial batch 32, sorted IDs, actual token/truncation accounting, local cache
  paths, model-file hashes and license. Lazy-load torch/model code so data/tests/
  report paths work when the encoder is unavailable. Cache reuse requires every
  declared provenance field and artifact hash to match.
- [ ] **Step 4: Run the stage and geometry tests** with the injected encoder;
  expect passes without network or real model inference. Actual repeatability
  and catalog encoding remain G2 work.
- [ ] **Step 5: Commit.**
  `git commit -m "feat: add immutable encoder provenance and shared vector cache"`.

## Task 8: Ordered stage execution and measured resource calibration

**Files:** Modify `run.py`, `__main__.py`, `budget.py`, and
`tests/test_stages_and_provenance.py`.

**Interfaces:** Consume Tasks 1–7. Produce
`run_stage(stage: str,config: StudyConfig,ledger: Ledger) -> dict`,
`select_calibration_users(episodes: list[Episode]) -> list[int]`, and
`forecast_scale(measurements: dict,cohort: EpisodeStore,ledger: Ledger) -> dict`.
Stage result dictionaries contain `status`, `artifacts`, `attempt_ids`, and
`stop_reason`; incomplete prerequisites are explicit, not successful no-ops.

- [ ] **Step 1: Write `test_calibration_orders_probe_cache_and_user_fits`.** Inject
  lightweight probe/cache/fitting callables and a fake clock; assert:

  ```python
  assert events == ["probe_128", "repeat_32", "admit_full_cache", "encode_catalog", "fit_32_users"]
  assert result["forecast"]["safety_multiplier"] == 1.5
  assert result["forecast"]["scale"] == [418, 1776]
  ```

  The fixture appends those events from instrumented stage dependencies. Test
  refusal after a bad probe, missing G1 evidence, absent complete cache, memory
  failure, or exhausted reserve. Assert exact primary resize order, all-62/225
  fallback use, and infeasibility when the smallest permitted scale cannot fit.
  Reusing a matching completed stage must add no new workload or evaluations.
- [ ] **Step 2: Run** `python -m pytest tests/test_stages_and_provenance.py -q`;
  expect absent orchestration/forecast behavior to fail.
- [ ] **Step 3: Implement G0–G2 and development-only smoke dispatch.** Wire
  `acquire`, `audit`, `calibrate`, and `smoke --development-users 8`, preserving
  prerequisite ordering and the common ledger. For calibration choose four
  support-count rank quartiles, ties by full user digest; in each choose four
  smallest digests plus four longest candidate lists not yet selected, ties by
  digest. Order the final 32 IDs by digest. This uses no outcomes/performance.
  Measure probe/repeat/cache load/encode/fits/scoring/writes/RSS, solver endpoints,
  and numerical variation. Forecast actual candidate/support sizes with 1.5
  margin; apply §9.3's optional-work/batch/chunk/scale sequence exactly. Persist
  global repair decisions and never change one method's support or tolerances
  selectively. Record canonical chunk/thread policy and planned cohort IDs.
- [ ] **Step 4: Run the stage tests and full synthetic suite** with
  `python -m pytest -q`; expect passes and no model downloads. Verify help now
  lists prerequisite and evidence-access behavior for implemented commands.
- [ ] **Step 5: Commit.**
  `git commit -m "feat: add gated acquisition calibration and development smoke"`.

## Task 9: Development grid, endpoint expansion, and frozen selection inputs

**Files:** Modify `run.py`, `models.py`, `metrics.py`, and
`tests/test_stages_and_provenance.py`.

**Interfaces:** Produce `select_family(rows: list[dict]) -> ModelConfig`,
`select_comparator(family_choices: dict) -> ModelConfig`,
`endpoint_requests(family_choices: dict,config: StudyConfig) -> list[ModelConfig]`,
and `develop(config: StudyConfig,ledger: Ledger) -> dict`.
Development output includes all attempts, chosen family configurations, `S*`,
`B*`, matched-control configurations, endpoint coverage, planned precision, and
history-stratum thresholds. It does not yet grant outcome access.

- [ ] **Step 1: Write `test_selection_ties_endpoints_and_failed_configurations`.**
  Create synthetic development rows with known AUCs; assert `<=1e-6` within-family
  ties prefer K/lambda/C/g/ID order and exact between-family ties prefer family ID.
  Assert SignedMax/matched controls cannot become `B*`, and failed configurations
  cannot win. For all eligible endpoints assert:

  ```python
  assert len(requests) == 14
  assert len(rbf_requests) == 10
  assert (outer_C, outer_g) in {(r.parameters["C"], r.parameters["g"]) for r in rbf_requests}
  assert len(initial_configurations(config)) + len(requests) == 92
  ```

  The fixture's outer C is `.001` or `1000`; outer g is `.0625` or `64`, according
  to the winning endpoint. Define `outer_C`/`outer_g` accordingly. Cover
  lower/upper endpoints, deduplication, exact request order, a single complete
  admission, no recursive expansion, alias evaluation charging, and missing
  required-family stop/qualification. Ensure development never invokes the
  confirmation outcome loader.
- [ ] **Step 2: Run** `python -m pytest tests/test_stages_and_provenance.py -q`;
  expect missing selection/expansion/development functions to fail.
- [ ] **Step 3: Implement development execution.** Persist each declared
  user/configuration attempt and score/metric artifacts before aggregation. Use
  only calibrated fixed development IDs. Log failed shares; rerun a fully
  affected family only within repair limits. Select the complete requested
  endpoint expansion once, or log all unavailable requested coverage if it cannot
  be admitted. Report persistent outer-boundary winners and interior gains;
  incomplete tuning cannot support a complete-suite claim. Estimate precision
  without changing delta; save matched control parameters and prototype identity.
  Implement the within-family tolerance by finding the maximum valid macro AUC,
  then choosing the simplest configuration among those within `1e-6` of that
  maximum; do not use a nontransitive pairwise approximate-sort comparator.
- [ ] **Step 4: Run the stage and metric suites**; expect deterministic choices
  independent of row order and explicit stop/coverage metadata for failures.
- [ ] **Step 5: Commit.**
  `git commit -m "feat: implement bounded development tuning and comparator selection"`.

## Task 10: Immutable freeze and locked confirmation with failure bounds

**Files:** Modify `protocol.py`, `data.py`, `run.py`, `__main__.py`, and both
`tests/test_stages_and_provenance.py` and `tests/test_data_and_firewall.py`.

**Interfaces:** Produce `freeze_protocol(config,development,ledger) -> Path`,
`verify_protocol(lock_path: Path,live_inputs: dict) -> VerifiedProtocol`, and
`confirm(lock_path: Path,ledger: Ledger) -> dict`.
`confirm` alone passes a verified permit to the confirmation evaluator; fitting
and scoring retain their label-free signatures. The protocol sidecar digest
hashes the immutable lock bytes; outputs reference that digest.

- [ ] **Step 1: Write `test_freeze_rejects_mutation_before_outcome_access`.**
  Define a `frozen_study` fixture containing tiny saved development artifacts,
  a clean source/config snapshot, and separate synthetic confirmation outcomes.
  Spy on the outcome loader and assert:

  ```python
  assert verify_protocol(lock_path, live_inputs).lock_hash == expected_lock_hash
  with pytest.raises(ProtocolMismatch):
      verify_protocol(lock_path, changed_model_inputs)
  assert outcome_reads == []
  ```

  Also test code/data/config/cache/table-schema/metric/ID mutations, dirty source,
  overwrite refusal, byte-identical scores after label permutation, every scheduled
  user retained after a failed fit, all-family/matched-control coverage, and
  implementation-changing repair after outcome exposure marked invalid then
  corrected exploratory. A systemic defect stops the affected family and keeps
  its original attempts. `ProtocolMismatch` is defined by this task.
- [ ] **Step 2: Run**
  `python -m pytest tests/test_data_and_firewall.py tests/test_stages_and_provenance.py -q`;
  expect missing lock/confirmation behavior to fail.
- [ ] **Step 3: Implement the complete §6.3 freeze payload and verification.**
  Include immutable source commit/content digests, environment/cache/config
  hashes, planned IDs, configurations/choices, seeds, thresholds, exact tie rules,
  timing/table schemas, metric code, deviations, and a ledger snapshot. Verify
  before reading outcomes, write attempts and raw aligned scores/metrics locally,
  and use Task 6's whole-denominator envelope. Track exposure and invalidation
  explicitly. Code commit provenance must remain reconstructible; mutable logs
  append to the run and never modify the lock.
- [ ] **Step 4: Run the firewall, stage, and inference suites**; expect no outcome
  read on any rejected lock, reproducible matched scores, and honest partial
  results. Run `python -m signed_study confirm --help` and `freeze --help`.
- [ ] **Step 5: Commit.**
  `git commit -m "feat: freeze protocol and enforce locked confirmation evidence"`.

## Task 11: Bounded mechanism, numerical, contamination, and timing diagnostics

**Files:** Modify `run.py`, `metrics.py`, and `tests/test_stages_and_provenance.py`.

**Interfaces:** Produce `history_statistics(history,embeddings) -> dict`,
`assign_stratum(H: float,O: float,thresholds: dict) -> str`,
`contaminate_history(history: History,seed: int) -> History`, and
`diagnose(lock_path: Path,ledger: Ledger) -> dict`.
History statistics are `H`, `O`, `negative_mean_norm`, and capped negative count.

- [ ] **Step 1: Write `test_stratum_boundaries_and_contamination_parity`.**
  With synthetic thresholds/history assert:

  ```python
  assert assign_stratum(.2, .3, {"H": .2, "O": .3}) == "high_H_high_O"
  assert set(corrupted.positive_ids) | set(corrupted.negative_ids) == original_ids
  assert len(corrupted.positive_ids) == len(history.positive_ids)
  assert len(corrupted.negative_ids) == len(history.negative_ids)
  ```

  Compute `corrupted` with the function under test. Assert exactly
  `q=max(1,floor(.1*min(h_pos,h_neg)))` swaps, seeds 1101–1103 reproducible across
  methods, original evaluation labels untouched, and frozen hyperparameters used.
  Test empty strata as unavailable, fewer than 100 users as imprecise, and
  contamination skipped without applicability or 20 minutes beyond the reserve.
  A recording timing fake must observe one warm-up/five repeats/32 development
  users, stable method rotation, full active catalog, and charged evaluations.
- [ ] **Step 2: Run** `python -m pytest tests/test_stages_and_provenance.py -q`;
  expect diagnostic APIs and skip/stratum behavior to fail initially.
- [ ] **Step 3: Implement only prescribed diagnostics.** Freeze development
  medians before confirmation; show four H/O strata, three negative-count bands,
  and mean-norm scale-confound diagnostics. Recompute geometry in float32 without
  refitting for the first 64 confirmation digest IDs. Apply conditional corruption
  to the first 200 confirmation IDs using original supports/parameters. Use the
  fixed 32 development users for full-catalog latency; capture cold/shared costs,
  fit/score/rank boundaries, process-tree RSS, median/p95/repeat variability, and
  the 1,000-request amortization scenario. Optional work never consumes reserve.
- [ ] **Step 4: Run stage tests and the full synthetic suite**; expect deterministic
  membership/swaps, explicit secondary labels and skipped reasons, no new tuning,
  and no accuracy claim from timing or float32 checks.
- [ ] **Step 5: Commit.**
  `git commit -m "feat: add bounded mechanism robustness and cost diagnostics"`.

## Task 12: Regenerable reports for complete, partial, invalid, and early-stop runs

**Files:** Create `src/signed_study/report.py`; modify `run.py`, `__main__.py`,
`README.md`, and `tests/test_stages_and_provenance.py`.

**Interfaces:** Produce `render_report(results_directory: Path) -> dict` and wire
`report --results results`. It reads manifests, ledgers, score/metric summaries,
and diagnostic artifacts; it does not call fitting, encoding, or downloads.
Return paths for `report.md`, tables, figures, and the artifact inventory.

- [ ] **Step 1: Write `test_report_preserves_scope_without_complete_results`.**
  Add small saved-evidence fixtures for all five outcome labels and for missing
  encoder/non-target stops. Assert:

  ```python
  assert "inconclusive evidence" in report_text
  assert "not executed" in early_stop_text
  assert "corrected exploratory" in corrected_text
  assert fit_call_count == encode_call_count == 0
  ```

  Assert primary `B*` remains named even if another confirmation comparator wins;
  invalid/missing baselines and bounded tuning appear as limitations; no benchmark
  table is invented without observations; cost totals include failed work; and
  regenerated aggregate tables match saved metric values. Exports must not
  include raw data, vectors, private labels, candidate IDs, or per-item scores.
- [ ] **Step 2: Run** `python -m pytest tests/test_stages_and_provenance.py -q`;
  expect report generation and complete/early-stop assertions to fail initially.
- [ ] **Step 3: Implement §12.2's ten report requirements and artifact inventory.**
  State actual stages, hardware, sources, selected parameters, delta and interval,
  all comparator results, controls, strata/robustness, full costs, failures,
  uncertainty, prior-art boundaries, and at most one justified next step. Plot
  paired differences/intervals only from actual observations. Document all CLI
  commands, environment recreation, provenance/retry behavior, fallback rules,
  local licensing/redistribution boundaries, and non-target usage in README.
- [ ] **Step 4: Run** `python -m pytest -q` and regenerate each saved fixture's
  report twice; expect passing checks and identical evidence tables. Inspect one
  complete and one early-stop report for claims unsupported by their artifacts.
- [ ] **Step 5: Commit.**
  `git commit -m "feat: render reproducible study reports including early stops"`.

## Task 13: Execute target feasibility, correctness, calibration, and development

**Files:** Update `requirements.lock.txt`, `config/study.json`, and `README.md` only
as actual validated setup requires. Local artifacts under `data/`, `.cache/`, and
`results/` remain ignored. Ordinary repairs modify the owning source/test files.

**Interfaces:** Consume all implemented commands and the validated code. Produce
G0–G3 evidence, exact environment/model provenance, selected cohorts/parameters,
and an immutable lock, or an explicit early-stop reason passed to Task 14.

- [ ] **Step 1: Inspect and acquire G0 evidence.** Run
  `python -m signed_study inspect --config config/study.json`, then `acquire` and
  `audit` with that config only when inspection admits the target workload.
  Verify actual M3 Max/native architecture, RAM/storage/power, archive digest,
  full cohort audit, and immutable model revision. Keep setup within 45 minutes
  and G0 within allocation. On indispensable hardware/model/access failure,
  preserve unaffected evidence and go directly to the early-stop report path.
- [ ] **Step 2: Establish G1 and lock the validated environment.** Run
  `python -m pytest -q` through recorded accounting; require all independent
  checks to pass. Save exact compatible installed versions in
  `requirements.lock.txt`, record Python/BLAS/model file hashes, and test the
  documented environment restoration command. Apply only allowed repairs and
  stage reallocations; unresolved validity defects stop the experiment.
- [ ] **Step 3: Commit the validated execution inputs.** Stage only package,
  config, environment lock, tests, and operator documentation as changed. Use
  `git diff --cached --check` and inspect staged paths before
  `git commit -m "build: lock validated target study environment and provenance"`.
  If this step has no tracked changes, record the existing source commit instead
  of creating an empty commit.
- [ ] **Step 4: Run G2 calibration and development-only smoke.** Execute
  `python -m signed_study calibrate --config config/study.json`, then
  `python -m signed_study smoke --config config/study.json --development-users 8`.
  Require admitted full active-catalog cache, repeatability, fair measured
  forecasts, required baseline feasibility, selected scale, and intact reserve.
  If no permitted scale fits, preserve practical-infeasibility evidence and go
  to reporting rather than weakening the study.
- [ ] **Step 5: Run G3 development and inspect readiness.** Execute
  `python -m signed_study develop --config config/study.json`. Inspect failures,
  endpoint coverage, selected families/`S*`/`B*`, matched controls, precision,
  histories-only thresholds, and remaining costs. A small noisy gain alone is
  not a stop reason. Missing required comparisons or unresolved correctness
  issues block the intended confirmation and must be reported.
- [ ] **Step 6: Commit any authorized pre-freeze repair/config changes.** Require
  affected tests to pass and log material repair cycles and costs. Stage exact
  relevant files, check the diff, and use a message describing the actual repair;
  do not modify the scientific question. Confirm the source tree is clean.
- [ ] **Step 7: Freeze only when prerequisites pass.** Execute
  `python -m signed_study freeze --config config/study.json`. Verify the sidecar
  digest, source/environment/data/cache inputs, all configuration choices,
  outcome-access history, table schemas, ledger snapshot, and reserved resources.
  Confirmation labels must still be unopened by selection code. The lock and raw
  run artifacts remain local and immutable, not committed into public git.

## Task 14: Execute locked comparison, applicable diagnosis, and final report

**Files:** Local `results/` evidence and report; optional tracked README corrections
and an aggregate-only `docs/study-outcome.md` if real execution produced a useful
shareable summary. Do not commit raw/local reproducibility artifacts.

**Interfaces:** Consume the verified lock from Task 13, or its explicit early-stop
status. Produce `results/report.md`, accurate artifact inventory, supported outcome
label/recommendation, and a clean tracked source tree.

- [ ] **Step 1: Run G4 only if admitted.** Execute
  `python -m signed_study confirm --lock results/protocol.lock.json`. Require all
  selected families and matched controls for the frozen cohort; preserve raw
  scores, fit diagnostics, failures, exposure history, and missing-run envelopes.
  Analyze the frozen primary comparison once. Do not change `B*`, delta, metric,
  representation, or candidate set after observing outcomes.
- [ ] **Step 2: Resolve execution defects under the failure policy.** Classify
  systemic failures, preserve all attempts, and charge allowed repairs. An
  implementation-changing post-exposure correction must remain labeled corrected
  exploratory on this holdout. Do not invent fresh confirmation users. If no
  repair is needed, record that and proceed without broadening tests or workload.
- [ ] **Step 3: Run only applicable G5 work that fits.** Execute
  `python -m signed_study diagnose --lock results/protocol.lock.json` when a lock
  exists and resources allow. Check secondary strata, numerical stability,
  conditional contamination, and measured cost; retain reasons for omitted work.
  Protect reporting time even if an optional diagnostic is scientifically useful.
- [ ] **Step 4: Generate G6 for every reached outcome.** Execute
  `python -m signed_study report --results results`, including early-stop cases
  with no lock. Check all §12.2 report contents against saved evidence, all
  comparator qualifications, selection-versus-component-evidence distinctions,
  and cost/deviation accounting. Regenerate tables/figures from saved artifacts
  and confirm the result is the same without fitting or outcome reselection.
- [ ] **Step 5: Preserve local evidence and commit only appropriate summaries.**
  Inventory hashes and commands needed for reconstruction. If writing
  `docs/study-outcome.md`, include only aggregate evidence, source/lock digests,
  reached gates, outcome limits, and the local regeneration command. Inspect
  staged content for dataset/derived-item material, run `git diff --cached --check`,
  and run `git commit -m "docs: record signed study outcome and reproducibility"`. If the
  run stopped with no useful public summary or tracked changes, keep the full
  local report and existing meaningful commits; do not invent an empty result.
- [ ] **Step 6: Hand off the evidence-supported outcome.** State actual commands,
  test/gate results, what did not run, primary/secondary conclusions and limits,
  cumulative costs, local report path, and relevant commits. Recommend at most
  one next study only if the observed evidence justifies it.

## Self-review and coverage

| Spec requirements | Owning tasks |
|---|---|
| Intent, authority, no new research scope, target stack (§1–3) | Global constraints; 1, 3, 12–14 |
| Acquisition, exact episodes, audit, fallback (§4) | 2, 3, 13 |
| Frozen representation and complete cache (§4) | 7, 8, 13 |
| Module contracts and evidence separation (§5) | File map; 1, 2, 5, 7, 10 |
| Prototype/scoring/numerical rules and strong comparisons (§6) | 4, 5, 9 |
| Metrics, missing results, uncertainty, precision (§7) | 6, 9, 10 |
| Strata, robustness, contamination, cost (§7) | 8, 11, 14 |
| Commands, ledger, caps, stage gates, freeze, repairs (§8) | 3, 8–10, 13, 14 |
| Local artifacts, report and outcomes (§9) | 1, 7, 10, 12, 14 |
| Independent oracles and acceptance evidence (§10) | 2–12; operational verification 13–14 |
| Builder choices and reviewed design (§11–12) | File map, shared conventions, Task 8 sampler |

The five Review Focus cases each have an owning test step above. Interfaces use
the same records and key names throughout. Future code may split an oversized
module by responsibility while preserving these contracts; this is not authority
to add infrastructure. Tasks 1–12 are implementation work; Tasks 13–14 execute
the gated study and may terminate early with a complete scoped report.

## Handoff

This plan is documentation awaiting execution; none of its checkboxes is a claim
of completed implementation or measurement. The user approved the design and
requested this plan, not execution of the study in the documentation turn.

For later execution, native implementation is a reasonable default because the
tasks share tightly ordered data, ledger, and provenance contracts; fresh task
contexts would repeatedly reload those boundaries. Subagent-driven execution is
also supported if the user prefers per-task independent review. Choose the
execution method with the user before implementation, then load the corresponding
Superpowers execution skill. Preserve the current-branch instruction.
