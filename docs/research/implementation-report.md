# Study integrity and bounded retrieval implementation report

This engineering continuation starts at inspected commit
`55653a53b41603d98b89843057e7f52d3682d950` on local branch
`integrity-retrieval`. The completed study remains an informative negative:
SignedMax did not beat the selected logistic baseline. No real-data confirmation execution,
real-data accuracy evaluation, retuning, model/dataset acquisition or native report
regeneration was performed. Synthetic tests exercise confirmation gates in isolated fixtures.

Implementation code is local commit `67fb038`. Measurements preceded the local
commit: they record base `55653a5`, dirty-diff identity and executed-file hashes.
Every recorded executed-code hash was checked against the delivered implementation.

## Delivered changes

| Workstream | Implementation and regression evidence |
| --- | --- |
| D1 | `__main__.py` freezes the verified effective development receipt. `stage_evidence.py` adds durable selection witnesses, retaining old bytes and failing closed on missing/stale committed selection. `calibration.py`/`stages.py` bind complete cohorts to acquisition/audit/episode evidence, official archive/schema and authorized access fallback. Synthetic CLI integrations cover actual supersession and full 62/225 fallback through freeze. Real parser count-schema and inspection ownership gaps found in review were fixed. |
| D2 | `report.py`, `publication.py` and `render_results_first_study.py` distinguish equivalent paths from different artifacts, historical blocked inspection from target eligibility, and reservations from actual work. Export schemas/identity sets/values are exact. Rendering is staged; retained bundles, selectors and exact terminal G6 ownership define recoverable publication. Tests inject rendering, validation, replacement, pointer, terminal-accounting and process-exit failures, including directory-flush ordering. |
| D3 | `embed.py` builds one exact derived ID index and rejects unsigned int64 overflow before conversion. Existing immutable vector bytes, row ordering/duplicates/errors, raw float32 and float64 geometry remain intact. `data.py` caches digests only for producer/loader-validated immutable records; arbitrary/replaced stores still hash. Audit/permit checks and each actual outcome-file hash/schema remain live. |
| D4 | `costs.py` prepares family/filter/chunk envelopes once; `development.py` and `calibration.py` preserve valid-input arithmetic and accumulation order. Missing durations and inconsistent/zero sizes fail. `workload.py`/`workload_journal.py` retain durable starts, terminals, timings, resources and numerical attempts, geometric partial checkpoints and one final aggregate. Schema-2 replay rejects corrupt transitions, mixed inputs, conflicting records and truncated completion; schema-1 reading remains available. I/O interruption launches no later fits. |
| D5 | `retrieval.py` supplies `GramCatalog`, `TopKResult`, score arrays, raw/model top-k and `score_user_cached`, with bounded allocation, protected public geometry and per-use model validation. Strict top-k always calls the full canonical reference with identical requested order/chunk; explicit fast mode ranks its own float64 scores. Existing pipeline scoring is unchanged. |
| D6 | Executable synthetic/local benchmarks, raw repeat observations, independent summary reconstruction, lookup/workload/forecast profiles and analytic fiber checks replace unavailable attachment scripts. Local mode reads verified original Torch geometry and earlier development histories only. [API and commands](retrieval.md), [mathematical scope](signed-fiber-bounds.md), [workload recovery](workload-evidence.md) and [publication recovery](publication-recovery.md) document the boundaries. |
| D7 | `pyproject.toml` declares directly imported threadpoolctl and the scikit-learn 1.8+ L2 API floor; native pins remain untouched. Read-only POSIX Python 3.11/3.12 CI runs synthetic/public/help/build/wheel checks. `MANIFEST.in` includes synthetic JSON fixtures. Reporting has an explicit saved/checkout/config boundary, tested in a wheel outside the source tree. README/REPRODUCE are updated. |

## Decisions and tradeoffs

No empirical epsilon is used as a ranking certificate. Strict retrieval falls
back on every query to the full reference. Fast mode can disagree at exact or
near ties, including P=N cancellation; scores are never rounded. Tests use
independent scalar oracles with dimension/penalty-aware diagnostics and exact
reference-ranking checks. Adapter validation proves negative support consistency,
not universal encoder identity; the empty-negative limitation is explicit.

The 256 MiB cap covers the Gram matrix alone. At 3,883 items its matrix is
120,621,512 bytes and its float64 catalog is 11,928,576 bytes, before IDs/indexes,
caller arrays and workspace. There is one private Gram matrix; public geometry
has immutable byte backing. Near-full queries use contiguous negative rows plus
a scalar gather. This is an optional reuse tradeoff, not a default scientific
backend or a total-RSS guarantee.

Publication uses retained versioned new/old members, a checksummed durable pending
transaction, selector-last projection and terminal-accounting qualification.
Ancestor directory links are flushed before the pending record. Recovery restores
unfinished owners' old projections or projects an exact completed owner's new
bundle; it never completes the owner or rolls back charged usage. This small
POSIX recovery protocol adds no database or migration framework.

Workload recovery retains complete raw evidence even if final materialization
fails. `complete=True` in replay does not grant successful stage ownership.
Schema-1 journals cannot supply fields their old writer omitted; these remain
unavailable. Final replay currently duplicates the reconstructed evidence in
memory, increasing linear peak working allocation. This is measured explicitly.

Projection screening remains deferred: supplied ranks 16/32/64 were slower, and
their heuristic floating-point pad was not interval arithmetic. ONNX is not a
core dependency or a byte-equivalent substitute. The supplied approximately
4.06×/1.97× ratios and optimizer audit remain inherited reports with unavailable
original raw runs. Kernel caching and safe thresholding are established ideas;
no novelty or predictive-quality claim is made.

## Verification environment and commands

Observed local environment: macOS arm64, Python 3.12.9 in `.venv`, NumPy 2.5.4,
SciPy 1.18.1, scikit-learn 1.9.1, Torch 2.15.0, pytest 9.2.0, psutil 7.2.2 and
threadpoolctl 3.7.0. The build frontend installed for this task is build 1.6.1.
Actual benchmark CPU/BLAS details and environment variables are in each raw run.
Accelerate's actual thread count remains independently unverified.

Substantive release commands executed from the repository root:

```sh
.venv/bin/python -m pip install -e '.[test]'
.venv/bin/python -m pip check
.venv/bin/python -m pytest -q
.venv/bin/python -m signed_study --help
.venv/bin/python scripts/verify_public_results.py
.venv/bin/python -m build --outdir .cache/integrity/dist
git diff --check
```

Tests requiring real process listing/cleanup ran with host process visibility.
The original unmodified revision was extracted to a separate scratch checkout:
**471 passed in 373.04 s**. This is a newly executed baseline, not the frozen
study's historical 462-test proof. The initial restricted run was **391 passed,
46 failed, 34 setup errors** with process visibility denied; its log is retained.

During implementation, source-changing/interrupted runs were diagnostic only.
One stable integration run had **705 passed, 2 failed**: synthetic legacy-adoption
fixtures omitted final score/timing fields required by the stricter reader.
The fixtures were corrected to the real historical schema, without weakening
the decoder; both targeted tests passed. A shared fixture extraction also needed
an explicit import in diagnostics; its isolated suite then passed 16 tests.

The final stable process-visible suite, including the phase-probe regression, passed **707 tests in 266.93 s, with no failures or skips**. Independent whole-branch review and scoped fix review report no unresolved Critical/Important implementation findings. The full logs remain in ignored `.cache/integrity/`; earlier failing/interrupted logs are retained separately.

Pip consistency, help and lazy-import checks, dependency-free public verification,
isolated wheel/sdist builds and `git diff --check` passed. An installed wheel was
imported ahead of the editable package with Python `-I`, outside the checkout;
core/strict retrieval, actionable missing configuration, explicit configuration,
and a saved-config early-stop report passed with Torch unloaded. An initial smoke
fixture incorrectly serialized overridden result paths into the fixed config;
using original saved config bytes corrected the fixture. Hosted CI has not run
in this session; Python 3.11 and Linux execution remain CI validation targets.

## Reproducible performance evidence

All following profiles are synthetic and retain raw observations. They have no
recommendation-quality interpretation:

| Boundary | Before | After | Scope |
| --- | ---: | ---: | --- |
| 16-ID lookup, 3,883 rows | .248201 ms | .003225 ms | Lookup/gather only, repeat medians |
| Synthetic K2/λ2 fit+score | 6.594250 ms | 4.344958 ms | One complete support-excluded synthetic query workload |
| Full 2,194-episode digest | 67.508 ms | .000075 ms | Cached validated record identity; 13.420 ms one-time structural validation |
| 64-user cumulative snapshot rows | 2,144 | 191 | One-entry fixed-boundary workload |
| Same snapshot bytes | 12,947,579 | 1,155,604 | Actual serialized temporary snapshots |
| Same instrumented bookkeeping | 1.834569 s | .289283 s | Fake fit/score/clock, real metrics/fsync, cProfile included |
| Same traced peak working bytes | 2,554,347 | 4,682,818 | Separate memory run; final replay copy increases linear memory |

At 16/32/64 episodes the forecast profile reduced fit-bin row visits from
2,304/9,216/36,864 to 144/288/576. Preparation-inclusive medians were
2.607/9.537/37.318 ms before and 3.650/7.192/14.836 ms after: small inputs lost
time to setup. Whole fixed-clock workload dictionaries matched exactly, as did
six complete original/new forecast outputs including nextafter budget boundaries.
New measured overhead can legitimately change prospective fresh forecasts; no
historical forecast or measurement was replaced.

Raw artifacts and executable reconstructions:

- [Identity observations](identity-profile.synthetic.json): `scripts/profile_identity_lookup.py`.
- [Workload observations](workload-profile.synthetic.json): `scripts/profile_workload_integrity.py`.
- [Forecast observations](forecast-profile.synthetic.json): `scripts/profile_forecast_costs.py`.
- `tests/fixtures/forecast-boundaries.json` records complete forecast digests from the exact original source.

Both final runs executed on Apple M3 Max, macOS 26.7.1 arm64, after the test/build work stopped. Numerical thread environment variables were set to 1 and supported observed pools were limited to 1; Accelerate thread count was not independently measured. Each run has 1,920 query observations, 1,920 separate phase probes and 384 correctness checks. Both retained every scheduled user/method/repeat, reported no exceptions, and verified unchanged executed sources.

| Input / candidates | Dense ms | Gram strict ms | Gram fast ms | Median paired dense/fast | Build amortization queries |
| --- | ---: | ---: | ---: | ---: | ---: |
| synthetic / full | 2.4426 | 3.3527 | 1.9844 | 1.249× | 134.3 |
| synthetic / capped_support_excluded | 2.3125 | 3.2344 | 2.0269 | 1.138× | 223.9 |
| torch-local / full | 2.2862 | 3.2378 | 1.9230 | 1.189× | 172.2 |
| torch-local / capped_support_excluded | 2.1761 | 3.1280 | 1.9971 | 1.100× | 324.5 |

| Input / candidates | score_user + rank ms | Cached adapter + rank ms | Median paired adapter ratio |
| --- | ---: | ---: | ---: |
| synthetic / full | 3.7180 | 1.8654 | 2.009× |
| synthetic / capped_support_excluded | 3.6276 | 1.9335 | 1.866× |
| torch-local / full | 3.5831 | 1.7994 | 2.007× |
| torch-local / capped_support_excluded | 3.4833 | 1.8827 | 1.865× |

Strict mode is slower than the optimized dense baseline here; it used the full reference for all 320 timed strict queries per run. There is no ambiguity classifier or formal separation gate. Fast mode ranks its own scores. These 384-dimensional runs had zero top-ten mismatches, zero exact kth-boundary ties and maximum observed score error 0.0; this does not establish universal floating-point ranking equivalence. Cancellation/near-tie tests cover the contrary general case.

synthetic: preparation 0.012150 s, shared catalog construction 0.063032 s, median prototype fit 1.1251 ms; encoder inference 0 (absent). Traced build peak 144,938,551 bytes; traced fast query workspace 3,279,920 bytes. RSS before/after construction 167,886,848/302,350,336 bytes; these are endpoints, not RSS peaks.
torch-local: preparation 0.018610 s, shared catalog construction 0.061480 s, median prototype fit 1.0532 ms; encoder inference 0 (absent). Traced build peak 144,938,635 bytes; traced fast query workspace 3,282,616 bytes. RSS before/after construction 156,532,736/290,095,104 bytes; these are endpoints, not RSS peaks.

The local Torch full-catalog component probes below are medians across users' repeat medians. Their retained chunks have different cache/allocation behavior from streaming APIs; the phase totals need not equal comparable public API time.

| Probe method | Validation/gather ms | Prevalidated score ms | Ranking ms |
| --- | ---: | ---: | ---: |
| reference | 1.4352 | 0.9483 | 0.0695 |
| dense | 1.4263 | 0.7605 | 0.0674 |
| gram_fast | 1.0607 | 0.5904 | 0.2542 |
| gram_strict | 1.9727 | 0.9164 | 0.2646 |
| adapter_reference | 2.6478 | 0.9391 | 0.0714 |
| adapter_cached | 1.1656 | 0.5897 | 0.0676 |

[Synthetic raw observations](retrieval-benchmark.synthetic.json) and [verified local Torch raw observations](retrieval-benchmark.torch-local.json) contain all query/probe repeats, model-fit durations, memory, ordered input digests and exact executed-file hashes. `scripts/summarize_retrieval_benchmark.py` reconstructed both saved summaries exactly. Real user/support/candidate ID mappings and arrays remain only in ignored `.cache/integrity/local-retrieval-final.npz`.

An earlier simple cache prototype was slower than dense; its scratch observations remain in `.cache/integrity/retrieval-synthetic-first.json`. The contiguous-row strategy improved this bounded reuse workload. Those development-stage scratch runs are not presented as independently reconstructable release results. Final benefits are smaller than the supplied attachment's local 4× figure, and no whole-study speedup is inferred.

Retrieval methods include validation/gather/scoring/ranking under their actual
API names. Model identity checking is inside `adapter_cached`; direct-formula
speedups are not attributed to it. Correctness/workspace calls precede the one
declared warmup; five timed repeats rotate method order. Ratios pair users'
repeat medians. Amortization divides build seconds by mean paired savings of
those medians, with no positive estimate when savings are nonpositive. RSS
endpoints and traced allocation peaks are separate quantities.

Each method/workload also has five rotated component probes after its comparable
API repetitions. Validation/gathering, prevalidated scoring and ranking have
separate raw durations. Their retained prepared chunks differ from the streaming
API's allocation/cache behavior; the probes are not subtractive accounting or
replacement API timings. Their numerical outputs are checked against each actual
method, and probe failures stay in the evidence. Adding this explicit separation
required one final harness regression after the first 707-test green run.

## Scientific preservation and remaining verification limits

The dependency-free public verifier passed before and after implementation: 13 methods, 12 paired comparisons, all 1,776 users and 23,088 confirmation evaluations, informative negative. All 14 original tracked documentation/scientific-evidence/native-lock files captured at setup retained their exact hashes. A separate read-only snapshot of 17,290 preexisting native result artifacts before local benchmarking also retained every hash afterward. No native packaging command ran against the active campaign.

No historical lock, G1 evidence, exposure record, model cache or public scientific
aggregate was replaced. Changed source intentionally invalidates old current-source
authorization; historical reading remains available. The public verifier checks
aggregate consistency/hashes, not independent predictions or bootstrap inference.

Required implementation is delivered subject to the final results above. Hosted
CI, Python 3.11/Linux execution, a full fresh installation on every supported
platform and private native report regeneration are unverified. The native lock
remains environment-specific. No fresh confirmation experiment or projection
backend was attempted. These are separate scope/verification limits, not omitted
accuracy claims or fabricated successful runs.
