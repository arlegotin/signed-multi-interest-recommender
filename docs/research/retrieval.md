# Optional Gram retrieval and reproducible engineering evidence

`signed_study.retrieval` is independent of the frozen study pipeline. Existing
`score_user`, confirmation, model grids, encoders, bootstrap inference and public
accuracy aggregates remain unchanged. SignedMax did not beat the selected logistic
baseline. Evaluating the same score faster is not a recommendation-quality gain.

## API and numerical policy

```python
from signed_study.retrieval import GramCatalog, score_user_cached

catalog = GramCatalog(item_ids, unit_float64_vectors)  # no acquisition or encoding
rows = catalog.rows([item_ids[3], item_ids[0]])
scores = catalog.scores(P, negative_item_ids, 2, candidate_ids=candidate_ids)
strict = catalog.topk(P, negative_item_ids, 2, k=10, candidate_ids=candidate_ids)
fast = catalog.topk(P, negative_item_ids, 2, k=10, candidate_ids=candidate_ids,
                    mode='fast')
cached_model_scores = score_user_cached(model, candidate_ids, catalog)
strict_model = catalog.topk_user(model, candidate_ids, k=10)
```

The raw-query reference is `models.signed_scores` on the **full requested
candidate order**, same geometry and declared `chunk_size` (default 512). The
model reference is `score_user` with that same model, IDs, geometry and chunk.
Strict top-k always uses the full reference call, then descending exact stored
score/ascending integer ID. There is no floating-point separation certificate,
reduced-subset rescore, score rounding or tolerance-based tie rule. Strict mode
therefore makes no cache-ranking speed claim and reports `reference_fallback=True`.

Explicit fast top-k ranks its own cached float64 scores with the same deterministic
tie rule. Its real-arithmetic formula is identical, but reduction shapes can
change float64 values and rankings. `P=N` cancellation can be exactly zero in the
reference and differ by a few ulps in a Gram calculation. `allclose` does not
certify reference ranking. Diagnostic test/benchmark tolerances scale with
dimension and penalty; they are numerical checks, not interval certificates.

Scores preserve negative cosines, empty negatives and λ=0 positive-only behavior.
No negative mean is normalized. Input IDs are exact int64 integers, never floats;
score queries preserve arbitrary order and duplicates, while top-k requires
distinct candidates and 1≤k≤candidate count. Empty score arrays are supported.
Rows must be finite unit vectors within norm tolerance 1e−12; penalties must be
finite nonnegative float64 values. Computed nonfinite scores are rejected.

The adapter supports successful SignedMax, AllExemplarMargin, PositiveMax and
Centroid models. Every call validates current family parameters and mutable model
state, infers the final `len(N)` support IDs and requires exact stored negative-row
equality. Other families use the existing scorer. This proves negative-support
consistency, not universal encoder identity; with empty negatives there are no
negative rows to compare. `catalog.identity_sha256` binds ordered IDs, geometry and
an optional supplied manifest digest. The original model record receives no new
mandatory fields and no validation is cached by object identity.

## Ownership and construction costs

The catalog owns immutable byte-backed geometry/IDs and a derived exact index.
Public rows are independent copies; public array views cannot simply have their
write flags re-enabled. The Gram matrix is private and never exposed as a public
view. Python private state is not an adversarial security boundary.

Matrix allocation is checked with Python integer arithmetic before quadratic
allocation. The default 256 MiB limit applies **only to N²×8 bytes**. For 3,883
items that is 120,621,512 bytes; catalog float64 geometry adds 11,928,576 bytes,
plus IDs/indexes, callers' arrays, query and BLAS workspace. Build blocks write
directly into one matrix, without a second full Gram copy. The benchmark records
Python/NumPy allocation peaks and RSS endpoints separately; RSS endpoints are not
total-process peak promises. Refusal raises `MemoryError`; use dense scoring when
construction or quadratic storage is inappropriate.

Fast scoring bounds positive matrix products by query/prototype chunks and reads
negative matrix rows without candidate×support×dimension tensors. At ≥75% catalog
coverage it reduces contiguous full negative rows into one scalar catalog array,
then gathers requested scores. Smaller subsets read only requested columns. The
threshold is a local engineering policy, not a universal hardware optimum.

## Fresh execution

Use the supported virtual environment from `REPRODUCE.md`. These commands are
offline and need no external assets:

```sh
python -m pytest -q tests/test_retrieval.py tests/test_retrieval_benchmark.py
python scripts/benchmark_retrieval.py --output .cache/retrieval/synthetic.json
python scripts/summarize_retrieval_benchmark.py .cache/retrieval/synthetic.json
python scripts/profile_forecast_costs.py
python scripts/audit_signed_fiber_bounds.py
```

The default benchmark uses 3,883 synthetic 384-dimensional rows, 32 histories,
fixed K=2/λ=2, top ten, one warmup and five timed repetitions. It rotates six
methods per user/repeat and limits observed numerical pools to one thread:
validated reference, exhaustive dense, Gram strict, Gram fast, canonical model
adapter and cached model adapter. API validation/gathering/scoring/ranking are
inside query timing. Preparation, fitting, build, correctness and workspace
observations are separate. Shared encoder inference is absent.

Five additional rotated component probes follow each comparable workload. They
separate validation/gathering, prevalidated score computation and ranking, and
check their score arrays against the actual method. Prepared probe chunks are
retained, so their allocation/cache behavior differs from streaming public APIs.
Their times are diagnostic boundaries, never subtracted from query timing or
used to claim that the public API omits validation.

Optional local mode reads an **already verified original Torch cache** and only
earlier development histories. Preparation verifies the existing acquisition and
audit evidence; it neither downloads assets nor opens outcome arrays:

```sh
python scripts/prepare_retrieval_benchmark.py --config config/study.json \
  --output .cache/retrieval/local.npz
python scripts/benchmark_retrieval.py --local-bundle .cache/retrieval/local.npz \
  --output .cache/retrieval/local-timing.json
python scripts/summarize_retrieval_benchmark.py .cache/retrieval/local-timing.json
```

The preparation output contains private ID/history mappings and must remain
Git-ignored. Public timing artifacts retain only opaque ordinals, counts and
digests. Full-catalog and **capped fitted-support excluded** workloads are separate;
the latter does not exclude every previously seen item. Failed methods/users stay
in the schedule/denominator. Source commit, dirty diff and executed-file hashes
(including untracked scripts), ordered workload digests, runtime/pool evidence,
raw repeat timings, exceptions and correctness observations are saved. Source
changes during execution invalidate stable-source status. Summaries use median
paired user ratios and mean paired savings; nonpositive savings have no positive
build-amortization estimate. No outcome accuracy is computed.

## Evidence interpretation

Fresh measured artifacts and final verification results are indexed in
[the implementation report](implementation-report.md). Raw synthetic timings can
be independently summarized with the command above. Local asset-based results
are complete only if listed as executed there. Hosted CI and native report
regeneration are separate verification claims.

The supplied attachment's approximately 4.06× full/1.97× support-excluded paired
ratios motivated this work. Those raw repeat observations were unavailable and
have not been fabricated. Its encoder was an independent ONNX Runtime
implementation (reported ORT 1.31.1/tokenizers .23.3/NumPy 2.5.4 preparation and
NumPy 2.3.5 Linux timing), rather than the original Torch byte stream. Its 0.286 s
build and 73/128-query amortization estimates are supplied historical local
measurements, not reproduction targets or pipeline promises. Projection ranks
16/32/64 were slower than dense. This integration keeps projection screening
deferred. [The mathematical note](signed-fiber-bounds.md) retains the theorem,
attributions, analytic examples and hypotheses for a separate study.
