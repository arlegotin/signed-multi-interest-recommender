# Workload evidence and recovery

New workloads retain schema-2 append-only journals and the existing schema-1
final aggregate. A fresh workload refuses an existing aggregate, partial file or
journal; interruption never authorizes an implicit retry. Every actual evaluation
has a flushed start before fitting/scoring, then terminal evidence. Supplemental
events retain write timing, per-entry measurements, finalized user overhead and
resources. The final journal digest precedes atomic aggregate materialization.

Partial checkpoints use `<name>.partial.json` at powers of two and explicitly
describe incomplete work. They never replace an older final aggregate. This
changes cumulative snapshot volume from quadratic to linear in evidence produced.
The final replay currently holds a second reconstructed evidence copy; measured
traced allocation increases proportionally to evidence volume. Resource guards
remain active, and neither the snapshot policy nor the Gram cap promises total RSS.

Inspect a retained journal without outcomes, inference or new accounting:

```python
from signed_study.workload_journal import replay_workload_journal

recovered = replay_workload_journal(path, expected_input_sha256=known_input_digest)
print(recovered['complete'], recovered['measurements']['actual_evaluations'])
```

When available, pass `expected_config=config_record(resolved_config)` too. Replay
validates the bound ordered users/configurations, candidate digests, chunk policy,
exact config, starts/terminals, numerical-repeat schedule, timing/metric schemas
and final consistency. Malformed complete/internal rows and conflicting duplicates
fail closed. A final append without newline remains incomplete, including otherwise
valid JSON. An unpaired start remains interrupted/unknown, never completed.

Schema-1 journals still accept identical or enriched repeated terminal records.
They explicitly report unavailable timing/resource fields that the old writer did
not retain. Replay does not invent those fields or automatically rewrite old files.
New records are limited to 16 MiB each; readers enforce that bound while streaming.

`complete=True` proves a sealed raw materialization, not successful scientific stage
ownership. A failed final rename may leave complete raw work with a failed ledger
owner. Preserve the raw evidence, original receipts and actual/reserved accounting;
use existing verified stage adoption only when its gates authorize it. No recovery
command reruns fits, reverses charged attempts or makes failed work selectable.

Reproduce the synthetic before/after bookkeeping profile:

```sh
python scripts/profile_workload_integrity.py \
  --output .cache/integrity/workload-profile.json --repeats 5
python scripts/profile_identity_lookup.py \
  --output .cache/integrity/identity-profile.json --repeats 5
```

The workload script reads the original implementation from Git commit `55653a5`,
uses injected fit/score/resource/clock boundaries and real synthetic metrics, and
compares entire final dictionaries exactly. Timings include profiling overhead and
actual temporary journal/snapshot writes. They do not estimate a scientific
campaign speedup. Raw public observations are in `workload-profile.synthetic.json`
and `identity-profile.synthetic.json` beside this note.
