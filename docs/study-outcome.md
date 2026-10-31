# Native study outcome: forecast stop; efficacy untested

The declared executable pipeline failed its prospective time-admission gate at
G2. All six permitted cohort sizes failed, including the smallest 256 development
/ 1000 confirmation schedule. The recommendation idea remains **untested**:
there is no selected S* or B*, final scale, frozen cohort, protocol lock,
confirmation result, paired interval or estimate relative to delta=.01. This
does not show that SignedMax is ineffective or computationally infeasible as a
method. It shows that the declared conservative pipeline forecast did not fit
the allocated time. Actual wall-time caps were not exhausted.

The study used official MovieLens 1M and the fixed
`sentence-transformers/all-MiniLM-L6-v2` representation on CPU. Native inspection
recorded Apple M3 Max, arm64, 36 GiB RAM, 14 logical CPUs and Python 3.12.9.
Encoder inference used float32, 384 dimensions, maximum length 256, one worker
and observed torch/OpenMP thread counts of one; geometry and metrics use the
float64 protocol. The full audit found 2194 eligible users: 418 development and
1776 confirmation. Auditor eligibility access is separate from calibration,
which read labels for only 32 development users. No confirmation outcomes were
opened by selection, confirmation or diagnostics.

G0 acquisition/audit and current G1 correctness were completed; the saved full
proof passed 413 tests in 250.32 seconds after final scientific source/input
commits. G2 encoding and numerical workload measurements completed, but its
forecast gate failed. Smoke, G3 tuning and endpoint expansion, selection,
precision/history readiness, freeze, G4 confirmation and G5 diagnostics were not
admitted. No accuracy or component-attribution conclusion follows.

Measured computation was small relative to the forecast:

| Actual G2 work | Seconds |
| --- | ---: |
| Cold model load | 3.887462792 |
| Full 3883-item catalog encoding | 11.186896917 |
| 352 fits | 1.334811540 |
| Canonical candidate scoring plus metrics | 0.500764263 |
| Repeated scoring plus metrics | 15.186073703 |
| Numerical comparison | 0.451780395 |
| Unattributed pipeline/write/resource/forecast/finalization residual | 44.414098640 |

The last five rows partition the 61.887528541-second calibration workload.
Cold load and catalog encoding have separate phase scopes and must not be added
to that partition as though they were its components. The verified shared cache
is 3883×384; raw repeat maximum difference was zero at tolerance 1e-5 and chunk
checks 1/7/512 passed. There were 1056 completed evaluations: 352 fit-and-score
evaluations and 704 scoring repeats. No evaluation failed or requested a
solver/prototype repair.

For the unexecuted smallest schedule, the saved prices were:

| Stage | Raw forecast seconds | 1.5× safety seconds | Effective allocation seconds |
| --- | ---: | ---: | ---: |
| G3 | 51995.444463 | 77993.166694 | 2400 |
| G4 | 10892.985989 | 16339.478983 | 2100 |
| G5 | 4462.020821 | 6693.031232 | 1200 |

G3's raw forecast comprises fit 224.723507, scoring/metrics 437.338386 and
**write/pipeline overhead 51333.382570 seconds**. This last term is the dominant
conservative upper envelope, not a measured scaled campaign or an isolated I/O
measurement. The forecast uses upper size-bin fit costs and nonnegative slopes,
maximum measured scoring rates and maximum family per-entry overhead. The
measured residual is unattributed; it does not establish which pipeline activity
caused that envelope. Canonical calibration used chunk512; the final failed
forecast policy used chunk7, which was not admitted as a primary scoring policy.

Raw future forecast was 67350.451273 seconds; safety forecast 101025.676909;
original terminal-plus-future forecast 104359.023177 versus protected work
allowance 12600 (14400−1800). Each stage independently violated its cap.
Projected future 40376 plus actual 1056 evaluations would total 41432/100000,
so evaluation count was not binding. The unexecuted G5 full-catalog timing
forecast alone was 4112.387355 seconds; it is distinct from the measured
11.186897-second catalog encoding. No G5 benchmark was run.

Before G6 packaging, permanent experimental time was 3473.892128874211 seconds:
G0 259.630140994501, G1 3069.812894503657, G2 99.623294709134 and repair-stage
44.825798666920. G3–G6 were zero at that historical snapshot. Separate setup was
77.582822208863/2700 seconds. Two material repair cycles were used; cumulative
repair attribution was 269.685277538257/1200 seconds, already included in
experimental accounting rather than added again. Exactly five logged transfers
left G1=3300 and G4=2100 seconds, allocated stage sum=13800, global=14400 and
reporting reserve=1800, with no resets or unfinished attempts. Pre-G6 peak
process-tree RSS was 662913024 bytes and project disk 3996961946 bytes, baseline
308236 bytes. Reporting has real cost: later costs and precise as-of ledger
prefixes are in `results/report.md`, `results/costs.csv`,
`results/outcome-reproduction.json` and the permanent ledger.

The two material cycles corrected native resource-monitor/implementation defects
and exact durable manifest publication respectively. Original failures, failed
proofs and failed arithmetic crosschecks remain retained. Exact installed-pin
validation, pip checks and editable dry-runs do not establish a fresh clean-room
rebuild. Native torch redirect/API warnings did not cause the stop. The saved
arithmetic diagnostic was replayed from original costs and the terminal G2
ledger prefix, without a new model run, outcome access or changed policy.

Scientific source/input HEAD was
`08bc0ac467544c03569a3ff74418df23f36181b1`; native manifest fix was
`2a123a9e00a7abb59109c5ba833331a071f1d455`. Provenance anchors:

| Object | SHA256 / revision |
| --- | --- |
| Official ML1M archive | `a6898adb50b9ca05aa231689da44c217cb524e7ebd39d264c56e2832f2c54e20` |
| Encoder revision | `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` |
| Environment evidence | `e5efe69820351624ac9e66a6d5ec974c853d8e94cb7060e676ee392563f9abba` |
| Exact requirements lock | `db25791dbf1f2c8fcd388912246ce493193da8e710fc9d650342cda594f133eb` |
| Scientific code content | `7b9a530e0ba8fefc3b78810243dbc879d37b2be6baaab66975d955a10cf109e8` |
| Calibration publication | `228966ad7ca82552ac772e9fa414c5b2e88f0f1696c04d007378091d63d2d0e4` |
| Saved forecast diagnostic | `740422f895ad2696747ee525ed7a4ce441a5dcd796e4aca971d8ef6e94b0d874` |

There is no lock digest because no freeze occurred. Raw/derived MovieLens
material, outcomes, vectors, per-item scores and local run artifacts remain
ignored. Tracked summaries contain aggregates and hashes only. Observed-rating
AUC would not establish online satisfaction, universal representation failure,
missing-is-dislike or priority over prior work even if it had been measured.

From the repository root, preserve local evidence at its bound paths and run:

```sh
.venv/bin/python -m signed_study report --results results
.venv/bin/python scripts/render_study_outcome.py --results results
```

The base CLI alone retains generic wording limitations. The tracked G6 wrapper
verifies the immutable diagnostic and referenced calibration/workload, then
corrects and supplements all ten report sections with measured and forecast
tables/figures. It does not fit, encode, run stages, open outcome arrays or change
the experimental decision rule. Repeating the wrapper checks identical scientific
table/figure hashes; cost snapshots change because reporting extends the ledger.
`results/artifact-inventory.json` records generated hashes, recipe, script digest
and an exact ledger byte prefix. Neither the recipe nor the report depends on
the temporary workflow workspace. See [README](../README.md) for prerequisites
and [the complete decision history](study-decisions.md) for historical rulings,
superseded budgets and repair policy.

The one justified next step is to audit overhead measurement/scaling and the
evidence pipeline before another run of the same scientific question.
