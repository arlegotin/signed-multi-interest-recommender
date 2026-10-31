# Study integrity and bounded retrieval implementation plan

> **For agentic workers:** Use superpowers:executing-plans and test-driven development. Check off verified deliverables below; the user's continuation directive is the binding specification.

**Goal:** Repair workflow/publication integrity, remove repeated work, and supply optional bounded retrieval with reproducible evidence.

**Architecture:** Retain the source-bound scientific workflow and historical readers. Use local prepared indexes/cost summaries, durable versioned evidence, and an independent retrieval module whose default ranking calls the reference scorer. Benchmark only synthetic or explicitly verified local development geometry.

**Tech stack:** Python 3.11–3.12, NumPy/SciPy, existing pytest and setuptools.

**Spec:** User-provided continuation, sections A–J, 2026-10-31. It overrides this execution outline.

## Global constraints

- Preserve the informative-negative result, historical scientific hashes, lock and G1 evidence; never run confirmation or retune on outcomes.
- No downloads, remote publication, native campaign writes, or replacement of historical measurements.
- Preserve exact integer identity, canonical score/tie semantics, full cohorts, provenance gates, actual/reserved accounting and single-writer durability.
- Test behavior before fixing it. Fail closed on corrupt evidence; preserve previous valid publications.
- Report actual environment results; unavailable private assets or hosted CI remain unverified.

## Review focus

- Interrupted selectors and publications must not reactivate obsolete receipts or imply terminal success.
- Fallback authorization must bind acquisition, audit, immutable episodes and ordered cohort membership.
- Malformed journal state transitions and torn tails must preserve started attempts without inventing completion.
- Cancellation/near ties and mutable input arrays must not compromise retrieval's strict reference contract.
- Installed reporting must fail clearly outside a checkout when configuration/source identity is unavailable.

### Task 1: D1 effective receipts and fallback

- [x] Reproduce CLI legacy receipt read and implement verified effective resolution.
- [x] Bind full fallback cohorts to official acquisition/audit/episode evidence; reject substitutions and malformed ordering.
- [x] Add integration regressions for legacy, superseded, deleted/corrupt selectors, interrupted publication and freeze.

### Task 2: D2 trustworthy reports and publication

- [x] Verify path-equivalent references with digest and G0 terminal ownership; distinguish blocked inspection from target eligibility.
- [x] Require exact CSV schemas, identities, finite values and work-evidence counts.
- [x] Stage and validate prospective output; preserve/recover prior publication across injected failures and terminal accounting.
- [x] Retain scientific bytes; test with synthetic fixtures, never the private campaign.

### Task 3: D3 immutable lookup and validation

- [x] Test exact integer boundaries, API compatibility and caller mutation isolation; build a per-cache index once.
- [x] Profile episode hashing; cache only a validated structurally immutable context if its invalidation proof is simple.
- [x] Keep each outcome file's hash/schema checks and authorization boundary.

### Task 4: D4 forecasts and durable workload evidence

- [x] Record fixed-input scan/serialization scaling; preaggregate exact cost envelopes while preserving arithmetic order and decisions.
- [x] Publish near-linearly while durably retaining starts, terminals, timings/resources and partial evidence.
- [x] Version replay and bind workload/evaluation identity; validate corruption, duplicates, interruption and recovery.

### Task 5: D5 optional Gram backend

- [x] Add independent-oracle tests for exact IDs, score semantics, strict ranking, ownership, model mutation and bounded allocation.
- [x] Implement GramCatalog, TopKResult and score_user_cached; strict top-k uses full canonical call; explicitly opt-in fast mode ranks cache scores.
- [x] Keep canonical pipeline unchanged; reject unsupported or nonfinite queries and expose separate memory costs.

### Task 6: D6 reproducible measurements and research note

- [x] Add offline synthetic and verified local catalog benchmarks with source/input/runtime identity and raw rotated-repeat observations.
- [x] Reconstruct paired-ratio/amortization summaries independently; name direct versus adapter timing boundaries.
- [x] Run fresh synthetic workloads, record costs/memory/correctness, and document theorem, attribution and limitations.

### Task 7: D7 developer/release gates and final review

- [x] Document supported dependencies and directly imported packages; preserve native lock.
- [x] Add read-only Python 3.11/3.12 synthetic CI, public verification, lazy CLI and wheel/sdist checks.
- [x] Test narrow installed-package reporting/config contract outside checkout.
- [x] Run full suite, pip check, help, public verification, build and diff checks; independently review and fix material findings.

## Execution rulings

- The detailed user directive explicitly authorizes execution and asks to avoid further planning/approval. No extra design-approval round is needed.
- Use the authorized local branch `integrity-retrieval`; retain the user's clean main and all native result paths.
- Independent D1/D2/D3 investigations may run concurrently under dispatching-parallel-agents, with disjoint file ownership. D4 calibration changes wait for D1. No competing performance runs.

## Final validation

The process-visible full suite passed 707 tests in 266.93 s, with no failures/skips. Both final 32-user synthetic/original-Torch benchmarks retained 1,920 API observations and 1,920 phase probes, source unchanged, no failed methods or mismatches. See docs/research/implementation-report.md for exact commands, raw artifact links, numerical/publication decisions, profiles and unverified hosted/native-regeneration limits. No remote publication or scientific confirmation ran.
