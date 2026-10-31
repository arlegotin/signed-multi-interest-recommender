# Study execution decisions

This is the chronological controller decision history for the implementation
and bounded native run. The line numbers refer to the original execution ledger,
preserved with the complete workflow archive in ignored
`results/operator_logs/workflow/`. They identify historical decisions, not steps
to repeat. Earlier allocations and interim statuses were superseded. The final
state is five transfers, G1=3300 seconds, G4=2100 seconds, two material repair
cycles, and a G2 forecast stop. [The outcome](study-outcome.md) separates that
operational stop from the unanswered efficacy question.

1. **Ledger 7 — use the existing main checkout.** The user required the current
   branch. Work therefore used ordinary reversible source commits on main; the
   cost of a wrong interpretation would have been changes on that branch.

2. **Ledger 8 — execute the approved study.** Earlier documentation-only wording
   described the preceding task; the latest user instruction authorized
   execution. If misread, implementation would have exceeded scope.

3. **Ledger 9 — distinguish development from material repairs.** Normal synthetic
   TDD iterations counted as development, with actual test time charged to G1;
   source-editing idle time was separate. Misclassification would require
   conservative reallocation or an early stop, never a clock reset.

4. **Ledger 10 — retain authorized global pre-freeze repairs.** The blueprint
   permits global solver/prototype corrections recorded in resolved settings.
   Fixed-default validation must not accidentally forbid those corrections.
   An incorrect interpretation would be an explicit deviation blocking an
   unqualified scientific claim.

5. **Ledger 11 — describe the label firewall honestly.** It is a verified
   application boundary, with auditor eligibility access allowed by the spec,
   not filesystem isolation. Claiming a stronger security boundary would be
   wrong; no such claim is made.

6. **Ledger 131 — validate only listed iteration alternatives.** Config accepts
   blueprint-listed global iteration limits with explicit repair records.
   Stages still enforce calibration, pre-freeze timing and budget conditions.
   A mistaken allowance would be a material deviation requiring qualified
   confirmation, not permission for silent tuning.

7. **Ledger 138 — define the Task 2→10 access contract.**
   `VerifiedProtocol.payload["data_access"]` carries confirmation membership
   and `episodes_sha256`. `episode_store_digest` hashes stable model-facing
   user/role/cutoff/support/candidate identifiers only. The confirmation loader
   checks record type, schema, membership and store digest; Task 10 checks live
   provenance before issuing the record. Public record construction is not
   unforgeable. A contract error would require integration rework without
   changing the scientific design.

8. **Ledger 139 — keep synthetic provenance out of production.** Synthetic
   datasets are valid only in tests. Production admission rejects them.
   Expected-count discrepancies retain expected and actual counts plus a
   cause/resolution artifact; silent progression is forbidden. Otherwise cohort
   evidence would be invalid.

9. **Ledger 145 — allow read-only snapshots during active work.** A constructor
   or snapshot can inspect an existing active ledger because restart tests need
   that behavior. Single-writer exclusion still controls mutations and admitted
   workloads. Concurrent workloads would invalidate accounting.

10. **Ledger 153 — preserve the Task 3 review checkpoint.** Fix round 1/5 was
    active on the original Task 3 implementer at base `1862037`. The controller
    required its report and covering test evidence, then a scoped review, before
    Task 4. Tasks 1 and 2 were complete and must not be repeated. The user had
    authorized all 14 tasks, subagent execution and meaningful commits on main;
    routine work did not need renewed permission. This was an interim workflow
    checkpoint, not a new scientific or budget rule.

11. **Ledger 159 — reject undefined ratio histories.** `ratio_scores` requires
    nonempty positive and negative references and otherwise raises an
    ineligible-history error. `signed_scores` retains its documented
    positive-only fallback when negatives are empty. Distance ratio has no
    empty-class definition and the production cohort is signed. A mistake would
    require API rework, not a changed eligible population.

12. **Ledger 166 — pass repair policy explicitly.** The shared interface is
    `fit_user(method, history, embeddings, parameters, *, config=None)`.
    Production stages pass resolved `StudyConfig`; pure fixtures can use nominal
    defaults. Diagnostics record effective limits. This avoids hidden global
    state; a mismatch would require integration rework before freeze.

13. **Ledger 173 — label complete-case summaries as descriptive.** Export mean
    difference (unavailable at zero users), standard error (unavailable below
    two), count and `complete_case_diagnostic`. A separate complete-case CI or
    threshold decision is unnecessary because the full scheduled-user envelope
    is primary. A wrong reading would require a report extension, not changed
    primary analysis.

14. **Ledger 186 — permit focused persistence and test modules.** Task 8 could
    add narrow episode/configuration persistence helpers and focused calibration
    tests rather than enlarging one test module. Resume needs model-facing
    loaders and the plan permits responsibility splits. The risk was reversible
    layout rework; scientific contracts were unchanged.

15. **Ledger 187 — separate reserved quota from actual work.** Batch reservations
    conservatively consume quota; attempt records establish actual execution.
    Reports must distinguish reserved or unperformed units from fits. Otherwise
    accounting would be ambiguous and evidence overstated, without changing
    outcome selection.

16. **Ledger 190 — fail closed on calibration repair requests.** Task 8 records
    structured `repair_required` evidence rather than retrying users silently.
    The controller may apply an authorized global pre-freeze repair, persist
    resolved policy and allowance use, then rerun affected calibration. This is
    not an extra human approval gate. If the rule is wrong, stop early or qualify
    the result; do not hide retuning.

17. **Ledger 194 — acquire provisional exact pins, then validate them.**
    Acquisition can emit observed installed requirement versions provisionally;
    G1 must validate those inputs before production calibration. Missing pins
    may be reported in implementation fixtures, never accepted silently for
    production. Acquisition identity excludes authorized scoring/iteration
    changes, while G1 and calibration bind the resolved current policy. The
    recorded ruling supplies no separate numerical cost-if-wrong estimate.

18. **Ledger 206 — share bounded evaluation without changing defaults.** Task 9
    could generalize workload evaluation with explicit declared entries, focused
    development/selection modules and a public verified-context wrapper. This
    reuses durable failure/evidence semantics and keeps orchestration thin.
    A mistaken abstraction would require reversible pre-freeze API/layout work.

19. **Ledger 209 — measure the complete endpoint-expansion request before
    admission.** A separately admitted timing-only pilot uses the existing
    history-based sampler restricted to the fixed development cohort. Count all
    time/evaluations and retain diagnostics, without pilot-AUC selection.
    Expansion stays inadmissible until the complete remaining measured campaign
    fits. Log every unavailable request; pilot configurations remain ineligible
    until admission. Cached scores may be reused only after complete admission,
    with provenance and no double charge. Preserve request order and ceilings.
    This prevents unsupported cross-parameter cost bounds and partial tuning;
    a wrong choice requires an explicit coverage/deviation qualification.

20. **Ledger 210 — retain the endpoint RED checkpoint.** The Task 9 selection
    test reproduced missing selection; the original implementer was adding the
    measured-pilot path under ruling 209. This is historical implementation
    status, not evidence that real endpoint expansion ran.

21. **Ledger 214 — add logged allocation transfers before Task 10.** After
    independent acceptance, transfer 600 seconds from unstarted G4 to unfinished
    G1: G1=1200, G4=4200. Stage sum 13800, global 14400, setup 2700, reporting
    reserve 1800 and all other ceilings stay fixed. Required validation needed
    allocation before confirmation began. An error would be an admission or
    accounting defect requiring tests and review before use. This was transfer 1.

22. **Ledger 216 — accept Task 9, preserve freeze requirements.** Commits
    `07ff1eb..10840a7` passed scoped review under the pilot ruling, with no
    findings or probes. Freeze must still enforce chosen/control coverage,
    provenance, feasible forecasts and the outcome boundary. Acceptance did not
    establish real scientific selection.

23. **Ledger 227 — label same-user implementation corrections exploratory.**
    An explicit `record_implementation_correction` can invalidate exposed
    original analysis only with clean changed source, current G1 and identical
    scientific/data/configuration/schedule inputs. Original lock/results remain
    immutable; strict primary verification still rejects old-source mismatch.
    A separate verified correction path permits one full same-user rerun inside
    repair admission, visibly `corrected_exploratory`, without retuning or a new
    holdout. This is the source-failure policy, not ordinary pre-freeze G2 repair.
    A mistake invalidates the comparative claim; verification and review are
    required before use. No such exposed correction was used in this run.

24. **Ledger 230 — apply transfer 2 once.** Diagnostics/report tests and final
    G1 proof required another 600 seconds from unstarted G4 before G2–G5.
    Event `ea34bebc-8188-453a-afba-aa1e44528bd6` gave G1=1800, G4=3600. Reopened
    snapshot preserved prior usage, resource hash and counters; stage sum 13800,
    global/setup/report caps were unchanged. Neither transfer could be repeated.

25. **Ledger 241 — predeclare optional contamination and instability triggers.**
    Before exposure, trigger contamination if the primary lower CI exceeds zero
    or full-cohort Signed versus tuned/matched MeanNegative lower CI exceeds
    zero. Optional high-H/high-O matched-mean triggers require at least 100 users.
    Material float32 instability means mean absolute user-AUC change at least
    frozen delta for any selected geometric slot. Raw score/rank changes are
    descriptive; float64 eta is not the float32 cutoff. Save reasons, counts and
    a reserve-gated conditional forecast. This avoids treating ordinary rounding
    or subgroup-only evidence as attribution and leaves the primary .01 threshold
    unchanged. A mistaken trigger would qualify optional diagnostics only. These
    diagnostics were not reached.

26. **Ledger 245 — allow verified historical reporting without execution
    authority.** Task 12 could factor saved-evidence reading that checks immutable
    lock/sidecar/freeze anchor, finished journals/artifact hashes and current
    ledger/invalidation without current source HEAD, environment or model cache.
    Such reading issues no outcome-access permit; confirm/diagnose stay strict.
    Reports should survive documentation commits and missing model installations.
    The risk is a provenance qualification error, requiring focused no-model and
    documentation-HEAD checks.

27. **Ledger 259 — apply transfer 3 for required validation.** At a measured
    193 seconds per suite, report validation plus current full G1 needed another
    600 seconds of unused G4 before real G2–G5. Effective G1 became 2400 and G4
    3000. Total, setup, reserve, evaluation and repair caps did not increase.
    The risk was insufficient later confirmation allocation, requiring an honest
    feasibility stop rather than a weakened protocol.

28. **Ledger 267 — review the whole implementation before outcome exposure.**
    After Task 12 acceptance, perform one broad review before real Tasks 13–14;
    retain operational reviews for those tasks and final changed-evidence checks.
    A post-exposure code correction would forfeit confirmation. Use the most
    capable reviewer, one complete fix wave and one scoped re-review, without
    duplicate broad review. Keep the scratch workspace until all 14 tasks and
    reviews finish. The broad reviewer cannot see future operational evidence;
    subsequent operational reviews cover that limitation. Extra native repairs
    still need focused verification and review.

29. **Ledger 291 — preserve the broad-review checkpoint.** The review ran at
    base `c75abd6`, head `7460b3a`, with a 606461-byte, 20-commit package on the
    designated `gpt-6-astra` xhigh reviewer. It was static, without suite or real
    data. One fix worker/wave and scoped re-review were allowed if needed, with
    no second wave. G1 had 43.48 seconds left; transfer 4 was not yet applied.
    Actual Tasks 13–14 remained required after review closure.

30. **Ledger 293 — overlap only native G0 readiness with read-only review.**
    After all 12 implementation tasks were accepted, inspect/acquire/audit could
    proceed independently. No tests, G1 proof, G2, scoring, benchmark, source edits
    or commits until review closure. Acquisition could generate pending pins;
    both lanes avoided CPU benchmarking. This reduced waiting while keeping
    scientific exposure gated. A provenance/source conflict was the risk;
    stable source and explicit pending-pin handoff addressed it.

31. **Ledger 303 — classify the native watchdog correction as material cycle
    1.** After focused G1 TDD, one combined covering command used the repair
    stage. Retain all related IDs and attributed experimental validation within
    1200 seconds, charging each runtime once. Source editing is separate,
    cached-download waiting uses setup, first successful parse/audit uses G0.
    Exact current full G1 after final source/pins is still required; no repair
    attestation substitutes for it. Narrowly allow repair validation in
    `run_command`, preserving cleanup, caps and zero evaluations. The risk is
    wrong repair attribution; verify IDs and aggregates without hidden resets
    or double charging.

32. **Ledger 304 — apply transfer 4 once.** Concrete fix TDD plus current full
    proof exceeded the remaining 43.48 seconds. Transfer 600 seconds of unused
    G4 before G2, event `004e74e3-86c9-4dd9-8e06-7c155f15757f`. Effective G1=3000,
    G4=2400; stage sum 13800, global 14400, setup 2700 and report 1800 stayed
    unchanged. Reopened-prefix verification preserved resource hash, counters,
    usage and attempts. The risk was reduced 40-minute confirmation allocation;
    stop or reduce scale only under the declared policy, never weaken the
    question or increase total time. Do not repeat the transfer.

33. **Ledger 312 — check native model construction before final G1.** After
    fix acceptance and successful G0 acquisition/audit, allow one accounted G0
    CPU encoder construction-only compatibility check. No encoding, probe,
    tuning, outcome access or cache build. It checks dependencies/model load and
    thread policy while corrections can still precede expensive G1. G2 must
    still measure its own cold/probe/repeat/full-cache work. Risk: small G0
    time/memory cost; require headroom, guard and actual-setting logs, with no
    empirical selection or substitution for G2 measurements.

34. **Ledger 327 — preserve the pinned invocation separately.** Save identical
    invocation values as `config/study-pinned.json`, restore the generic NULL
    template `config/study.json`, and use the pinned copy for later stages/proof.
    Filename is not part of canonical configuration/acquisition identity.
    Verify byte/value equality and reuse bound environment/acquisition/audit;
    do not rerun inspection or change its environment hash. Same first model
    revision, no outcomes, no implementation/test repair or material cycle.
    Risk: receipt-binding mismatch; require canonical equality and current full
    G1 before G2.

35. **Ledger 328 — apply transfer 5 once, the final allocation.** A measured
    216.73-second full proof plus safety exceeded 246.48 seconds remaining.
    Transfer 300 unused G4 seconds before G2, event
    `d6f55004-8fa4-47ca-addd-0e69f7b4596c`. Final G1=3300, G4=2100, exactly five
    transfers, stage sum 13800; total 14400, setup 2700, report 1800 and resource
    hash/counters/usage unchanged. Reopened prefix plus one event was verified.
    The risk was reduced 35-minute G4 allowance forcing a permitted feasibility
    stop. No transfer repetition, cap increase or question weakening was allowed.

36. **Ledger 331 — authorize final material cycle 2 for N1.** Correct durable
    manifest-writer stability; restore original bytes only after exact old-SHA
    checks, retaining failed bytes. Use focused TDD and one parent-owned repair
    acceptance with actual existing-dataset reload, without inference. Log all
    attribution within cumulative 1200 seconds without double charge. Add only
    the minimal N1 identifier support to the existing report validator, rather
    than a general framework. The same execution agent owns the fix, exact
    current full G1 and remaining stages; scoped Task 13 review covers the patch
    before any confirmation. Risks are corrupted provenance and repair-cap
    violation; exact reference hash, semantic/native round-trip, regression and
    log checks are mandatory. No third material cycle is available.

The accepted native run stopped at G2 without a lock or confirmation exposure.
Task 14 used only the intact G6 reserve for verified saved-evidence packaging.
The generic renderer's frozen-cohort and dominant-cost wording was corrected by
a separate tracked analysis wrapper, as authorized in the final integration
instructions; experimental source and policy were not repaired again. Final
reporting and archive costs extend the permanent ledger rather than resetting
any historical snapshot.
