# POC EXECUTION BRIEF

## 1. Project decision and mandate

**Project:** Determine whether nearest-negative penalties add useful preference information to a small multi-interest model built on frozen semantic item embeddings.

**Recipient:** An autonomous coding agent configured as **gpt-6.1-sol with xhigh reasoning**.

**Research and planning date:** 29 October 2026.

**Deliver the study, including an honest negative or early-stop result.** Implement the stages authorized below, run the stages whose gates pass on the target MacBook with an M3 Max, inspect the results, repair ordinary implementation defects within the stated allowance, and produce the research report. Do not ask for confirmation of routine implementation choices. Do not continue a failed research premise merely to complete a benchmark. This document is the instruction for that work; it does not assert that the PoC has already been implemented.

### 1.1 Selected project

Use one stable public dataset, MovieLens 1M, and one frozen public text encoder, sentence-transformers/all-MiniLM-L6-v2. Encode only the supplied movie title, release year, and genres. For each eligible user, construct at most four positive interest prototypes from earlier high ratings and retain a bounded set of earlier explicit low ratings. Rank the user's later explicitly rated items with

$$
S_{\lambda,K}(u,x)
=\max_{k\le K}x^\top p_{uk}
-\lambda\max_{n\in N_u}x^\top n.
\tag{1}
$$

Compare it with equally informed linear, prototype, nearest-class, and kernel alternatives. Separate the benefit of multiple positive interests from the benefit of replacing an average negative direction with a nearest-negative penalty.

**The smallest defensible contribution is a controlled empirical diagnostic, not a new scoring family:** establish whether this particular finite geometric readout has a practically meaningful advantage under frozen content features and bounded explicit feedback, or establish a scoped limitation that argues against further investment. A reproducible failure to improve on credible baselines is useful. Publication novelty is not promised.

The practical decision is whether to invest in this signed readout as a simple personalization component before attempting richer metadata, additional domains, online feedback, or retrieval infrastructure.

### 1.2 Why this project, and what was narrowed

The proposal supplies a useful distinction between representation and preference readout. However, the main mechanisms have substantial prior art:

- Maximum-over-interest scores are explicit in MaxMF (2013), including a procedure that freezes item factors while fitting user interests [R1].
- PinnerSage (2020) clusters user histories with fixed item embeddings [R2].
- Nearest-relevant and nearest-nonrelevant distances were combined for relevance feedback well before modern semantic embeddings [R3].
- PRIME (2026) combines frozen semantic positive and negative interest descriptions with signed matching, although its paired-interest and language-model pipeline differs from Equation (1) [R4].

Consequently, do not claim to invent multi-interest recommendation, signed similarity, frozen-embedding personalization, or a general preference language. The unresolved question for this project is their incremental value in the precisely defined regime below.

| Original direction | Decision and reason |
|---|---|
| Multiple interests plus explicit negative examples | Primary mechanism; equations are executable and available data can test it. |
| Context-dependent interest offsets | Exclude. MovieLens rating timestamps are not reliable observations of consumption context [R9]. |
| Active pairwise questioning | Exclude. No authorized human feedback experiment is available; simulated comparisons would not establish feedback efficiency. |
| List diversification/MMR | Exclude. It changes a different objective and would confound individual preference ranking. |
| Distinctiveness/hubness calibration | Exclude. It adds a separate objective and additional fitted reference distribution. |
| Interaction-trained representations | Exclude from the empirical claim. Their representational learning and cross-user information change the question. |
| Approximate nearest-neighbor retrieval | Exclude. The main task scores its entire explicit-label candidate set exactly; positive-only retrieval can miss the signed winner. |
| Rich movie understanding | Exclude as a conclusion. Title/year/genre text cannot establish that pacing, tone, or narrative structure is encoded. |

This is a content-based personalization study, not a recommendation-system leaderboard or a test of online satisfaction.

### 1.3 Supplied material and the missing-paper issue

The only supplied attachment was **Pasted text(20261029-082523).txt**, made available during planning at:

    /workspace/scratch/76c20debc0cf/upload/Pasted text(20261029-082523).txt

Its embedded proposal, “What if embedding proximity is solving the wrong problem?”, contains the XOR example, positive-max score, attraction/repulsion score, and signed synthesis. Its later instructions mention an attached mathematical paper, but **no separate paper was supplied**. There was no attached theorem, proof, appendix, or measured embedding figure to audit. The proposal's illustrations are conceptual descriptions, not experimental data.

This brief proceeds openly from the supplied equations and the independently retrieved literature. It does not substitute another paper for an unnamed missing paper and does not inherit any missing guarantee. A separate paper is unnecessary to execute this bounded diagnostic. If the intended task is specifically to validate that paper's theorem, obtain its title/DOI or PDF before making any paper-specific claim. All definitions needed for the present study are reproduced here; the coding agent need not receive the original attachment.

## 2. Research question, estimand, and decisions

### 2.1 One primary question

**Among eligible MovieLens users, does the signed multi-interest readout improve macro user AUC by at least 0.01 over the strongest comparator selected using development users, when all methods receive the same frozen content vectors and the same earlier explicit feedback?**

Let S* be the signed configuration selected on development users. Let B* be the comparator configuration selected on development users from all required baseline families in Section 6. Freeze both before confirmation. For confirmation user u, let

$$
d_u=\operatorname{AUC}_u(S^*)-\operatorname{AUC}_u(B^*),
\qquad
\Delta=\frac{1}{U}\sum_u d_u.
\tag{2}
$$

**Primary direction:** larger is better. **Practical threshold:** delta = 0.01 absolute AUC. This means one additional correctly ordered positive-negative pair per hundred pairs, averaged equally across users. It is an explicit R&D investment threshold, not a published standard, an estimate of commercial value, or a value to adjust after seeing results.

The primary estimand is conditional on later observed extreme ratings, the selected cohort, the encoder, and the feedback caps. It is not an exposure-corrected estimate of utility over the entire catalog. Missing interactions cannot be interpreted as explicit dislikes; exposure and preference are distinct variables [R14].

### 2.2 Mechanistic prediction

The nearest-negative identity can change with the candidate. This could preserve multiple positive regions while reducing scores around particular rejected examples. Improvement is expected when the frozen representation preserves distinctions relevant to the user's dislikes, negative labels are informative, and a single negative direction loses useful structure.

The effect may disappear or reverse when:

- a linear preference direction already suffices;
- negative feedback is sparse, noisy, stale, or reflects attributes absent from the metadata;
- an isolated low rating becomes an inappropriate nearest-negative veto;
- positive clusters do not correspond to stable preferences;
- a conventional nearest-class ratio or regularized kernel model captures the same structure more effectively;
- the encoder does not preserve the relevant preference information.

These are hypotheses to test or limitations to report, not established characteristics of MovieLens users.

### 2.3 Primary and mechanistic interpretation

Use a paired, user-level 95% confidence interval [L, H] for Delta, specified in Section 8.

| Observation after valid confirmation | Required interpretation |
|---|---|
| L > 0.01 | Supports a practically meaningful advantage over the development-selected B*, within the tested scope. |
| H < 0.01 | Evidence against the predeclared practically meaningful advantage. A smaller benefit may still exist. |
| H < 0 | Evidence of harm relative to B*. |
| L <= 0.01 <= H | Primary practical question is unresolved at this precision. |
| Entire interval lies inside [-0.01, 0.01] | Supports a small-difference conclusion at this resolution; do not claim exact equality. |
| A required comparator is invalid, missing, or inadequately tuned | No supported comparative claim against that comparator or the intended complete suite. |

Also report every required baseline on confirmation, even if it was not B*. The primary interval does **not** establish superiority over every baseline. If another comparator has a higher confirmation mean, explicitly state that the practical case for the signed method is weakened or absent despite any result against B*. Do not choose a new primary comparator after examining confirmation.

To determine whether replacing an average negative direction with a nearest-negative score adds practical value, report both:

1. Signed versus independently tuned MeanNegative.
2. Signed versus MeanNegative using exactly S*'s K and lambda, with identical prototypes.

The first is the fair practical ablation; the second isolates the complete aggregation substitution without changing selected parameters. The raw negative mean has a user-specific norm, so the substitution changes penalty magnitude as well as the identity of the controlling negative. Even a broad global lambda grid does not eliminate these between-user scale differences. These controls therefore do not uniquely attribute a gain to candidate-dependent geometry. Log negative-mean norms and retain this attribution limitation. A separately tuned normalized-mean control would be needed before making that stronger claim; it is outside this minimal study.

Positive-only scores at S*'s K are another matched control. Also evaluate SignedMax at K=1 with S*'s lambda to isolate the positive-prototype-count change while holding the negative term fixed. These controls require no new tuning grid. They are secondary comparisons, with intervals identified as secondary.

If selected lambda is zero, the data did not select negative feedback. If selected K is one, the data did not select multiple interests. Selecting lambda>0 and K>1 describes the fitted configuration; it does not independently confirm that either component contributes. A specific multiple-interest claim requires the frozen K=1 comparison, and any component claim must retain the secondary-analysis qualification.

If an interval comparison against an RBF or nearest-class reference rules out the meaningful advantage, report that result as a useful limitation of this heuristic. A higher comparator point estimate or a nonsignificant difference alone is not an established negative result. Do not reinterpret either finding as evidence against nonlinear preference models generally. A favorable engineered fixture alone is correctness evidence, not an empirical contribution.

## 3. Evidence and mathematical audit

### 3.1 Objects, domains, and assumptions

| Object | Definition |
|---|---|
| I | Fixed movie catalog with stable integer item IDs; 3,883 metadata records in the planning download. |
| e_i | Frozen encoder output in real dimension d = 384. |
| x_i | e_i normalized to Euclidean norm one, recomputed in float64 from the stored float32 encoder output. |
| H_u+ | Earlier ratings >=4, capped at 128 items by the rule in Section 4. |
| H_u- = N_u | Earlier ratings <=2, capped at 64 items; each negative is its unit item vector. |
| P_u(K) | K or fewer unit positive prototypes constructed only from H_u+. |
| K | One of {1, 2, 4}; actual count cannot exceed the number of distinct positive vectors. |
| lambda | Nonnegative scalar selected from the declared grid, globally across users. |
| C_u | All later observed items with ratings >=4 or <=2, not sampled unseen items. |
| Output | A finite real score per candidate and deterministic ranking, largest score first. |

Assumptions to inspect rather than take on faith: nonempty signed histories; valid metadata joins; finite encoder outputs; stable item IDs; no user/item overlap across that user's support and evaluation; enough eligible users; labels and metadata containing some preference signal. A low star rating is an explicit negative preference proxy, not proof of a permanent rejection in every context.

### 3.2 What the linear obstruction establishes

For the proposal's two positives (1,1), (-1,-1) and two negatives (1,-1), (-1,1), the positive vectors sum to the same vector as the negative vectors. Thus, for any w and intercept b, the sums of the two positive and two negative affine scores are equal. Strictly placing both positives above both negatives would require the first sum to exceed the second, a contradiction. Normalizing all four by the same norm does not alter the argument. A strictly increasing transform of the scores does not rescue their ordering.

This establishes a limitation of a fixed linear readout on those fixed features. It does **not** establish that real semantic embeddings exhibit this configuration or that interaction-trained item embeddings cannot transform the original problem into a linearly representable one.

### 3.3 Exact nearest-distance equivalence

For unit x, p, and n, define

$$
d_P(x)=\min_{p\in P}\|x-p\|_2,\qquad
d_N(x)=\min_{n\in N}\|x-n\|_2.
$$

Because x transpose p = 1 - ||x-p||²/2,

$$
S_\lambda(x)
=1-\lambda-\frac{d_P(x)^2}{2}
+\frac{\lambda d_N(x)^2}{2}.
\tag{3}
$$

At lambda = 1 this is half a nearest-negative versus nearest-positive squared-distance margin. Its sign gives the nearest-class decision, apart from ties. Compressing positives changes the reference set; it does not introduce a new mathematical primitive.

**Our derivation:** Equation (3) follows directly from expanding squared Euclidean distances. It is not attributed to a theorem in the missing attachment. Unit normalization of the **prototypes**, as well as of the items, is necessary for this particular formula.

Giacinto and Roli [R3, Sections 2–3, Equation (1), PDF page 4] instead rank by

$$
R(x)=\frac{d_N(x)}{d_P(x)+d_N(x)}.
\tag{4}
$$

Equation (4) is a close mandatory baseline. Distance difference and distance ratio need not yield the same ranking. The ratio is rank-equivalent to d_N²/(d_P²+d_N²) when the denominator is positive; do not create redundant baseline families for these monotone transforms.

### 3.4 Geometry, guarantees, and limitations

The score is a difference of support functions:

$$
S(x)=h_{\operatorname{conv}P}(x)
-\lambda h_{\operatorname{conv}N}(x).
$$

It is positively homogeneous and piecewise linear on ambient space. On a region with winning positive k and negative j, it is x transpose (p_k - lambda n_j). Its threshold sets satisfy

$$
\{x:S(x)\ge t\}
=\bigcup_k\bigcap_j
\{x:x^\top(p_k-\lambda n_j)\ge t\}.
\tag{5}
$$

These are algebraic statements, not learning or generalization guarantees. Equation (5) follows by subtracting the same negative maximum from each positive term, then expressing the negative maximum as an intersection of inequalities.

For unit prototypes, |S(x)-S(y)| <= (1+lambda)||x-y||. This stability bound follows from Cauchy–Schwarz and the Lipschitz property of a finite maximum. It does not protect the ordering of almost-tied candidates.

Goodfellow et al. [R5, Section 4, Propositions 4.1–4.2 and Theorem 4.3] discuss unrestricted affine maxout units with arbitrarily many pieces. That approximation result supplies **no** guarantee for four normalized centroids, fixed item exemplars, no learned offsets, or the clustering heuristic here. Do not claim universal approximation, exact preference recovery, global clustering optimality, or convergence to a statistically optimal ranker.

The word “local” describes which negative exemplar controls a region. The raw term is not compactly supported and is not always a positive penalty: if every x transpose n is negative, subtracting their maximum increases the score. Clipping at zero would be a different model and is excluded from the main protocol.

Other exact restrictions:

- A single positive and negative direction remains linear.
- Adding a negative cannot increase a candidate's score, but it may change rankings.
- Duplicating the same negative does not change the maximum.
- Equal item vectors always receive equal scores, including when their labels conflict.
- A single mislabeled negative can dominate a region.
- Frequency affects positive prototype construction, while a negative maximum ignores duplicate frequency.
- Clustering is adaptation to feedback even though the item encoder remains frozen.

### 3.5 Constructive bridge and costs

There is no hidden optimization oracle in Equation (1). Given finite vectors, exact scoring consists of two matrix products, row maxima, and subtraction. Prototype construction is a bounded heuristic, fully specified in Section 5. No convergence theorem is needed to define its finite output.

For m candidates, dimension d, K prototypes, and J negatives, scoring costs O(m d (K+J)). A chunk of b candidates needs b-by-K and b-by-J similarity blocks. Do not allocate a users-by-items-by-history tensor. Prototype construction costs at most O(T K d |H_u+|), with T capped at 50.

Embedding download and inference, data parsing, prototype construction, baseline fitting, tuning, and result writing are real costs. Include them in the cumulative budget and report them separately from scoring latency.

### 3.6 Candidate-generation obstruction

Positive retrieval alone is not guaranteed to preserve the signed winner. Let

$$
p=(1,0),\quad n=(1/2,\sqrt3/2),\quad
a=(1,0),\quad b=(\sqrt3/2,-1/2).
$$

With lambda = 1, positive similarities are 1 and sqrt(3)/2, but signed scores are 1/2 and sqrt(3)/2. A positive top-1 query drops the signed winner b. Both negative similarities are nonnegative, so this is not merely an antipodal-reward artifact.

The present study therefore performs exact scoring of all C_u. A catalog-scoring microbenchmark measures computation only; it cannot establish catalog ranking quality or candidate recall.

### 3.7 Compact claim-to-evidence map

| Central claim | Evidence status and location | Uncertainty / resolving check |
|---|---|---|
| Fixed linear scores cannot separate the supplied XOR ordering | Exact derivation above; proposal Section 1 | Reproduce analytic proof and tiny independent feasibility check. |
| Max-over-interests is established | MaxMF Equation (2), Algorithm 2 [R1]; MIND/ComiRec [R6–R7] | No novelty claim for this component. |
| Frozen item vectors plus clustered histories are established | PinnerSage Sections 2 and 3.1 [R2] | Its graph-informed embeddings differ from our content-only regime. |
| Signed maxima equal a nearest-class distance margin | Exact derivation, Equation (3) | Check independently by direct distances. |
| Nearest-positive/negative relevance scoring is prior art | [R3] Equation (1) | Include its ratio and the uncompressed margin as close baselines. |
| Negative centroid feedback can hurt | Reported Science Concierge experiment [R8, Section 3.1/Figure 3] | Its feedback was simulated from topics; it does not predict our result. |
| Fixed semantic positive/negative interest matching is not new | PRIME [R4, Section 3.3, Equations (11)–(13)] | Different paired descriptions and downstream reasoning, not our exact ranker. |
| More expressive geometry improves this task by >=0.01 | New empirical hypothesis | Locked comparison and matched ablations. |
| Ratings represent later preference submissions, not later watching | Dataset authors [R9, Section 3.2], plus planning audit | Narrow the estimand; no claim about future consumption. |
| The study fits an M3 Max in hours | Planning estimate only | Target-machine calibration and hard resource gates. |

## 4. Data, representation, and separation of evidence

### 4.1 Primary acquisition and provenance

Retrieve the stable MovieLens 1M archive from:

<https://files.grouplens.org/datasets/movielens/ml-1m.zip>

Read and retain its README [R10]. Planning downloaded **5,917,549 bytes**, SHA256:

    a6898adb50b9ca05aa231689da44c217cb524e7ebd39d264c56e2832f2c54e20

Verify this digest. A mismatch is a data-version discrepancy: inspect the README/schema and document the reason before proceeding; do not silently accept different data under the same expected counts.

Parse ratings.dat as UserID::MovieID::Rating::Timestamp. Parse movies.dat as MovieID::Title::Genres with an explicit latin-1 decoder, preserving original title text, then write derived text as UTF-8. Ratings are integer stars 1–5. Movie IDs are not contiguous. Do not read demographic fields from users.dat; they are unnecessary.

Validate unique (user_id, item_id) pairs, unique metadata IDs, valid timestamps and rating range, complete rating-to-metadata joins, and absence of nonfinite values. The inspected primary archive had no duplicate user-item pairs. Unexpected duplicates or missing joins are a constructive-feasibility discrepancy to resolve before producing splits; do not silently remove rows and retain the old cohort claim.

Assume this is noncommercial research use. The README requires acknowledgment, restricts redistribution, and states conditions on commercial use [R10]. Keep downloaded raw data and derived per-item content local; do not include them in a public repository or redistributed result bundle. Package acquisition scripts, hashes, aggregate tables, and the required local reproducibility outputs. Do not contact anyone or publish data as part of this task.

### 4.2 Exact per-user episode construction

Construct the split before feature-dependent diagnostics:

1. Sort **all** a user's ratings, including three-star ratings, by (timestamp, movie_id).
2. Let n be the number of records and j = (7*n)//10 using integer arithmetic and zero-based indexing.
3. Let t_cut be the timestamp of record j.
4. History is every record with timestamp < t_cut; evaluation is every record with timestamp >= t_cut.
5. High ratings >=4 are positive; low ratings <=2 are negative. Three-star records affect the temporal boundary but are not supervised labels or evaluation candidates.
6. Retain users with at least 10 history positives, 5 history negatives, 5 evaluation positives, and 5 evaluation negatives.
7. From history, select the newest at most 128 positive and newest at most 64 negative records independently, ordered by (timestamp, movie_id). Select from the end of each ascending list. The cap may break a history timestamp tie by movie ID; it never moves evaluation items into history.
8. Use every evaluation positive and negative as C_u. Do not cap, resample, or add unlabeled candidates.

This eligibility rule conditions the target population on having sufficiently many observed labels. Report the entire filtering funnel and make no claim about users with no explicit dislikes. All methods receive identical selected item IDs and signs.

**Important exactness:** do not implement j as int(0.7*n). For n=90, that floating calculation produced 62 in the planning environment, while the specified integer formula gives 63. The final audited counts below use the exact formula.

### 4.3 Development and confirmation users

Assign each eligible user independently of model performance:

~~~python
digest = hashlib.sha256(
    ("signed-study:" + str(user_id)).encode("utf-8")
).hexdigest()
is_development = int(digest[:8], 16) % 5 == 0
~~~

Development users supply both their history and later labels for debugging, hyperparameter selection, precision planning, and selection of B*. Confirmation users supply only histories for model construction until the protocol is frozen. Their later labels may be accessed by the data auditor to determine the predeclared eligibility counts; they must not be exposed to model selection, feature choices, subgroup selection, or outcome inspection.

User-specific histories necessarily contain each confirmation user's earlier labels. “Held-out user” means held out from **global protocol and hyperparameter selection**, not a user for whom the model has no feedback.

No cross-user supervised fitting, item-popularity estimate, item bias, learned scaler, whitening, or collaborative representation is used. Embedding the full fixed metadata catalog is allowed because the encoder is frozen and receives no ratings. This is user-disjoint evaluation with within-user temporal ordering; it is not a globally time-causal replay of the historical service. Generalization is conditional on the catalog and the chosen user population.

### 4.4 Planning data audit actually performed

Small data inspection was performed on the planning environment, **Linux x86_64, Python 3.12.14**, not on the user's Mac. It downloaded the official archives, inspected schemas, counted labels/cohorts, and examined timestamp structure. No encoder was run, no recommender was fitted, and no comparative ranking outcome was examined.

| Quantity, exact primary protocol | Observed during planning |
|---|---:|
| Raw ratings / users | 1,000,209 / 6,040 |
| Metadata movies / movies with ratings | 3,883 / 3,706 |
| Positive / negative / neutral ratings | 575,281 / 163,731 / 261,197 |
| Eligible users | 2,194 |
| Development / confirmation users | 418 / 1,776 |
| Median history / evaluation records, including neutral | 168 / 74 |
| Median history positive / negative counts before caps | 91.5 / 23 |
| Median evaluation positive / negative counts | 33 / 14 |
| Users with positive / negative history truncated by cap | 730 / 347 |
| Eligible users whose entire recorded history spans <1 day | 976 |
| Split boundaries on the same UTC calendar date | 2,154 |
| Median gap across the split boundary | 33 seconds |

The short gaps agree with the dataset authors' explanation of historical rating backfilling [R9]. Use “later rating submissions” throughout the report. This chronology prevents label leakage within a user's episode but does not turn ratings into consumption events.

The counts must be reproduced on acquisition. They are feasibility evidence, not a successful PoC or an M3 performance claim.

### 4.5 Frozen embedding specification

Use the official model **sentence-transformers/all-MiniLM-L6-v2** [R11]. Its model card describes 384-dimensional sentence/short-paragraph embeddings and default truncation at 256 wordpieces. It is a sentence-similarity encoder, not a user-preference model.

Resolve and record one immutable repository revision before encoding. Download the required model/tokenizer files locally; record revision, file hashes, license, installed versions, and whether safetensors or another supported weight format was used. Use the standard supported model architecture with remote custom code disabled.

For every metadata item, construct exactly:

    Movie: {original title including year}. Genres: {alphabetically sorted genre names joined by ", "}.

Preserve the original title, including trailing articles and release year. Normalize whitespace and sort genre strings deterministically; do not rewrite movie titles or generate plot summaries. No ratings, reviews, aggregate scores, tags, popularity, demographics, or language-model-generated text may enter the template.

Use explicit CPU inference in float32, evaluation mode, no gradients, and batch size 32 initially. Set maximum sequence length to the checkpoint's documented 256; log actual token-length distribution and truncation count. Cache outputs once, in item-ID order. Store float32, then cast to float64 and renormalize each vector for scoring. Reject a nonfinite or norm <=1e-12 output as a failed preprocessing check.

Check deterministic repeated encoding of a 32-item calibration sample. Record maximum difference; require <=1e-5 absolute component difference for the same CPU setup, otherwise investigate threading/backend behavior before freezing the cache. Do not average repeated encodings to conceal variation.

No encoder fine-tuning, projection fitting, PCA, whitening, coordinate standardization, or per-method embedding changes are permitted. All methods use the exact same cached vectors.

The model revision's training corpus was not audited for historical movie or benchmark contamination. A fixed public language encoder may encode external knowledge; it is not evidence of a historically deployable 2000-era system. No model was trained on this study's split by the agent.

### 4.6 Bounded access fallback

If the official 1M archive remains unavailable after two retrieval attempts and one official-page check within the acquisition allowance, use MovieLens 100K **only as a declared reduced-data fallback**:

<https://files.grouplens.org/datasets/movielens/ml-100k.zip>

Planning archive: 4,924,029 bytes; SHA256:

    50d2a982c66986937beb9ffb3aa76efe955bf3d5c6b761f4e3a7cd717c6a3229

Read u.data as tab/whitespace-separated user, item, rating, timestamp. Read u.item as pipe-separated latin-1 metadata; title already contains year, and the supplied genre indicators map to names in u.genre. Ignore release-date and URL fields for the encoder. Apply exactly the same episode, labels, caps, hash split, models, and metric rules.

The exact fallback audit produced 287 eligible users: 62 development and 225 confirmation. It is a credible correctness/pilot dataset, with a weaker confirmation opportunity. Keep delta = 0.01; never lower it to compensate. Apply the same interval interpretation, explicitly acknowledging the smaller population and development set. Do not present it as a replication on 1M or switch after unfavorable 1M outcomes.

All catalog-sized encoding, validation, and timing operations use the active dataset's metadata cardinality: 3,883 for MovieLens 1M or 1,682 for the 100K fallback. Primary-size examples elsewhere do not override this rule.

If the encoder cannot be downloaded or executed within the bounded setup/repair allowance, complete mathematical checks, data audit, and reproducible code preparation. Do not silently substitute TF-IDF or genre vectors and claim to have tested the frozen semantic hypothesis. Such a substitute would require a separately stated follow-up project.

## 5. Unambiguous computational mechanism

### 5.1 Prototype constructor

Implement a small deterministic spherical Lloyd heuristic in NumPy. This avoids confusing ordinary Euclidean k-means followed by normalization with spherical clustering. Its finite output is the algorithm; no global-optimality claim is made.

Input: unit float64 positive rows X in ascending item-ID order, associated IDs, requested K.

1. Retain duplicate vectors as observations for mean updates, but initialize from distinct vectors. Distinctness here is exact equality of the normalized stored rows; do not invent a similarity threshold.
2. Set K_eff = min(K, number of distinct vectors).
3. Compute the arithmetic mean mu. Select the first prototype as the observed row maximizing its dot product with mu. If mu has norm <=1e-12, select the smallest item ID. Exact ties use smallest item ID.
4. Select each remaining prototype as the as-yet-unselected distinct row minimizing its maximum dot product to the selected prototypes; ties use smallest item ID.
5. Assign each row to its highest-dot-product prototype; ties use smallest prototype index.
6. Replace each nonempty cluster prototype with its normalized vector sum. If the sum norm <=1e-12, retain the previous unit prototype. Retain a previous prototype for an empty cluster.
7. Recompute objective Q = sum_i max_k X_i transpose p_k. Stop when assignments are unchanged after an update, or improvement <=1e-10 times the number of positive rows. Check objective decrease beyond 1e-9 times the row count as a numerical/implementation defect.
8. Stop after at most 50 updates even if the criterion is not met. Return the final unit prototypes, objective, iteration count, empty-cluster flags, zero-sum flags, and cap-hit flag.

K=1 produces the normalized positive mean unless that mean vanishes, in which case it retains the deterministic observed-row fallback. Log this rare fallback; it is not an arbitrary division by zero. Unused/duplicate resulting prototypes may be deduplicated for scoring without changing the maximum, but preserve requested and effective K in results.

A cap hit is a reported finite-heuristic outcome, not automatically a failed user. If more than 5% of calibration users hit the cap, inspect the objective traces. A single pre-freeze increase to 100 updates for **all** prototype-using methods is allowed with recorded scientific evidence; record it. Do not tune initialization using held-out labels.

### 5.2 Scoring pseudocode

~~~text
fit_user(history, embedding_matrix, K):
    positive_ids, negative_ids = exact capped signed history
    P = spherical_prototypes(E[positive_ids], K)
    N = E[negative_ids]
    mean_N = arithmetic mean of rows of N, or zero if N is empty
    return P, N, mean_N, constructor_diagnostics

score_candidates(model, candidate_ids, lambda, chunk_size=512):
    require lambda >= 0
    for chunk in candidate_ids:
        X = E[chunk]                         # float64, unit rows
        A = row_max(X @ P.T)
        B = row_max(X @ N.T) if len(N) else zeros
        yield A - lambda * B

rank(scores, item_ids):
    require all scores finite
    sort by descending score, then ascending integer item ID
~~~

Do not normalize the final score, clip negative cosine values, update history with evaluation labels, or recalibrate one user's scores using later labels.

For the primary cohort N is nonempty. The general function must return positive-only scores if N is empty, and must raise a clear ineligible-history error if P is empty. This is a function-contract check, not permission to silently include additional cohorts.

### 5.3 Numerical conventions

- Encoder inference: CPU float32. All geometry, fitting, distances, and metrics: CPU float64.
- Renormalized item/prototype norms must differ from 1 by <=1e-12, except that linear combined weight vectors need not be unit.
- Dot products can stray slightly outside [-1,1] through rounding. For distance reconstruction only, clip to [-1,1] and record any excess >1e-10 as a defect. Direct score computation uses the dot products without nonlinear clipping.
- For Equation (4), use Euclidean distances. Return 0.5 when both distances are exactly zero, 1 when only d_P is zero, and 0 when only d_N is zero. Otherwise use the stated ratio. No large arbitrary denominator epsilon.
- A zero Rocchio weight gives all-zero scores and valid ties; do not divide by its norm.
- No FP16, BF16, quantization, stochastic dropout, or automatic device selection.
- Score comparisons for metric ties use exact equality of the stored float64 scores; report near-tie frequencies and the separate precision check. Do not introduce a model-dependent tie tolerance.

## 6. Baselines, ablations, and tuning

### 6.1 Required model families

Define A_K(x) = max_k x transpose p_k, A_all(x) = max_{h in H_u+} x transpose h, B_max(x) = max_{n in N_u} x transpose n, and B_mean(x) = x transpose mean(N_u). The negative mean is **not normalized**: this is literally mean cosine similarity.

Let Lambda = {0, 0.25, 0.5, 1, 2, 4, 8}. The broad range gives the mean-negative control room to compensate for its smaller vector norm. Large lambda values can be harmful; that is handled by development selection, not by deleting them after confirmation.

| Family | Score / fit | Development configurations | Role |
|---|---|---:|---|
| Centroid | A_1 | 1, reused | Simple reference; deterministic zero-mean fallback as specified. |
| PositiveMax | A_K | 3, reused | Tests multiple positive interests. |
| RocchioDirection | A_1 - lambda B_mean | 7, reused | A single linear positive-minus-negative direction. |
| MeanNegative | A_K - lambda B_mean | 3 x 7 = 21 | Mandatory matched information/positive-geometry control. |
| SignedMax | A_K - lambda B_max | 3 x 7 = 21 | Proposed finite readout. |
| AllExemplarMargin | A_all - lambda B_max | 7 | Established nearest-class-style geometry without positive compression. |
| PrototypeRatio | Equation (4), P=P_u(K), N=N_u | 3 | Closest relative-distance comparison with matched compression. |
| ExemplarRatio | Equation (4), P=H_u+, N=N_u | 1 | Conventional all-example relevance-feedback baseline. |
| LinearLogistic | Weighted L2 logistic regression on the unit vectors | 5 | Strong learned single-direction baseline. |
| RBFSVM | Weighted RBF SVM on the same vectors | 20 | Strong simple nonlinear reference. |

There are **78 declared grid entries** before duplicate reuse: 21+21+7+3+1+5+20. Centroid, PositiveMax, and RocchioDirection are extracted from these entries, not extra fitting campaigns. Score-equivalent entries, such as lambda=0 in both signed families, may share computation; each declared user/configuration evaluation still counts toward the evaluation ceiling.

MeanNegative with K=1 is a Rocchio-type direction; the positive centroid normalization convention differs from some textbook parameterizations. State the exact implementation rather than claiming to reproduce every historical Rocchio variant.

AllExemplarMargin at lambda=1 is exactly the distance-margin identity. PrototypeRatio and ExemplarRatio make the ratio-versus-difference and compression distinctions visible. Their scores are not calibrated probabilities.

Do not implement MIND, ComiRec, PRIME, or PinnerSage's full production stack. They establish prior mechanisms but involve different data, training, or objectives. Do not compare this experiment to their published numbers. A collaborative matrix-factorization model is also outside the fixed-content, per-user-information claim; this exclusion does not justify any assertion of superiority over interaction-trained recommenders.

### 6.2 Linear and nonlinear fits

Use binary labels 1 for positive and 0 for negative. Every model receives the identical capped support. Use balanced class weights h/(2*h_class), computed separately for each user's support.

**LinearLogistic:** scikit-learn LogisticRegression with L2 regularization, solver=liblinear, fit_intercept=True, tolerance 1e-8, maximum 2,000 iterations, random_state=0, and C in {0.01, 0.1, 1, 10, 100}. Rank by its decision_function, not by hard class predictions. The fitted intercept cannot change within-user ranking. Verify the installed API expresses L2 correctly; recent official documentation deprecates some older penalty arguments [R12].

**RBFSVM:** scikit-learn SVC with kernel='rbf', balanced weights, tolerance 1e-6, maximum 100,000 iterations, cache_size=64 MB, no probability calibration, and decision_function scores. Let m2 be the median **positive** off-diagonal squared distance between support vectors, calculated only from that user's history. Set m2=1 if all off-diagonal distances are zero. Use

$$
\gamma=g/m2,\quad
g\in\{0.25,1,4,16\},\quad
C\in\{0.01,0.1,1,10,100\}.
$$

There are 20 configurations. Use contiguous float64 arrays, one user/model at a time, and release model objects after saving required outputs. SVC has at least quadratic sample scaling [R13]; the support cap of 192 avoids a large full-dataset kernel.

Treat solver warnings, nonfinite decision values, nonzero SVC fit_status_, and iteration exhaustion as failed fits requiring the bounded failure policy. Do not silently substitute a simpler classifier. One global pre-freeze increase to 4,000 logistic iterations / 200,000 SVM iterations is allowed if calibration demonstrates a convergence issue and non-time resources permit it.

### 6.3 Tuning and freeze

Score all development users for every declared configuration, using the full primary cohort under results-first (legacy bounded studies retain their predeclared outcome-independent calibration subsets). Choose each family's configuration by maximum macro development AUC. For differences <=1e-6, prefer smaller K, then smaller lambda, then smaller C, then smaller g, then lexicographic configuration ID as applicable.

Select B* by maximum development AUC among the selected required comparator families, excluding SignedMax and the deliberately untuned matched variants. An exact tie uses the lexicographic family ID. Do not choose B* by confirmation.

A boundary winner can indicate a truncated grid. Before freeze, permit **at most 14 additional configurations total**:

- If LinearLogistic's winning C is an endpoint, add its next outer decade.
- If RBFSVM's winning C or g is an endpoint, add at most the next outer row/column needed, with duplicate entries removed.
- If MeanNegative, SignedMax, or AllExemplarMargin selects lambda=8, add lambda=16 for the selected K (or the all-example family).

Generate eligible requests in this fixed order: (1) RBFSVM's outer C row and/or outer g column, including their intersection when both are requested, sorted by numeric (C,g); (2) LinearLogistic's outer C; (3) MeanNegative at lambda=16 and its selected K; (4) AllExemplarMargin at lambda=16; (5) SignedMax at lambda=16 and its selected K. Only request an expansion when its endpoint condition holds. Expanding both RBF endpoints changes a 5-by-4 grid to 6-by-5 and adds 10 entries, including the new corner; with the other four requests the maximum is 14. The maximum total is therefore 92 declared configurations before duplicate reuse.

Use the same development set and log every added configuration. Admit this one-time expansion only if its complete projected cost fits; otherwise record exactly which requested tuning coverage remains unavailable and do not claim a fully tuned comparison. Do not silently omit the RBF corner. Report each winning endpoint's gain over its adjacent interior setting, and label coverage limited whenever the winner remains on the expanded outer boundary. No further expansion or implicit significance-based rescue is allowed.

Before opening confirmation scores, write immutable protocol.lock.json containing data/model hashes, eligible IDs, selected support/candidate IDs or their digests, all configurations, selected S* and B*, preprocessing, thresholds, seeds, precision, timing protocol, budget ledger, and any deviations. Hash the code commit and lock file. Freeze report-table schemas and primary metric code too.

Do not change the question, primary comparator, embedding, eligibility, candidate set, practical threshold, or analysis after reading confirmation outcomes. A discovered validity defect requires the invalid-experiment procedure, not a quiet rerun until the result looks better.

## 7. Correctness and smallest sufficient experiment set

### 7.1 Independent correctness gates

Implement a slow direct-distance/loop reference independently of the vectorized max-dot scorer. Tests should validate mathematical and measurement behavior, not merely mirror implementation lines.

| Check | Expected result / acceptance |
|---|---|
| Equation (3), random normalized vectors | Direct-distance and max-dot scores agree within 1e-10 absolute in float64 for lambda <=16. |
| Max-min form | max_k min_j x transpose (p_k-lambda*n_j) agrees with Equation (1). |
| Lambda zero | Exact positive-max score, including ranking. |
| K=J=1 | Agreement with direct linear score x transpose (p-lambda*n). |
| Prototype updates | Unit norms, finite objective, no material objective decrease, deterministic repeated output. |
| Orthogonal rotation | Scores agree within 1e-10 with prototypes and examples rotated together. |
| Candidate order and chunk size | Scores agree within eta=1e-10*(1+lambda) for chunk sizes 1,7,512; pairwise ordering is unchanged when the reference score gap exceeds 2*eta. Log any near-tie reorderings. |
| Duplicate negatives | Same SignedMax scores; mean-based duplicate behavior need not match. |
| Identical vectors with conflicting labels | Unavoidable ties; no hidden item-ID predictive score. |
| Empty negatives / positives | Positive-only function output / explicit ineligibility error. |
| Ratio corner cases | Both-zero 0.5; positive-only zero distance 1; negative-only zero distance 0. |
| XOR ordering | Analytic linear impossibility; an independent linear-program feasibility check may verify it on the four points. Nonlinear positive prototypes can separate them. |
| Metric oracle | AUC: perfect=1, reversed=0, all ties=0.5; hand-computed mixed-tie example agrees with sklearn. NDCG has hand-checked tied and short-list examples. |
| Data firewall | Replacing or permuting a confirmation user's evaluation labels leaves fitted model parameters and scores byte-identical. |
| Split boundary | Tied timestamps never cross history/evaluation; n=90 uses index 63; neutrals affect the split. |
| Classical fits | On a small separable linear fixture the linear model orders classes correctly; on a nonlinear fixture the RBF model can recover the expected ordering with a declared sufficiently flexible setting. |

The fixture-fit settings test correct implementation, not the final tuned performance of those models. Do not insist that every hyperparameter setting solve a synthetic fixture.

### 7.2 Exact local-negative positive control

Use P={+e3,-e3}, N={+e1,-e1}, lambda=1. Desired candidates are

    (0, 0.6, 0.8), (0, -0.6, -0.8)

and undesired candidates are

    (0.6, 0, 0.8), (-0.6, 0, -0.8).

All are unit vectors. PositiveMax is 0.8 for all four; the negative mean is zero. SignedMax scores are **0.8, 0.8, 0.2, 0.2**. The relevance-ratio baseline also separates the two groups. This detects the claimed aggregation effect without pretending that it is unique or newly discovered.

Also verify the retrieval counterexample in Section 3.6. These are analytic fixtures with deterministic expected results; repeated random seeds add no scientific evidence.

### 7.3 Actual planning computations

The planner executed the distance identity on NumPy 2.3.5 float64 with rng=np.random.default_rng(413), drawing standard-normal arrays in order P=(5,16), N=(7,16), X=(8,16), then normalizing each row, with lambda=0.7. Independent direct squared distances and max-dot evaluation differed by at most **2.7755575615628914e-16**. The local-negative fixture produced the scores above, with ordinary floating rounding at 0.2. The max-min and retrieval identities were also checked by the mathematical reviewer.

These checks verify arithmetic and help specify tests. They establish no recommendation accuracy, runtime advantage, or M3 performance. Reproduce the essential tests in the implementation; do not treat planning notes as a substitute for executable validation.

### 7.4 Experiment matrix

| Experiment | Work | Claim or uncertainty resolved |
|---|---|---|
| Exact fixtures and independent references | Section 7.1–7.2 | Algorithm, baseline, split, and metric correctness. |
| Resource calibration | 128-item encoder calibration, then admitted full shared cache; 32 development users covering short/median/long histories; all model families | Actual preprocessing, fitting, scoring, memory, and solver feasibility. |
| Development comparison | 78 configurations plus bounded expansion, on 418 development users if feasible | Choose honest strong comparisons and fixed parameters; estimate achievable interval width. |
| Locked primary comparison | All selected families plus matched MeanNegative, PositiveMax, and K=1 SignedMax controls on 1,776 confirmation users if feasible | Primary practical question and operational component diagnosis. |
| History-defined strata | Reuse locked scores; no extra fits | Whether observed effects align with predeclared mechanism conditions. |
| Numerical robustness | Recompute geometric scores in float32 for the first 64 confirmation IDs in hash order; compare with float64 | Detect sensitivity to finite precision, without changing primary backend. |
| Negative-label contamination, conditional | First 200 confirmation users in hash order; 3 fixed seeds; SignedMax, MeanNegative, LinearLogistic, RBFSVM | Diagnose vulnerability to erroneous explicit feedback if there is a primary or mechanistic gain, or unexplained unstable behavior. |
| Catalog-scoring timing | 32 fixed development users, all 3,883 metadata items, selected methods | Computational cost only, separate from ranking evidence. |

No random unobserved negatives, repeated benchmark hunting, extra datasets after a disappointing result, or large synthetic sweep is needed. Real held-out observations are the representative cases; the exact fixtures isolate behavior. If only the fixtures work, state that plainly.

For contamination, swap the signs of q positive and q negative support items, with q=max(1,floor(0.1*min(h_pos,h_neg))). Sample without replacement using seeds 1101,1102,1103, preserve class counts and the original support IDs, refit all four methods, and leave evaluation labels unchanged. Use their original frozen hyperparameters; this is a stress diagnostic, not a newly tuned regime. Cache the swap assignments so every method gets identical corrupted histories. Run it when the diagnostic question is applicable and actual non-time resources permit it. Under the 31 October results-first amendment, the original 20-minute headroom rule is advisory; otherwise report it unperformed.

## 8. Metrics, uncertainty, and interpretation limits

### 8.1 Primary metric

For each eligible user,

$$
\operatorname{AUC}_u
=\frac{1}{|C_u^+||C_u^-|}
\sum_{i\in C_u^+}\sum_{j\in C_u^-}
\left[\mathbf1(s_i>s_j)+\tfrac12\mathbf1(s_i=s_j)\right].
\tag{6}
$$

Compute macro AUC by giving each user equal weight. The independent resampling unit is the user, not a rating or a positive-negative pair. All pairs from a user share history and items and are not independent observations.

Implement an efficient rank-based AUC and verify it against direct enumeration on tiny arrays and sklearn.metrics.roc_auc_score. Do not materialize every positive-negative pair for large users.

### 8.2 Secondary metrics

Compute NDCG@10 using binary gains for positives and zero for negatives, k=min(10,|C_u|), discount 1/log2(rank+1), and an ideal ordering of those same observed candidates. Primary eligibility guarantees at least ten signed candidates, but the general metric function should handle shorter lists. Ties use ascending item ID for this metric; report tie frequency.

Also report observed negative fraction in the top 10, median/p95 fitting and scoring time, peak resident memory, prototype-count distribution, convergence/cap-hit rates, and failure rates. These do not replace AUC or trigger a different success definition. Do not label conditional candidate metrics as full-catalog Recall@K.

### 8.3 Paired uncertainty and precision

For the primary difference, take **10,000 paired bootstrap resamples of confirmation users**, seed 20261029, and report the 2.5% and 97.5% percentiles. Resample each user's entire vector of model metrics together. Also report the simple paired standard error sd(d_u)/sqrt(U) as a diagnostic; substantial disagreement with the bootstrap requires inspection. Use the frozen canonical scoring path, default chunk size 512 unless calibration selected another documented size, and exact stored-score tie convention for primary metrics.

Check numerical variation on the 32 calibration users before freeze. The later float32 diagnostic casts the fixed item/prototype vectors and recomputes geometric scores without refitting; it does not replace primary float64 results. Near-tie reordering is a diagnostic. A material discrepancy discovered before confirmation must be resolved before freezing. A discrepancy discovered after outcome inspection that requires changing the primary computation invokes Section 8.5's invalidation/corrected-exploratory rule; it does not authorize a silent backend or tolerance change. A failure of float32 equivalence alone limits float32 deployment claims, not an otherwise valid canonical float64 experiment.

Compute in bootstrap batches of at most 200 resamples. Do not allocate a resample-by-user-by-item tensor. Deterministic model runs do not need multiple training seeds merely to inflate the sample count.

These intervals describe variation across users conditional on the fixed catalog, cohort construction, metadata, frozen tuning, and observed rating process. They do not include encoder-training uncertainty, global temporal drift, or item-population uncertainty. Shared items and the selected user population limit wider generalization; do not claim that the bootstrap makes the evaluation unbiased.

Before freeze, estimate the paired standard deviation on development users and report the planned half-width:

    w_est = 1.96 * sd_development_difference / sqrt(planned_confirmation_users)

This is a planning estimate, potentially optimistic after tuning, not a power guarantee. A target half-width of 0.005 would distinguish effects separated by roughly half the 0.01 investment threshold. If w_est exceeds 0.01 with all feasible users, explicitly expect that modest effects may remain inconclusive. Continue a feasible locked comparison if it can still detect a consequential larger benefit or harm; do not lower delta or reuse users as independent repeats.

### 8.4 History-defined mechanism strata

Before confirmation, compute on the capped history:

$$
H_u=1-\|\operatorname{mean}(H_u^+)\|_2^2,
\qquad
O_u=\frac1{|H_u^+|}\sum_{p\in H_u^+}\max_{n\in N_u}p^\top n.
$$

H is an embedding-dispersion statistic, not a verified measure of psychological taste diversity. O measures positive-negative embedding overlap, not a clean semantic attribute. Also save the negative-mean norm ||mean(N_u)||; show its distribution and its association with the signed-minus-mean difference as descriptive evidence about the scale confound. Do not select another subgroup threshold from this association.

Freeze the development median of H and median of O as thresholds. Report four confirmation strata defined by above/below these thresholds, including counts and the same selected methods. The directional hypothesis is that the local-negative gain over MeanNegative may be larger when both H and O are high. Also report history negative-count bands 5–8,9–16,17–64.

These are secondary diagnostic analyses with unadjusted intervals explicitly marked as such. Do not turn a favorable subgroup into the primary result, retune by subgroup, or claim a validated adaptive routing policy. If a stratum has fewer than 100 confirmation users, show estimates/counts but treat its mechanistic conclusion as imprecise.

### 8.5 Failure and missing-run policy

Record every attempted user/configuration, elapsed time, status, warning, and exception. Never silently drop failed users or retain only successful seeds.

During development, ordinary defects may be repaired within the allowance and the full affected family rerun on the same fixed development users. A configuration with unresolved solver failures is not eligible for selection; report its failed share and why. If a required family's reasonable configurations all fail, the intended strong comparison is blocked.

During confirmation:

1. Save partial outputs and stop that family on a systemic defect.
2. Determine whether the cause is purely execution/environmental and can be repaired without looking at outcomes. Log what was exposed.
3. An implementation-changing repair after outcomes were inspected invalidates the original confirmatory interpretation. A corrected rerun on the same users is a **corrected exploratory analysis**, unless fresh untouched confirmation users were reserved before freeze. Do not invent a new holdout after observing results.
4. Keep every originally scheduled confirmation user in the denominator. For a missing paired difference, use [-1,1] if both AUCs are missing; [AUC_S-1,AUC_S] if only the comparator is missing; and [-AUC_B,1-AUC_B] if only the signed result is missing. Known differences have equal lower/upper bounds.
5. Bootstrap the per-user lower-bound and upper-bound arrays using the same user resamples as the primary analysis. Use the 2.5th percentile of bootstrapped lower means and 97.5th percentile of bootstrapped upper means as the missing-result confidence envelope. Report the corresponding sample-mean bounds too. Apply decision thresholds to this entire envelope, not just complete cases or point bounds. Show complete-case results only as a labeled diagnostic.
6. If a conclusion changes across this envelope, it is inconclusive. A bug or timeout is not evidence that the scientific hypothesis is false.

Exact failures of data leakage or metric correctness classify the attempted experiment as invalid. Preserve it and report the correction, rather than erasing unfavorable artifacts.

## 9. Target-machine feasibility and accounting

### 9.1 Inspect the actual Mac

Before experimental workloads, capture:

~~~sh
uname -a
uname -m
sw_vers
sysctl -n machdep.cpu.brand_string
sysctl -n hw.memsize
sysctl -n hw.logicalcpu
df -h .
python3 --version
~~~

Record actual chip, RAM, available storage, macOS, native arm64 versus translation, Python, NumPy BLAS, CPU thread counts, package versions, and power state. Use the laptop plugged in and note low-power mode or visible contention. Do not infer memory, storage, supported precision, or package compatibility from “M3 Max.”

Use a virtual environment and native arm64 wheels if supported. The primary implementation uses CPU; CUDA and MPS are not required. This avoids relying on unverified Metal support or GPU float64. If the machine is not the requested Mac, prepare code, documentation, data-source checks, and analytic derivations as possible, but do not run or claim the target experiments elsewhere.

### 9.2 Results-first execution policy and hard non-time ceilings

The user-authorized 31 October 2026 amendment supersedes elapsed-time stop rules.
Append an explicit, validated `results_first` policy event to the permanent ledger
before freeze or confirmation exposure; do not change the original resource hash,
reset history, inflate configured caps, or discard failed attempts. Unamended
legacy ledgers retain bounded behavior. Adoption is durable and idempotent.

Original timing numbers below are historical planning estimates, not measured M3
performance. Under results-first, actual setup/experimental/stage/repair elapsed
time, allocations, 1.5 forecast multiplier and reporting reserve remain recorded
and advisory. They cannot reject, shrink or stop the study. Ordinary material
implementation repairs remain logged and reviewed without the former two-cycle
quota preventing completion. Fixed scientific solver/prototype repair rules,
numerical iteration limits and all scientific gates remain enforced.

| Resource | Ceiling / rule |
|---|---|
| Experimental wall time | Historical 240-minute estimate; advisory under results-first. Count preprocessing, calibration, tuning, failures, repairs, comparisons, diagnostics, analysis and packaging cumulatively. |
| Acquisition/setup waiting | Historical separate 45-minute estimate; advisory under results-first. No paid or remote experimental compute. |
| Peak process-tree resident memory | min(8 GiB,25% of installed RAM). Use one worker initially. |
| Free-memory check | Require at least 4 GiB of estimated available headroom before a heavy stage; log the measurement method. |
| Project disk growth | <=6 GiB, including environment, model cache, raw/derived data, temporary arrays, outputs, and logs. Require 12 GiB free initially. |
| Model/config/user evaluations | <=100,000 cumulatively, including tuning, repairs, diagnostics, and repeated timing invocations. Count attempted failures too. |
| Hyperparameter additions | <=14 entries beyond the 78-entry declared grid, once and before freeze. |
| Repair allowance | Historical 20-minute/two-material-cycle estimate; advisory for ordinary implementation repairs. Preserve costs, failures and review evidence. |
| Reporting reserve | Historical 30-minute planning reserve; advisory under results-first. Complete evidence preservation, analysis and reporting. |

Downloading libraries and editing source are not the same as benchmark runtime. **Estimated end-to-end effort:** setup 30–45 minutes, implementation/correctness work 4–8 hours, experimental execution typically 1–4 hours, interpretation/reporting 1–2 hours; roughly one to two working days depending on access and repair needs. These are effort estimates, not promises. The first informative computational pilot should take approximately 45–60 minutes once code and dependencies exist.

Use a persistent ledger of stage elapsed seconds, attempted evaluations, peak memory, and bytes. Switching dataset, retrying a model, changing a stage name, or restarting the process does not reset caps. Record setup waiting separately. Determine optional work from scientific applicability and actual non-time resource availability.

### 9.3 Scaling and calibration rules

The full primary float64 embedding array is approximately 3,883 x 384 x 8 =11.9 million bytes; float32 storage is approximately 6.0 MB. A 512-by-64 negative-similarity block is 256 KiB in float64. A 192-by-192 kernel is about 288 KiB. Framework/model overhead and retained solver objects, not the base embedding matrix, should dominate memory.

These arithmetic estimates do not establish total RSS or runtime. Transformer inference can dominate preprocessing, while RBF fitting and the tuning grid can dominate experimental time. Measure them.

G1's correctness fixtures are independent of text encoding. G2 owns construction of the complete embedding cache and proceeds sequentially:

1. Load the pinned encoder, time 128 representative title/genre inputs, and run the 32-item repeatability check. Select calibration metadata by deterministic item-ID hash, without rating outcomes.
2. Project full-catalog preprocessing cost as a diagnostic. When actual non-time resource checks pass, encode all 3,883 metadata items once in item-ID order and save/verify the shared cache. Count the initial 128 and repeated 32 encodings as additional calibration work; re-encoding them in the complete cache is intentional and budgeted.
3. With that complete cache, calibrate 32 development users selected from the fixed digest order to cover support-size quartiles and long evaluation candidate sets. Include all families and high-cost endpoints.

The calibration command performs all three steps. Develop must refuse to run unless a complete cache has the expected item-ID mapping, pinned model revision, template hash, dimensions, finite values, and normalization checks. Measure:

- data parsing and validation;
- cold model load separately from encoding;
- encoding throughput and peak RSS;
- prototype construction, linear fit, RBF fit, and exact scoring;
- output-write costs and memory with an actual representative run.

Project costs using measured per-item/per-user work and candidate/history sizes, not chip marketing throughput. For ranking matrices, include the d*m*(K+J) scale. For SVMs, use measured support-size sensitivity rather than claiming a fixed cubic or quadratic law for this exact workload.

Retain measured forecasts and the historical 1.5 safety multiplier as diagnostics.
Under results-first they cannot reject stages, remove optional work, change chunks
or shrink cohorts for elapsed-time headroom. Use all 418 development and 1776
confirmation users in the original full-SHA256 digest order, determined before
outcomes. Keep the canonical measured scoring chunk unless a measured non-time
resource issue requires a globally valid policy and verified chunk invariance.
Memory-bound embedding changes require measured verification. Never remove strong
baselines, selectively shrink feedback, relax solver tolerances or change metrics.
The separate access-only 100K fallback retains all 62 development and 225
confirmation users and its 1,682-item catalog; time forecasts do not authorize it.

The first native stop at G2 was a forecast-only historical outcome. Preserve its
report, `results/calibration.json` bytes/path/hash, failed attempt and dependencies.
`resume-calibration` verifies that original terminal publication, historical G1,
unchanged scientific source/settings, completed workload/journal, accepted
numerical checks and the complete cache, then publishes a versioned calibration
artifact and successful stage under current G1. Its new admissions reserve zero
evaluations; the original 1056 evaluations remain counted once. Archive original
correctness bytes before replacing active G1. Consumers resolve calibration from
the completed stage artifact, preserving original publications. No re-encoding or
refitting occurs in this adoption. Hard memory/disk/evaluation violations or
unresolved scientific defects still block continuation and must remain explicit.

### 9.4 Timing and memory protocol

Use time.perf_counter_ns around clear boundaries. Report cold preprocessing once, per-user fit, per-user score, and total evaluation including writes. Cache reuse is allowed across lambda values, but report both cached implementation cost and standalone formula operation counts.

For selected-method microbenchmarks, use the same 32 development users and all 3,883 metadata item vectors. Run one warm-up and five timed repetitions, rotating method order deterministically. Fit timing starts from that user's raw capped vectors; scoring timing starts with a fitted user representation and a resident candidate matrix. Include dot products, reductions, and stable sorting in end-to-end rank latency; optionally separate score-only timing. Do not include downloads in warm scoring.

Use the fixed calibrated CPU thread policy: one BLAS thread and one inference worker throughout the results-first continuation. Apply the same policy to compared methods and record actual library settings. Do not oversubscribe user processes and BLAS threads.

Measure process-tree RSS with psutil sampling at 100 ms and OS maximum-RSS facilities where available; report the maximum and the method. Python tracemalloc alone misses native arrays/framework memory. Timing runs execute serially without unrelated experimental processes. Report median and p95 across users, and repeat variability. No accuracy-seed confidence interval should be inferred from warm timing repeats.

Give cold total cost and an explicit amortization scenario of 1,000 user scoring requests, separating shared item embedding cost, per-user fit cost, and request scoring. Do not extrapolate measured small-catalog latency into a million-item production claim.

## 10. Evidence gates and autonomous next actions

The historical stage estimates below total 230 minutes, with 10 minutes of scheduling overhead. All elapsed allocations and reserve rules in this table are advisory under the results-first amendment; non-time resource and scientific gates remain authoritative.

| Gate | Question, prerequisites, and minimum work | Evidence / budget | Decision and next action |
|---|---|---|---|
| G0 Constructive feasibility | Does the actual Mac, accessible dataset/model, and license-compatible local workflow support the fixed study? Inspect environment; acquire data; reproduce schema/cohort counts; resolve encoder revision. | environment.json, data_manifest.json, cohort_audit.json; 10 experimental minutes | Continue if valid. Two bounded access attempts then 100K fallback if applicable. Unavailable indispensable hardware/model: finish unaffected work and report blocker. |
| G1 Tiny correctness | Are geometry, baselines, splitting, and metrics valid? Implement only the scorer, prototype constructor, essential fits, and independent tests. | tests, oracle outputs, analytic counterexamples; 10 minutes | Continue only if all mandatory checks pass. Repair within shared allowance; unresolved defect means invalid experiment/stop. |
| G2 Resource calibration | Can preprocessing and all comparator families finish within caps? Prerequisite G1. Run the 128-item encoder probe, admit/build the complete cache, then calibrate 32 development users as in Section 9.3. | calibration.json, verified embedding cache, memory/time traces, planned sample sizes; 20 minutes | Verify numerical validity, full cache and non-time resources; use the full primary cohort. Keep the 1.5 forecast as advisory diagnostic. Preserve the original forecast-only stop as history. |
| G3 Mechanism pilot and development | Does the setup detect the known effect, fit meaningful references, and support a fair locked study? Run exact controls, development grid, bounded endpoint expansions, and precision estimate. | all development runs, chosen parameters/B*, failure table, protocol.lock.json; 40 minutes | A small noisy advantage is not a stop rule. Continue if valid and resource-feasible; if real signal is weak, retain question and limits. Missing strong baseline or unresolved numerical issue blocks confirmation. |
| G4 Locked comparison | Does S* clear the fixed threshold against B* on untouched evaluation outcomes? Prerequisite frozen protocol. Evaluate all selected models and matched controls. | raw per-user metrics and candidate scores, model diagnostics, primary paired interval; 80 minutes | Analyze once. Do not retune after outcomes. A fatal protocol defect invokes invalid-experiment rules. Preserve all partial evidence and bounds on actual non-time resource failure; elapsed time cannot stop comparison. |
| G5 Bounded diagnosis | Is an observed effect consistent with negative aggregation and stable numerically? Reuse strata; precision check; conditional contamination; selected-method timing. | diagnostic tables/figures, latency and RSS; 20 minutes | Perform only applicable diagnostics that fit. Do not invent rescue methods. Elapsed headroom cannot skip applicable diagnostics; preserve actual non-time resource limitations. |
| G6 Diagnosis and report | What conclusion is justified by the reached stages? Regenerate tables/plots, check claims, package code and local reproducibility evidence. | report.md, results tables, figures, README, manifests; 30 minutes protected | Complete the outcome even for early stop. State unperformed work and one justified next step, if any. |
| Shared repair reserve | Ordinary bugs, setup-compatible API fixes, fixed scientific convergence repairs; ordinary material cycle quota advisory. | repair log and rerun IDs; 20 minutes | Record cumulative time and cycles. Post-freeze changes follow invalidation rules; never erase original attempts. |

Preserve all historical budget allocations and transfers. Under results-first no reallocation is needed for elapsed headroom; retain forecasts and actual cumulative costs. Do not use an unspent expensive-stage budget to launch a new research direction.

## 11. Repository, interfaces, and intended commands

### 11.1 Minimal stack

Use Python 3.11 or 3.12 in a virtual environment, NumPy, SciPy, scikit-learn, torch, sentence-transformers, huggingface-hub, psutil, matplotlib, and pytest. JSON/JSONL/CSV/NPY suffice for local storage; use standard-library parsing where straightforward. No database, web service, dashboard, job framework, ANN package, CUDA, paid inference API, or distributed execution is needed.

Exact target-compatible dependency versions were **not installed or validated during planning**. Resolve a compatible set on the actual Mac, run smoke tests, and freeze exact versions and model revision before confirmation. Do not quote the live documentation's version as the installed version. Pin the environment after the correctness gate; use public CPU-capable releases and record any compatibility adjustment.

Keep .venv and a dedicated .cache directory inside the project. Pass a project-local model cache directory through the supported cache_folder/cache_dir interface; set any required Hugging Face cache locations to that scoped directory. Install with pip --no-cache-dir or a project-local pip cache. Include all project caches and temporary downloads in disk accounting and .gitignore. Do not clear or modify user-global caches. If an existing external cache is intentionally reused, record its footprint and disclose that fresh-download time/space was not measured.

### 11.2 Files and responsibilities to create

~~~text
signed_preference_poc/
  pyproject.toml
  requirements.lock.txt
  README.md
  config/study.json
  src/signed_study/
    __init__.py
    __main__.py
    data.py          # acquisition, checks, episodes, caps, user split
    embed.py         # fixed text template, revision, local cache
    models.py        # prototypes, scores, sklearn reference adapters
    metrics.py       # independent AUC/NDCG, paired bootstrap
    run.py           # stage execution, freezes, budget/failure ledger
    report.py        # tables, plots, report from saved outputs
  tests/
    test_geometry.py
    test_data_and_firewall.py
    test_metrics_and_baselines.py
  data/              # local only, ignored by public git
  results/
    environment.json
    data_manifest.json
    cohort_audit.json
    calibration.json
    budget_ledger.jsonl
    protocol.lock.json
    runs.jsonl
    user_metrics.csv
    scores/          # compressed per-user/model scores and candidate IDs
    tables/
    figures/
    report.md
~~~

The agent may combine small modules or use ordinary scripts if that is simpler, while preserving these responsibilities and interfaces.

Core contracts:

~~~text
load_dataset(config) -> ratings, metadata, manifest
make_episodes(ratings, config) -> episodes, cohort_audit
encode_items(metadata, config) -> item_id_to_row, unit_embeddings, manifest
fit_user(method, episode.history, embeddings, parameters) -> model, diagnostics
score_user(model, candidate_ids, embeddings) -> scores
evaluate_user(scores, candidate_labels, item_ids) -> metrics
run_stage(stage, config, ledger) -> immutable run records
render_report(results_directory) -> tables, figures, report
~~~

An episode contains user ID, cutoff timestamp, split role, selected positive/negative support IDs, candidate IDs, and a separately loaded candidate-label array. Model fitting must never accept candidate labels. Test this interface boundary.

### 11.3 Configuration contents

Create a single JSON configuration with these fixed values: dataset ml-1m; model ID; resolved revision; exact template; split numerator 7/denominator 10; eligibility [10,5,5,5]; support caps [128,64]; hash prefix signed-study: and modulus 5; K grid [1,2,4]; Lambda grid [0,.25,.5,1,2,4,8]; linear and RBF grids; prototype iteration/tolerance rules; CPU precisions; bootstrap count 10,000/seed 20261029; contamination seeds; delta 0.01; resource caps; and output paths.

The unresolved model revision and installed versions are **acquisition outputs**, not an invitation to choose a different model. Populate them before G2, then require exact matches on resume. Store resolved configuration and its SHA256 with every stage.

### 11.4 Intended commands

The following interfaces are requirements for the agent to implement, **not commands already tested by the planner**. Use an existing suitable Python executable; if absent, report the required installation rather than assume macOS has it.

~~~sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --no-cache-dir --upgrade pip
python -m pip install --no-cache-dir -e .
python -m signed_study inspect --config config/study.json
python -m signed_study acquire --config config/study.json
python -m signed_study audit --config config/study.json
python -m pytest -q
python -m signed_study calibrate --config config/study.json
python -m signed_study smoke --config config/study.json --development-users 8
python -m signed_study develop --config config/study.json
python -m signed_study freeze --config config/study.json
python -m signed_study confirm --lock results/protocol.lock.json
python -m signed_study diagnose --lock results/protocol.lock.json
python -m signed_study report --results results
~~~

The smoke run uses only development users, scores a small known candidate set, and counts against budget. Each stage must refuse an invalid prerequisite, mismatched lock, exhausted resource cap, or accidental evaluation-label access during development. Resume must reuse immutable completed outputs by exact hashes and append attempts, not overwrite failure history.

Do not build a generalized experiment framework. A short command dispatcher and explicit stage functions are enough.

## 12. Required outcomes and artifacts

### 12.1 Outcome taxonomy

| Outcome | Evidence required |
|---|---|
| Supported contribution | Valid implementation, fair required comparisons, frozen confirmation, supported specified effect within scope. Distinguish primary comparator result from the complete-suite result and mechanistic attribution. |
| Informative negative result | Valid comparison whose interval rules out the predefined useful advantage or demonstrates harm; alternatively a verified analytic obstruction to a claimed application. Nonsignificance or a stronger baseline's higher point estimate alone is insufficient. |
| Practical infeasibility | Measured target-machine cost or supported scaling prevents the specified study/application within caps. Identify the dominant cost and smallest meaningful scale. |
| Inconclusive evidence | Intervals span the threshold, data/representation coverage is inadequate for the intended wider claim, an indispensable resource is absent, or failures leave decision-changing uncertainty. |
| Invalid experiment | Leakage, baseline defects, implementation errors, or measurement problems invalidate comparative interpretation. Preserve attempted evidence and state exactly what cannot be concluded. |

A failure of this one metadata/encoder setting does not prove that embeddings lack useful preference information or that signed methods fail everywhere. A success does not prove that observed-rating AUC predicts online satisfaction. A timeout is not a scientific refutation.

### 12.2 Report contents

Write results/report.md as a standalone research report containing:

1. The question, final outcome label, and practical recommendation.
2. What was actually executed, on which hardware, and which gates were not reached.
3. Data/model provenance, eligible-user funnel, temporal-burst limitation, feedback caps, and frozen protocol.
4. Primary Delta, its interval, delta=0.01, the selected B*, and all required comparator results.
5. Matched-ablation results, including K=1 SignedMax, and whether lambda>0 and K>1 were selected; distinguish selection from evidence that a component helps.
6. History-stratum diagnostics, clearly secondary, and applicable robustness results.
7. Full cost accounting: setup, embedding, fitting, tuning, scoring, failures, peak memory, disk, and assumed amortization.
8. Leakage/correctness evidence, numerical tolerances, baseline failures, missing-result bounds, and every material deviation.
9. Closest prior art and the exact difference or limitation established, with no first-ever claim.
10. What remains unresolved and, at most, one evidence-justified next step.

Useful final visuals are a paired per-user difference distribution, an interval comparison of selected methods, and a compact cost table. Do not use decorative embedding plots as proof of preference structure.

### 12.3 Artifacts proportionate to stages reached

If implementation occurred, deliver runnable code, exact environment lock, README commands, tests, and source/config hashes. If experiments occurred, preserve raw runs, local per-user scores/metrics, timing/memory data, failures, analysis scripts, and regenerable tables/figures. Keep per-item raw data and derived content out of a public bundle; document how the local researcher can regenerate them.

For an analytic or constructive early stop, deliver the verifiable derivation/counterexample or acquisition/environment evidence and a concise scoped report. Do not fabricate empty benchmark tables or claim unperformed validation to satisfy an artifact checklist.

## 13. Research coverage and supporting references

### 13.1 Search scope and source fidelity

Research was conducted on 29 October 2026. It covered multi-interest maximum scores, fixed item embeddings with clustered histories, positive/negative relevance feedback, nearest-class and normalized-distance formulations, maxout approximation results, current signed semantic-interest work, explicit-rating evaluation, MovieLens provenance, and local CPU implementation documentation.

Backward references from MIND and PinnerSage led to MaxMF; the nearest-case feedback literature supplied the close distance-ratio baseline; current searches found PRIME. Source reading concentrated on definitions, equations, assumptions, construction, and experimental scope that affect this project. The critical nearest-case and PRIME equations were inspected in PDF page images when text extraction was ambiguous. Independent reviewers separately investigated prior art, mathematics, and data/execution feasibility.

This was a decision-focused search, not an exhaustive global novelty search. Some source URLs timed out on an initial request; accessible author/arXiv/publisher copies were used. Science Concierge's publisher text was available when a PMC route was obstructed. No inaccessible paper is treated as verified, and no existing implementation was executed as part of literature review.

Unless marked as a planning observation or our derivation, source-supported statements refer to the cited work's scope. Reported historical improvements do not transfer to this study. The sources below are retrieval instructions as well as citations; exact pages refer to the linked PDF unless a printed page is explicitly identified.

### 13.2 References

**[R1] Jason Weston, Ron J. Weiss, and Hector Yee. 2013. “Nonlinear Latent Factorization by Embedding Multiple User Interests.”** RecSys 2013, pp. 65–68. DOI: <https://doi.org/10.1145/2507157.2507209>. Author PDF: <https://www.ee.columbia.edu/~ronw/pubs/recsys2013-usermax.pdf>. Critical locations: Section 3, Equation (2), Algorithms 1–2, PDF page 2. The maximum-over-user-factors score and frozen-item second-stage procedure were read directly.

**[R2] Aditya Pal, Chantat Eksombatchai, Yitong Zhou, Bo Zhao, Charles Rosenberg, and Jure Leskovec. 2020. “PinnerSage: Multi-Modal User Embedding Framework for Recommendations at Pinterest.”** KDD 2020. DOI: <https://doi.org/10.1145/3394486.3403280>. <https://arxiv.org/abs/2007.03634> and <https://arxiv.org/pdf/2007.03634>. Critical locations: Section 2 design choices; Section 3.1, Algorithms 1–2 and Equations (2)–(3), PDF pages 4–5; Section 5 evaluation. Fixed PinSage vectors include interaction-graph information; this is not identical to content-only sentence embeddings.

**[R3] Giorgio Giacinto and Fabio Roli. 2004 conference / 2005 proceedings. “Instance-Based Relevance Feedback for Image Retrieval.”** Advances in Neural Information Processing Systems 17. <https://proceedings.neurips.cc/paper/2004/file/5acdc9ca5d99ae66afdfe1eea0e3b26b-Paper.pdf>. Alternate: <https://papers.neurips.cc/paper_files/paper/2004/file/5acdc9ca5d99ae66afdfe1eea0e3b26b-Paper.pdf>. Critical locations: Sections 2–3, Equation (1), PDF pages 2–4. Assumes useful instance proximity and combines nearest relevant/nonrelevant distances; reported image-retrieval experiments do not guarantee recommendation improvement.

**[R4] Anqi Wang, Jianye Xie, Weiming Liu, Rong Jiang, Lianyong Qi, Haolong Xiang, Xiaolong Xu, Wenmin Lin, Yang Zhang, and Xiaokang Zhou. 2026. “Joint Multi-Modal Multi-Interest Profiling and Preference-Grounded Reasoning for Explainable Recommendation.”** IJCAI 2026, pp. 3052–3060. DOI: <https://doi.org/10.24963/ijcai.2026/339>. <https://ijcai.org/proceedings/2026/0339.pdf>. Critical location: Section 3.3, Equations (11)–(13), printed page 3056/PDF page 5. Frozen CLIP encodes positive/negative descriptions; signed matching chooses an interest for a larger reasoning pipeline. Its paired-interest formulation differs from separate maxima over our item histories.

**[R5] Ian J. Goodfellow, David Warde-Farley, Mehdi Mirza, Aaron Courville, and Yoshua Bengio. 2013. “Maxout Networks.”** ICML, PMLR 28(3), pp. 1319–1327. <https://proceedings.mlr.press/v28/goodfellow13.html> and <https://proceedings.mlr.press/v28/goodfellow13.pdf>. Critical locations: Section 3 definition; Section 4, Propositions 4.1–4.2 and Theorem 4.3, PDF pages 2–3. Arbitrarily many unrestricted affine pieces are allowed; this is not a guarantee for the finite constrained constructor in this brief.

**[R6] Chao Li, Zhiyuan Liu, Mengmeng Wu, Yuchi Xu, Huan Zhao, Pipei Huang, Guoliang Kang, Qiwei Chen, Wei Li, and Dik Lun Lee. 2019. “Multi-Interest Network with Dynamic Routing for Recommendation at Tmall.”** CIKM 2019. DOI: <https://doi.org/10.1145/3357384.3357814>. <https://arxiv.org/pdf/1904.08030>. Critical locations: Section 3.1 Equation (3), PDF page 3; Sections 3.3–3.4. Learned multi-interest matching is precedent, not evidence for frozen signed-content ranking.

**[R7] Yukuo Cen, Jianwei Zhang, Xu Zou, Chang Zhou, Hongxia Yang, and Jie Tang. 2020. “Controllable Multi-Interest Framework for Recommendation.”** KDD 2020, pp. 2942–2951. DOI: <https://doi.org/10.1145/3394486.3403344>. <https://arxiv.org/pdf/2005.09347>. Critical location: Section 3.3, Equations (11)–(12), PDF page 5. Distinguishes max-interest matching and relevance/diversity aggregation. Official implementation identified in the paper: <https://github.com/THUDM/ComiRec>; not executed in planning.

**[R8] Titipat Achakulvisut, Daniel E. Acuna, Tulakan Ruangrong, and Konrad Kording. 6 July 2016. “Science Concierge: A Fast Content-Based Recommendation System for Scientific Publications.”** PLOS ONE 11(7):e0158423. <https://doi.org/10.1371/journal.pone.0158423>. Full text: <https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0158423>. Critical locations: Section 2.6 Equations (4)–(5); Section 2.7's simulated topic preferences; Section 3.1/Figure 3 and Section 4's negative-feedback discussion. A negative centroid weight of zero was favored in the studied topic simulation; that is not a general theorem about dislikes.

**[R9] F. Maxwell Harper and Joseph A. Konstan. December 2015. “The MovieLens Datasets: History and Context.”** ACM Transactions on Interactive Intelligent Systems 5(4), Article 19. <https://doi.org/10.1145/2827872>. Author manuscript: <https://files.grouplens.org/papers/harper-tiis2015.pdf>. Critical locations: Section 3.2 “Limitations,” especially PDF page 15 (zero-based page 14), and Section 3.3 usage guidance. The accessible author manuscript has placeholder publication metadata; the DOI and final bibliographic details above come from the official dataset README.

**[R10] GroupLens Research. “MovieLens 1M Dataset README.”** Stable dataset released 2003; README accessed 29 October 2026. <https://files.grouplens.org/datasets/movielens/ml-1m-README.txt>. Landing page: <https://grouplens.org/datasets/movielens/1m/>. Critical locations: Summary, Usage License, Ratings File Description, Movies File Description. Fallback source: **“MovieLens 100K Dataset README,” 1998**, <https://files.grouplens.org/datasets/movielens/ml-100k-README.txt>. Stable-versus-latest guidance: <https://grouplens.org/datasets/movielens/>.

**[R11] Sentence Transformers. “sentence-transformers/all-MiniLM-L6-v2” model card.** Live documentation, accessed 29 October 2026; immutable model revision must be recorded on acquisition. <https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2>. Read Usage, Intended Uses, and Training Procedure. API documentation: <https://www.sbert.net/docs/package_reference/sentence_transformer/model.html>. CPU/device documentation: <https://sbert.net/docs/sentence_transformer/usage/efficiency.html>. These specify the supported public encoder interface; they do not verify the target Mac installation.

**[R12] scikit-learn developers. “LogisticRegression.”** Official live API documentation, accessed 29 October 2026. <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html>. Critical fields: regularization, solver, C, class_weight, tol, max_iter, decision_function; current deprecation notes for penalty. Exact installed version must be pinned and validated.

**[R13] scikit-learn developers. “SVC” and “Support Vector Machines.”** Official live documentation, accessed 29 October 2026. <https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html> and <https://scikit-learn.org/stable/modules/svm.html>. Critical fields: RBF kernel, C/gamma, class weights, stopping status, cache/memory, decision_function, computational scaling, and probability-calibration overhead.

**[R14] Yuta Saito, Suguru Yaginuma, Yuta Nishino, Hayato Sakata, and Kazuhide Nakata. 2020. “Unbiased Recommender Learning from Missing-Not-At-Random Implicit Feedback.”** WSDM 2020, pp. 501–509. DOI: <https://doi.org/10.1145/3336191.3371783>. <https://arxiv.org/pdf/1909.03601>. Abstract/introduction and the exposure/positive-unlabeled setup explain why missing interactions do not establish dislikes. This brief does not apply its propensity estimator: the required exposure assumptions/information are absent. The resulting observed-rating estimand remains conditional, not debiased.

## 14. Review findings and disposition

Three distinct review passes were performed during planning: mathematical/novelty review, experimental-design review, and execution review. Their material findings are recorded here so that the implementation does not reintroduce repaired defects.

| Challenge | Disposition |
|---|---|
| A mathematical paper is missing; a theorem might be nonconstructive or inapplicable. | No paper-specific guarantee is claimed. The finite equation is directly executable; maxout's unrestricted theorem is explicitly not inherited. |
| The proposed mechanism may already be known. | Exact nearest-distance equivalence established; MaxMF, PinnerSage, nearest-case feedback, and PRIME cited. Contribution narrowed to a controlled diagnostic. |
| A win could be caused only by receiving negative labels or more interests. | Equal capped histories, MeanNegative ablation, matched-K/lambda controls, and conventional strong references retained. |
| A weak linear baseline could manufacture a nonlinear advantage. | Learned linear logistic, RBF SVM, all-exemplar margin, and relative-distance baselines are required and tuned. |
| A low negative-mean magnitude could cause unfair under-tuning. | Raw mean specified; lambda grid extends to 8, with a bounded endpoint expansion to 16. |
| A matched max-versus-mean gain might be attributed too specifically to locality. | Attribution narrowed to the complete aggregation substitution, including user-specific magnitude differences; negative-mean norms are logged. |
| Chronological MovieLens splits might be interpreted as future viewing. | Rating-burst audit and source warning included. Estimand restricted to later rating submissions. |
| Floating-point arithmetic can alter cohort membership. | Replaced int(0.7*n) with exact (7*n)//10; final count corrected to 2,194 eligible,418 development,1,776 confirmation. |
| All unobserved items might be treated as true negatives. | Only explicit high/low held-out ratings are scored for accuracy. No catalog utility or exposure correction is claimed. |
| Positive-only retrieval might hide failures. | Exact scoring adopted; a two-candidate counterexample documents the recall obstruction. |
| Preprocessing or hidden intermediates may dominate. | Encoding included in budget; chunking and sequential per-user fits; process-tree RSS and target calibration required. |
| Full encoding lacked a clear place in command order, and global caches could escape quotas. | G2 now explicitly admits and builds the complete cache; real-user smoke follows it. Project-local caches make disk accounting concrete. |
| Resizing and endpoint additions could differ between implementations or omit a strong RBF corner. | Fixed scale-pair order, explicit 100K exemption, and one complete endpoint expansion with a 14-entry ceiling. |
| Only favorable synthetic examples may work. | Exact fixtures are correctness evidence only. Real confirmation remains necessary for the practical claim. |
| A strong baseline may erase the gain. | Valid anticipated outcome; no post-hoc redefinition of the primary comparator or task. |
| A pilot may be noisy or the dataset inaccessible. | Precision forecast, fixed threshold, non-rescuing 100K access fallback, and explicit inconclusive outcome. |
| A post-freeze bug may require rerunning. | Preserve failed runs; corrected same-holdout analysis is exploratory after outcome exposure. |
| Missing-run mean bounds ignored sampling uncertainty. | Primary decisions now use a paired-bootstrap confidence envelope over missing-result bounds. |
| Exact chunk/ordering invariance could falsely reject valid BLAS results. | Tolerance-based score checks and near-tie-aware ordering checks replace bitwise equivalence. |
| Finite precision or resource limits may change results. | CPU float64 geometry, explicit degeneracies, independent oracle, precision diagnostic, and hard cumulative non-time caps. |
| The main hypothesis may be false. | Stop after valid evidence, publish the scoped limitation in the report, and recommend further work only if a specific unresolved cause warrants it. |

Residual risks are real: title/genre features may be too weak; ratings are self-selected and context-limited; public encoder pretraining is not fully auditable; a single catalog cannot establish broad generality; and actual Mac compatibility, speed, and memory remain unmeasured. These risks are either early gates or explicit boundaries on the conclusion. They are not reasons to manufacture a positive outcome.

**Final execution instruction:** carry out the valid authorized stages autonomously, preserve every material attempt, follow the frozen decision rules, and end with the strongest conclusion the evidence actually supports.
