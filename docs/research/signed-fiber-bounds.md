# Signed residual fibers: scope and checks

This is a reconstruction of the mathematical content supplied with the retrieval
contribution, not a historical novelty claim or a new predictive method. The
production retrieval module does not use projection screening or a convex solver.

For nonempty finite positive and negative sets P,N, penalty λ≥0, orthogonal
projector Q, R=I−Q, y in im Q and r≥0, define
s(x)=max(pᵀx:p∈P)−λ max(nᵀx:n∈N). The exact extrema over the residual **ball**
x=y+z, z∈ker Q, ‖z‖≤r are

\[
U(y,r)=\max_{p\in P}\min_{a\in\lambda\operatorname{conv}N}
 [(p-a)^Ty+r\|R(p-a)\|],
\]

\[
L(y,r)=\min_{n\in N}\max_{b\in\operatorname{conv}P}
 [(b-\lambda n)^Ty-r\|R(b-\lambda n)\|].
\]

To see the upper identity, enumerate the finite positive branches, express the
negative maximum using its convex hull, interchange the compact convex minimax
problem, and apply the Euclidean ball support function. The lower identity
interchanges the negative branches and uses the corresponding infimum support
function. This invokes established convex minimax results, especially
[Sion (1958)](https://msp.org/pjm/1958/8-1/pjm-v8-n1-p14-p.pdf).

For unit candidates, real arithmetic gives r=√(1−‖y‖²). An implementation should
form residuals directly instead of subtracting nearly equal squared norms. Let
h=dim ker Q. When r>0 the sphere requires h≥1. The upper ball endpoint is attained
on that sphere if h≥|N|; the lower endpoint is attained if h≥|P|. A nonzero
residual direction that improves all relevant affine branches moves an interior
optimizer to the boundary. This direction argument has an antecedent in
[Wang and Xia (2016)](https://arxiv.org/abs/1604.02212). These are sufficient
conditions for endpoint attainment, not conditions for valid ball-bound
screening. For h=1 the sphere has just two points and need not attain intermediate
values; for r=0 the fiber is a singleton.

The dimensions are worst-case sharp. Let v₀,…,vₕ be centered unit regular-simplex
vertices in an h-dimensional residual space and e an orthogonal unit vector.
Take Q projecting on e and y=0,r=1. With P={e}, N={vᵢ}, λ>0, the ball upper
endpoint is 0 but the sphere upper endpoint is −λ/h. Swapping P,N gives ball
lower endpoint 0 and sphere lower endpoint 1/h. These are analytic examples;
they require no general numerical optimizer.

Any feasible negative convex mixture is a valid upper witness; optimality of the
mixture is unnecessary. Its mean must remain **unnormalized**. Cancellation is
captured by ‖R(p−a)‖, rather than a sum of independent residual norms. For
P={(1,0)}, N={(.8,.6),(.8,−.6)}, λ=1.25 and Q=0, s(x)=−.75|x₂|. The negative
midpoint gives upper bound 0, while either vertex gives .75. An arbitrary subset
of genuine negatives also gives an upper bound if its actual inner products are
evaluated. Approximate witness selection preserves feasibility; approximate
witness evaluation requires a separate valid error allowance.

Exactly scoring at least k distinct seed candidates supplies a lower bound τ on
the true kth score. Only an upper bound **strictly below** τ permits exclusion.
Keep equality and apply descending stored score, then ascending integer ID.
Kernel caching and safe threshold selection are established principles:
FEXIPRO and LEMP are antecedents identified in the supplied novelty audit.
[DoSA](https://arxiv.org/html/2610.13990v1) uses bounds, seed thresholds and
selective evaluation for a different signed nonlinear attention score. No
historical priority is claimed here.

Real-arithmetic inequalities do not certify floating-point implementations.
The supplied projection sketch used direct residuals, an orthogonality-defect
allowance and a heuristic roundoff pad, not directed interval arithmetic. Its
reported 36 cases/144,000 sampled sphere points, maximum primal-dual discrepancy
3.23e−11 and no sampled violations are inherited evidence whose original raw
runs are unavailable. They do not certify every machine input. Supplied
projection ranks 16/32/64 were slower than exhaustive dense scoring. Projection
screening is therefore deferred; strict Gram top-k calls the full reference.

`python scripts/audit_signed_fiber_bounds.py` runs the reconstructed cancellation,
simplex and feature-lift checks. It is an analytic regression aid, not a numerical
proof of the general theorem or a reproduction of the supplied optimizer audit.

## Hypotheses requiring another study

For full-support iid negative draws on a finite catalog, the maximum negative
cosine approaches one, making that penalty nearly constant. This implies neither
AUC→.5, monotonic harm, nor an explanation of this capped MovieLens result.

Normalizing a spherical centroid can break invariance to an added uninformative
constant feature. With positive supports (±√3/2,1/2), negative (1/5,2√6/5), λ=1,
and A=(0,1), B=(1,0), original scores are approximately .020204 and −.2.
Lift every vector by Tₐ(z)=(√(1−a²)z,a), a²=.5, and refit the normalized prototype:
the scores become approximately −.041215 and .032456, reversing order. Centering,
unnormalized centroids, soft maxima and density ratios remain exploratory ideas.
None replaces the frozen scorer or makes exposed confirmation users a fresh holdout.
