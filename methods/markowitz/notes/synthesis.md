# Synthesis — Markowitz mean–variance / efficient frontier / critical-line algorithm

## Pain point / research question
Stage-two portfolio choice: beliefs about securities (means, variances, covariances) are given;
how do you turn them into a *choice of portfolio* X = (X_1,...,X_N), X_i = fraction in security i,
sum X_i = 1, X_i >= 0 (no short sales)?

The prevailing "rule" was: maximize the discounted/anticipated expected return
R = sum_i X_i R_i (Williams 1938, dividend-discount / present-value). Markowitz shows this is wrong
*as a guide*: since R is a weighted average of fixed per-security discounted returns R_i with
weights X_i >= 0, sum X_i = 1, it is maximized by putting everything in the single security with
the largest R_i (corner). It NEVER prefers a diversified portfolio to all undiversified ones.
But diversification is observed and sensible — so the rule must be rejected.

A patch people reached for: "diversify among all securities that share the maximum expected return;
the law of large numbers will make the realized yield ≈ expected yield." Reject this too: returns
are too intercorrelated, the LLN does not apply across securities, diversification cannot eliminate
all variance. (Williams 1938 believed risk could be fully diversified away — pp.67-70; others were
"seduced by the law of large numbers" — Bernoulli 1713. Rubinstein 2002.)

## The central insight (load-bearing)
Variance of a weighted sum is NOT the weighted sum of variances. For R = sum_i X_i R_i,
  E = sum_i X_i mu_i           (linear, cheap)
  V = sum_i sum_j X_i X_j sigma_ij  = sum_i X_i^2 sigma_ii + sum_i sum_{j!=i} X_i X_j sigma_ij
The cross terms sigma_ij = rho_ij sigma_i sigma_j are the whole story. A security matters not by its
own variance but by its covariance with the rest of the portfolio. So the right kind of
diversification is across LOW-covariance securities (across industries, not 60 railroads).
Markowitz 1952 contains the first published occurrence of V = sum_j x_j^2 sigma_j^2 + sum_j sum_{k!=j}
x_j x_k rho_jk sigma_j sigma_k in financial economics (Rubinstein). Roy 1952 wrote the same equation
independently.

## E and V are in tension -> the E–V rule and the efficient set
The min-V portfolio is generally not the max-E portfolio. There's a rate at which you trade V for E.
So: don't pick one number to maximize. Look at the set of attainable (E,V) pairs over all feasible X.
The investor only ever wants the *efficient* subset: min V for given E (or more), max E for given V
(or less). Everything else is dominated. This reduces an N-dimensional choice to a 1-parameter curve.

## Geometry (3-security case, from the 1952 paper)
Use sum X_i = 1 to drop X_3 = 1 - X_1 - X_2; work in (X_1,X_2). Attainable set = triangle (the three
X_i >= 0 constraints become three half-planes). 
  - E = E(X_1,X_2) is affine -> isomean curves are parallel straight lines.
  - V = V(X_1,X_2) is a convex quadratic -> isovariance curves are concentric ellipses centered at the
    unconstrained global-min-variance point X-hat.
  - For each isomean line, the min-V point on it is where the line is tangent to an isovariance ellipse.
    As E varies, that tangency point traces a STRAIGHT LINE: the *critical line*.
  - Efficient set = walk from the min-variance point along the critical line (V increasing, E increasing)
    until you hit a boundary of the attainable triangle; then turn and run along that boundary edge to
    the max-E vertex. So the efficient set is a sequence of connected line segments (in X-space) ->
    connected parabola segments when you plot V vs E.
For N securities: same story in higher dimensions. Each subset of "free" (interior) securities defines
a critical line in its own subspace; you start at global min variance and proceed along critical lines,
turning whenever a line meets the boundary of a larger/smaller subspace, until you reach max E.
This is the seed of the critical-line algorithm; the 1952 paper says the analytic N-security technique
"will be presented in the future" — realized in 1956/1959 as the CLA.

## The QP and the parametric / KKT structure (critical-line algorithm)
Two equivalent canonical forms of the frontier optimization:
  (return-targeting)   min_w  w^T Σ w   s.t.  μ^T w = m,  1^T w = 1,  (w >= 0)
  (utility / lambda)   max_w  μ^T w - (δ/2) w^T Σ w  s.t. 1^T w = 1, (w>=0); or min (1/2)w^TΣw - λ μ^T w
Tracing the whole frontier = solving these for ALL targets m (equivalently all λ in [0,∞)).

KKT for min (1/2) w^T Σ w - λ μ^T w  s.t. 1^T w = 1, l <= w <= u:
Lagrangian L = (1/2) w^TΣw - λ μ^T w - γ(1^T w - 1) - sum_i α_i(w_i - l_i) - sum_i β_i(u_i - w_i).
Stationarity: Σ w - λ μ - γ 1 - α + β = 0, with complementary slackness. Partition assets into
FREE set F (l_i < w_i < u_i, so α_i=β_i=0) and BOUND set B (w_i pinned at l_i or u_i).
For the free block: Σ_FF w_F + Σ_FB w_B = λ μ_F + γ 1_F. Solve:
  w_F = Σ_FF^{-1} ( λ μ_F + γ 1_F - Σ_FB w_B ).
Impose 1^T w = 1 (i.e. 1_F^T w_F = 1 - 1_B^T w_B) to pin down γ as an affine function of λ:
  γ = [ (1 - 1_B^T w_B) + 1_F^T Σ_FF^{-1} Σ_FB w_B - λ (1_F^T Σ_FF^{-1} μ_F) ] / (1_F^T Σ_FF^{-1} 1_F).
=> on a fixed (F,B) partition, w_F is AFFINE in λ -> the critical line. As λ decreases from +∞
(max-E corner) toward 0 (global min variance), w moves linearly until some event:
  (a) a free weight hits a bound (l_i or u_i) -> move it to B;  (b) a bound weight's multiplier
  (α_i or β_i) changes sign / would become infeasible -> free it, move to F.
Each event is a "turning point"; between turning points the segment is linear in w and a hyperbola
segment in (σ, m). Computing the λ of each candidate event and taking the binding one gives the next
turning point. This is exactly what PyPortfolioOpt's CLA._solve does:
  _compute_w: w_F = -Σ_FF^{-1}Σ_FB w_B + γ Σ_FF^{-1}1_F + λ Σ_FF^{-1}μ_F  (matches the KKT solve).
  _compute_lambda: c = -c1*c2[i] + c3*c4[i] with c1=1^TΣ^{-1}1, c3=1^TΣ^{-1}μ, c2=Σ^{-1}μ, c4=Σ^{-1}1;
   λ for asset i to hit bound b_i is (c4[i]-c1 b_i)/c (all-free case) — the λ at which free weight i
   reaches a boundary. The algorithm walks turning points, λ from high to 0 (min-variance end).
The unconstrained (no bounds) min-variance and the two scalars 1^TΣ^{-1}1, 1^TΣ^{-1}μ recover the
classic two-fund / hyperbola frontier. global_min_vol = sqrt(1 / (1^T Σ^{-1} 1)) (matches EF code).

## Why VARIANCE (design choices and their why)
- Why E and V specifically (not full distribution)? Only the first two moments are needed for choice
  under the E–V maxim; tractable, you only need μ and Σ (you do NOT need the joint distribution).
  Markowitz notes the third moment M_3 connects to a propensity to gamble: if U=U(E,V) with U_E>0,
  U_V<0 the investor never takes an actuarially fair bet; allowing U=U(E,V,M_3), dU/dM_3 != 0 admits
  some fair bets. E–V is for "investment" not "speculative" behavior.
- Why variance and not σ or coefficient of variation? Choice still lands in the same efficient set
  (σ=√V, σ/E monotone in the relevant region) so it doesn't change the efficient set; variance is the
  quadratic that makes V = w^TΣw and the QP tractable.
- Why no-short (w>=0)? Markowitz's stated frame excludes short sales; with shorts and the pure
  expected-return rule "an infinite amount" goes to the top security (footnote 4). The bound makes the
  feasible set a polytope (simplex) -> the efficient set is piecewise-linear -> CLA. PyPortfolioOpt
  default weight_bounds=(0,1); set (-1,1) to allow shorting.
- Why covariance is the object, not own-variance: from V's cross terms; a security's marginal risk
  contribution is its covariance with the portfolio (∂V/∂X_i = 2 sum_j sigma_ij X_j).
- Why the efficient SET, not one portfolio: Markowitz refuses to assume a utility function (so does
  Roy). Compute the whole frontier; let the investor pick the point (or pick max-Sharpe / a δ).
- max_sharpe variable transform: max (μ-r)^T w / sqrt(w^TΣw) is not convex; substitute y=w/k (k>0),
  fix (μ-r)^T y = 1, minimize y^TΣy, sum y = k -> convex QP; recover w=y/k. (Cornuejols-Tutuncu.)

## Two implementations to land on
1. cvxpy EfficientFrontier (modern, general): min_volatility (min w^TΣw, sum w=1), efficient_return
   (min w^TΣw s.t. μ^T w >= target, sum w=1), efficient_risk (max μ^T w s.t. w^TΣw <= target_var),
   max_quadratic_utility (max μ^T w - (δ/2) w^TΣw), max_sharpe (transformed). Bounds 0<=w<=1.
2. CLA (the native algorithm): _init_algo (start at max-E corner: pour into highest-μ assets up to
   upper bounds until sum=1), then _solve walks turning points (case a bound a free weight, case b
   free a bound weight), λ from high to 0, storing (w, λ, γ, free-set) per turning point; frontier =
   piecewise-linear interpolation between adjacent turning points.

## Sources actually read this run
- Markowitz 1952 "Portfolio Selection", J. Finance 7(1):77-91 — full text (efalken.com clean scan). PRIMARY.
- Lo & Foerster 2021 technical supplement to Markowitz (1952) — covariance/variance algebra, 2-stock numeric example. EXPLAINER.
- Rubinstein 2002 "Markowitz's Portfolio Selection: A 50-Year Retrospective", J. Finance 57(3) — lineage (Williams 1938, Bernoulli LLN, Roy 1952, vN-M, Tobin 1958, 1959 book/CLA). HISTORY.
- Roy 1952 "Safety First and the Holding of Assets", Econometrica 20:431-449 — content via WebSearch summaries (full text paywalled): disaster level d, maximize (μ-d)/σ, same portfolio-variance equation, mean-variance efficient set, independent of Markowitz. ANCESTOR (parallel).
- PyPortfolioOpt (github.com/robertmartin8/PyPortfolioOpt): efficient_frontier.py, objective_functions.py, _base_optimizer.py, cla.py — CANONICAL CODE.

## Unsourced / gaps
- Roy 1952 read only via reputable secondary summaries (Wikipedia/Rubinstein) — full text behind JSTOR/Econometrica paywall; its formula (μ-d)/σ and efficient-set claim are corroborated across Rubinstein + multiple secondaries, so used as ancestor context only.
- Niedermayer CLA chapter was a Springer landing wall (not readable); the KKT/critical-line derivation here is derived analytically and cross-checked line-by-line against PyPortfolioOpt cla.py (_compute_w, _compute_lambda) — self-consistent, no unsourced numeric claims.
