# Synthesis — Erdős Minimum Overlap Problem (principled method)

## The problem (discrete)
Partition [2n]={1,...,2n} into A,B with |A|=|B|=n. For integer k, M_k = #{(a,b)∈A×B : a-b=k}.
M(n) = min_{A,B} max_k M_k. Estimate M(n). Erdős 1955.
- Trivial averaging: Σ_k M_k = n^2 over 4n-1 values of k → average > n/4, so M(n) > n/4.
- Trivial upper bound: A=[n/2,3n/2] gives M(n) ≤ n/2.
- Haugland 1996: limit μ = lim M(n)/n exists. "Minimum overlap constant."
- Best known: 0.379005 ≤ μ ≤ 0.3809268534330870 (lower=White 2022, upper=Haugland 2016).

## The continuous reformulation (THE methodological leap #1 — Swinnerton-Dyer)
Moser & Murdeshwar introduced a function analogue. Swinnerton-Dyer proved (in Haugland 1996, JNT 58:71-78) that μ equals the continuous variational quantity. Two equivalent statements:
- (Haugland/SwD form) μ = inf over step functions f on [0,2], values in [0,1], ∫_0^2 f = 1, of max_k ∫ f(x)(1-f(x+k)) dx.
- (White's working form) For f:[-1,1]→[0,1] with ∫_{-1}^1 f = 1, g=1-f, define M(x)=∫_{-1}^1 f(t)g(x+t)dt on [-2,2]. Then μ = inf_f ||M||_∞.
The leap: replace combinatorial search over 2^{2n} partitions by an analysis problem over density functions. A is "smeared" into a density f(x) on the interval; M(x) is the autocorrelation-type overlap of f with the translated complement g=1-f.

## Properties of M(x) (constraints)
- Mass: ∫_{-2}^2 M(x)dx = 1 (so average M ≥ 1/4 → μ ≥ 1/4). [Discrete analogue of Erdős averaging.]
- Variance/second moment (Moser-Murdeshwar): ∫ x^2 M = 2/3 + (1/2)E(M)^2, where E(M)=∫xM. Equivalently variance ≤ 2/3. Concentrating mass at mean → μ ≥ 1/√8 ≈ 0.354. Moser's refinement gives √(4-√15)≈0.35639.
- KEY NEW (White): even cosine Fourier coefficients of M are ≤ 0:
  A_{2m} = (4 sin(mπ)/...) ... = -2(a_{2m}^2 + b_{2m}^2) ≤ 0. In general
  A_m = (4 sin(mπ/2)/(mπ)) a_m - 2(a_m^2+b_m^2), and A_{2m} ≤ 0 since sin(mπ)=0.
  B_m = -(4/(mπ)) sin(mπ/2) b_m, and B_{2m}=0.
  This is because M = 4 \bar{f̂} ĝ = 4\bar{f̂}(1̂_{[-1,1]} - f̂), and on even k the indicator term vanishes leaving -4|f̂|^2 ≤ 0.

## Fourier lemmas (White §2)
- Lemma (expfou): M̂(k) = (4/(kπ)) sin(kπ/2) \bar{f̂(k)} - 4|f̂(k)|^2, for k≠0.
- Relate sine-cosine coeffs of f on [-1,1] (c_k,d_k) to coeffs of f,M on [-2,2] (a_m,b_m,A_m,B_m). For odd m these are infinite sums over c_k,d_k; for even m, a_m = c_{m/2}/2 etc.
- Tail bounds (Parseval + Cauchy-Schwarz): Σ_{k>T} truncation error bounded; Σ(c_k^2+d_k^2) ≤ 1/2.

## The LP / convex program (THE methodological leap #2 — White)
Discretize: N intervals of width L=2/N. Variables w_j = average of M on [(j-1)L, jL], v_j = average on negative side.
Objective: minimize Ω where 0 ≤ w_j,v_j ≤ Ω (Ω is a proxy for ||M||_∞). Any true M gives a feasible point with Ω = ||M||_∞, so the program's optimum Ω* ≤ ||M||_∞ — i.e. Ω* is a valid LOWER bound on μ.
Constraints (all linear or convex-quadratic in the variables):
- mass: L Σ(w_j+v_j) = 1
- mean bounds: h_1 ≤ L^2 Σ(j w_j - (j-1)v_j), and the upper version ≤ h_2.
- second moment: L^3 Σ (j-1)^2 (w_j+v_j) ≤ 2/3 + h_2^2/2.
- cosine constraints: (L/2) Σ α^-_{j,m}(w_j+v_j) ≤ A_m = (4 sin(mπ/2)/(mπ)) a_m - 2(a_m^2+b_m^2).  [quadratic in a_m,b_m]
- sine constraints similarly with β arrays.
- Fourier-coefficient variables c_k,d_k (k≤T), tail-slack ε,δ, with Parseval Σ(c_k^2+d_k^2) ≤ 1/2.
α^±_{j,m}, β^±_{j,m} = piecewise bounds of cos/sin on each small interval (Lipschitz: ±πmL/4).

Simplified LP (assume M even, drop sine/quadratic): variables Ω,w_j; constraints 0≤w_j≤Ω, Σw_j=N/4, Σ α^-_{j,2m} w_j ≤ 0 (cosine), L^3 Σ(j-1)^2 w_j ≤ 1/3 (moment). With N=80000,R=20: proves μ ≥ 0.375 for even M.

Full convex program (no evenness assumption, with sine + quadratic Am): proves μ ≥ 0.379005 (Theorem). N,T,R up to 25000,7000,10.

## Certification via DUALITY (the rigor step)
Numerical solvers (large N) are fast but inexact. To make the bound RIGOROUS:
- Reformulate primal as a Second-Order Cone Program (SOCP) — quadratic constraints become ||...||_2 ≤ linear.
- Write the SOCP dual; eliminate equality constraints by solving for one y-variable per row (each Φ_j has a 'c' vector with a unique nonzero entry).
- Any FEASIBLE dual point gives a lower bound on the primal optimum (weak duality) → a rigorous lower bound on μ.
- Check each dual inequality is satisfied with a margin exceeding worst-case IEEE-754 double rounding error → certified.
- Divide-and-conquer: split valid ranges of (E(M), c_1, d_1) into chunks; min over chunks. Since h,p only enter the objective (not dual constraints), reuse a dual point and recompute objective over a region → covering "ellipses" in (h,p) plane. 7 ellipses cover the last region → 0.379005.

## Upper bound side (Haugland, the constructive complement)
Pick a step function f with n steps (constant on (2i/n, 2(i+1)/n)), symmetric f(x)=f(2-x). Evaluate max_k ∫ f(1-f(·+k)) on the grid (the convolution becomes a finite sum; max over shifts k=multiples of 2/n). Optimize the step heights numerically (local/nonlinear optimization) to minimize the max overlap. 15 steps → 0.38153; 19 → 0.381112; 51 → 0.3809268534330870.
The integral ∫ f(x)(1-f(x+k)) dx for piecewise-constant f on a uniform grid is a discrete (cross-correlation) sum of the step heights — that is what makes it cheaply evaluable and optimizable.

## Anti-pattern (context only, NOT the method)
AlphaEvolve (2025) → 0.380924; TTT-Discover (2026) → 0.380876; SimpleTES → 0.380868. These are evolutionary/LLM-driven NUMERIC SEARCHES over the SAME step-function objective — better records, not a new principled method. The methodological content is entirely in (a) Swinnerton-Dyer's reduction and (b) White's Fourier→convex-program→dual-certificate pipeline.

## Design decisions → why
- Why density functions not partitions? Swinnerton-Dyer equivalence makes lim a clean analytic infimum; removes integrality.
- Why Fourier? The overlap M is a (cross-)correlation f * g̃; in Fourier domain it factorizes M̂ = 4\bar{f̂}ĝ, exposing the sign structure (even coeffs ≤0). Generates infinitely many linear inequalities ||M||_∞ must respect.
- Why average values w_j as variables (not pointwise)? An LP needs finitely many variables; averaging over small intervals keeps constraints exact inequalities (via interval bounds α^±) rather than approximations, so the bound stays rigorous.
- Why the dual? Primal optimum needs the true minimizer; a dual feasible point alone certifies a lower bound — and a dual feasible point can be checked exactly against rounding.
- Why SOCP? Standard way to dualize convex-quadratic constraints with a clean, writable dual (Lobo-Vandenberghe-Boyd-Lebret).
- Why divide & conquer on (E(M),c_1,d_1)? The full-range program collapses to ~0.25; fixing these low-order moments tightens A_2 = -(c_1^2+d_1^2)/2 and the moment bounds, lifting the bound. Reusing dual points across (h,p) via the objective-only dependence avoids thousands of solves.
- Why margin check vs rounding unit u=2^{-53}? Converts a numerical solution into a theorem.

## Code
White's program: build constraint matrices for the SOCP, solve with a numerical SOCP/QCQP solver (he used CPLEX), extract dual point, verify margins, sweep (h,p) ellipses. No public repo; reconstruct faithfully from the paper's SOCP/dual tables (cvxpy/ECOS or scipy.linprog for the simplified LP).
Haugland's upper bound: parametrize step heights, define overlap(k) as a finite correlation sum, minimize the max over k with a nonlinear optimizer (scipy.optimize / Nelder-Mead / SLSQP).
