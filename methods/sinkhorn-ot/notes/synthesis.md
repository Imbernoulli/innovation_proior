# Synthesis — Entropic-regularized OT / Sinkhorn

## Pain point (research question)
Optimal transportation (EMD/Wasserstein) distances between two histograms r,c in the simplex Σ_d
require solving an LP: min_{P∈U(r,c)} ⟨P,M⟩, where U(r,c) = {P≥0 : P1=r, P^T1=c} is the
transportation polytope. Exact solvers (network simplex / interior point / Pele-Werman FastEMD,
Rubner EMD) cost ≥ O(d^3 log d) in the general ground-metric case, super-cubic in practice. A
single pair of dimension-few-hundred histograms can take seconds. The LP optimum sits on a vertex
of U(r,c): a sparse table with ≤ 2d−1 nonzeros (Brualdi §8.1.3) — a near-deterministic, "extreme"
coupling. Two problems: (1) cost, (2) the optimum is a vertex so the objective is piecewise-linear,
non-differentiable in r,c,M, and the vertex solution is brittle/non-robust.

## Background facts (sourced, pre-method)
- Σ_d = {x∈R^d_+ : x^T 1 = 1}. U(r,c) = transportation polytope = all joint distributions of (X,Y)
  with marginals r,c (contingency tables). The independence table rc^T ∈ U(r,c).
- Entropy h(P) = −Σ p_ij log p_ij; KL(P||Q)=Σ p_ij log(p_ij/q_ij).
- Basic info-theoretic inequality (Cover&Thomas §2): ∀P∈U(r,c), h(P) ≤ h(r)+h(c), tight at rc^T.
  And KL(P||rc^T) = h(r)+h(c)−h(P) = mutual information I(X;Y).
- LP vertex fact: optimum on vertex, ≤2d−1 nonzeros (Brualdi). Worst-case O(d^3 log d) (Pele-Werman §2.1).
- d_M is a metric when M ∈ cone of distance matrices (m_ii=0, triangle ineq) — Villani §6.1.
- Max-entropy principle (Jaynes 1957; Dudik-Schapire 2006): for a given cost level, prefer the
  smoothest (max-entropy) coupling.
- Sinkhorn-Knopp 1967 (Pacific J Math 21:343-348): a nonnegative matrix A with total support has a
  UNIQUE doubly-stochastic-scaling D1 A D2; alternately normalizing rows/cols converges to it.
  General prescribed-marginal version: unique D1 A D2 with row sums r, col sums c. Older names:
  IPFP (Deming-Stephan 1940), RAS (Bacharach 1965), gravity models in transport economics
  (Erlander-Stewart 1990), softassign (Kosowsky-Yuille 1994).
- Gluing lemma (Villani Lemma 7.6) is what makes d_M satisfy triangle inequality.

## The derivation (KKT)
Penalized problem: P^λ = argmin_{P∈U(r,c)} ⟨P,M⟩ − (1/λ) h(P).  (ε = 1/λ.)
−(1/λ)h(P) = (1/λ)Σ p log p is strictly convex ⇒ unique minimizer (vs LP's flat faces).
Lagrangian with multipliers α,β for the two marginal equalities:
  L = Σ_ij (1/λ) p_ij log p_ij + p_ij m_ij + α^T(P1−r) + β^T(P^T1−c).
∂L/∂p_ij = (1/λ)(log p_ij + 1) + m_ij + α_i + β_j = 0
 ⇒ log p_ij = −1 − λ m_ij − λα_i − λβ_j
 ⇒ p_ij = e^{−1/2 −λα_i} · e^{−λ m_ij} · e^{−1/2 −λβ_j} = u_i K_ij v_j, K = e^{−λM}.
So P^λ = diag(u) K diag(v). By Sinkhorn 1967 this is the UNIQUE element of U(r,c) of that form.
Marginals impose: diag(u)K diag(v) 1 = r ⇒ u ⊙ (Kv) = r ⇒ u = r/(Kv);
 and v ⊙ (K^T u) = c ⇒ v = c/(K^T u). Alternate ⇒ Sinkhorn iteration (matrix scaling).
Cost per iter O(d^2) (or O(d^2 N) for N targets at once, vectorized, GPU-friendly).

## Convergence (explainer, in-frame derivable)
- View as Bregman/KL iterative projections onto affine sets C1={P1=r}, C2={P^T1=c}; U=C1∩C2.
  KL-projection onto an affine set, alternating, converges (Bregman 1967). Sinkhorn iterates ARE
  these alternating KL projections.
- Linear (geometric) convergence: the Sinkhorn map is a contraction in Hilbert's projective metric
  (Birkhoff / Franklin-Lorenz 1989; nonlinear Perron-Frobenius). Rate = κ(K) contraction ratio < 1.
- ε→0 (λ→∞): K becomes diagonally dominant / near-singular, contraction ratio → 1, more iterations,
  and entries of K underflow to 0 (numerical breakdown). ε→∞ (λ→0): P→rc^T (independence).

## Metric properties (paper's own, used in reasoning)
- α large ⇒ U_α(r,c)=U(r,c) ⇒ Sinkhorn distance = d_M (since h(P) ≥ ½(h(r)+h(c))).
- α=0 ⇒ U_0={rc^T} ⇒ d_{M,0}=r^T M c (independence kernel); n.d. kernel if M Euclidean dist matrix.
- Triangle inequality via gluing lemma with entropic constraint: S_ik = Σ_j p_ij q_jk / y_j; need
  S ∈ U_α(x,z): row/col sums check out, and data-processing inequality I(X;Y)≥I(X;Z) gives h(S)
  sufficient. Then chain: d(x,z) ≤ ⟨S,M⟩ ≤ Σ(m_ij+m_jk) p q / y = d(x,y)+d(y,z).
- The hard-constrained d_{M,α} and the penalized d_M^λ are linked by duality: each α ↔ some λ∈[0,∞].
  In practice use the penalized (dual-Sinkhorn) form; recover d_{M,α} by bisection on λ
  (entropy of P^λ decreases monotonically in λ).

## Code (grounded in POT ot.bregman.sinkhorn_knopp)
- K = exp(−M/reg)  (reg = ε = 1/λ)
- init v=1; loop: u = a/(Kv); v = b/(K^T u); check ||(diag(u)K diag(v))^T 1 − b|| < thr.
- return diag(u) K diag(v) = u[:,None]*K*v[None,:]; loss = Σ u (K⊙M) v.
- Cuturi's Algorithm 1 folds it onto x=1/u: x = diag(1/r) K (c ⊙ 1/(K^T (1/x))). Same fixed point.
- Numerical: log-domain (sinkhorn_log) for small ε to avoid underflow.

## Design decisions → why
- entropy penalty (not L2 / graph-norm like Ferradans 2013): gives the closed multiplicative form
  diag(u)Kdiag(v) ⇒ matrix scaling ⇒ cheap+parallel; also strictly convex ⇒ unique+differentiable;
  also a *distance* (triangle ineq survives). L2 wouldn't factorize into scalings.
- KL ball / mutual-information constraint: max-entropy = "most plausible coupling at given cost",
  robust vs the brittle ≤2d−1-nonzero LP vertex.
- sign: minus entropy (penalize low entropy) ⇒ +(1/λ)Σ p log p in objective ⇒ strictly convex.
- 1/λ vs ε: λ→∞ sharper/closer to EMD but slower & underflow; small λ recommended.
- fixed iteration count in practice: checking ||Δx|| each step is costly on GPU.
- the bistochastic→general-marginal scaling needs K>0 (total support), guaranteed by K=e^{−λM}>0.
