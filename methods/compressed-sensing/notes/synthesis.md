# Synthesis — Sparse recovery via ℓ1 + proximal-gradient (ISTA/FISTA)

## The pain point / research question
Underdetermined linear system y = Ax, A ∈ R^{m×n}, m << n. Infinitely many solutions. But the
signal of interest is *sparse* (few nonzeros) or sparse in a known basis. Want to recover the
unique sparse x. Naively this is "minimize number of nonzeros (ℓ0) subject to Ax=y" — combinatorial.

## Key facts (all sourced)
- **ℓ0 minimization NP-hard**: Natarajan 1995, "Sparse Approximate Solutions to Linear Systems",
  SIAM J. Comput. 24(2):227-234. min ‖x‖0 s.t. ‖Ax-b‖≤ε is NP-hard. → CONTEXT.
- **CRT 2006** (arXiv:math/0409186, "Robust Uncertainty Principles"): (P0) min‖g‖_{ℓ0} s.t. ĝ|Ω=f̂|Ω
  is combinatorial (exponentially many supports, ~4^N·3^{-3N/4} for |Ω|~N/2). Replace by convex
  (P1) min‖g‖_{ℓ1} s.t. ĝ|Ω=f̂|Ω. KEY RESULT: solutions of (P0) and (P1) coincide for overwhelming
  fraction of T,Ω with |T| ≤ α·|Ω|/log N. → exact recovery by ℓ1 (LP).
- **Donoho 2006** "Compressed Sensing" IEEE TIT 52(4):1289-1306: n = O(m^{1/4} log^{5/2} m) nonadaptive
  measurements suffice; reconstruct compressible signal via ℓ1-type nonlinear procedure.
- **Tibshirani 1996 LASSO** (read via 2011 retrospective, JRSS-B 73(3):273-282): min ‖y-Xβ‖² s.t.
  ‖β‖_1 ≤ t, equivalently penalized min ½‖y-Xβ‖²+λ‖β‖_1. ℓ1 penalty does selection (sets coeffs to
  exactly 0) — geometry: ℓ1 ball has corners on axes.
- **Basis pursuit**: Chen, Donoho, Saunders 1998 "Atomic decomposition by basis pursuit",
  SIAM J. Sci. Comput. 20:33-61. min ‖x‖_1 s.t. Ax=b.

## The objective and convention note (IMPORTANT)
The FISTA paper writes F(x) = ‖Ax-b‖² + λ‖x‖_1 (NO 1/2 factor on the LS term), with f(x)=‖Ax-b‖²,
g(x)=λ‖x‖_1, and Lipschitz constant L(f) = 2λ_max(AᵀA). The prompt and standard LASSO use
½‖Ax-y‖²+λ‖x‖_1 with L=λ_max(AᵀA). I will use the **standard ½ convention** (matches pyproximal /
Vandenberghe / LASSO) in the deliverables, since that is what the canonical code implements, and
note the gradient ∇f = Aᵀ(Ax-y), L = λ_max(AᵀA) = ‖A‖². Both are internally consistent; I must keep
factors consistent throughout. (The paper's L=2λmax is correct for *its* no-½ objective.)

## Derivation chain (the heart of reasoning.md)
1. y=Ax underdetermined → sparsity prior → min ‖x‖0 s.t. Ax=y.
2. ℓ0 NP-hard (Natarajan), combinatorial (CRT). Need a tractable surrogate.
3. Convex relaxation: ℓ1 is the tightest convex surrogate of ℓ0 on the unit ball (convex envelope of
   ‖·‖0 on [-1,1]^n is ‖·‖1). ℓ1 ball has vertices on axes → solution lands at sparse points. (LASSO
   geometry.) With noise, penalized form: min_x ½‖Ax-y‖² + λ‖x‖_1.  [convex but nonsmooth]
4. Why not interior-point / SOCP? Works (Chen et al.) but A is large-scale & dense; second-order
   methods choke. Need first-order, matrix-vector-product-only (Aᵀ(Ax-y)) methods. (FISTA intro.)
5. Smooth part f=½‖Ax-y‖² has Lipschitz gradient (L=‖A‖²); nonsmooth part g=λ‖·‖_1 is simple/separable.
   SPLIT. Gradient descent only works on smooth f. Subgradient on the whole thing → O(1/√k), slow,
   and doesn't give exact zeros nicely. Better: handle f by linearization+quadratic model, g exactly.
6. Proximal-gradient = majorization-minimization: at y, upper-bound f by its quadratic model
   Q_L(x,y)=f(y)+⟨x-y,∇f(y)⟩+(L/2)‖x-y‖² (valid for L≥L(f), descent lemma / Lemma 2.1). Minimize
   Q_L(x,y)+g(x). Complete the square → x = argmin_x g(x)+(L/2)‖x-(y-(1/L)∇f(y))‖²
   = prox_{(1/L)g}(y - (1/L)∇f(y)). The map p_L(y).
7. prox of λ‖·‖_1 (step τ=1/L): separable, per-coordinate min_u λ|u| + (1/2τ)(u-v)². Subdifferential:
   0 ∈ λ∂|u| + (u-v)/τ. Three cases → SOFT-THRESHOLD:
   soft(v,τλ)_i = sign(v_i)·max(|v_i|-τλ, 0). DERIVE the three cases. This is where exact zeros come from.
8. ISTA: x_{k+1} = soft(x_k - τ Aᵀ(Ax_k - y), τλ), τ=1/L. Rate O(1/k) (Theorem 3.1; derive via
   Lemma 2.3 telescoping). Function values monotone nonincreasing.
9. Wall: O(1/k) is slow (ISTA "arbitrarily bad"). Gradient descent on smooth convex is also O(1/k),
   but Nesterov 1983 has an *optimal* O(1/k²) method using ONE gradient eval + a cheap extrapolated
   point. Lower bound (Nemirovsky-Yudin) says O(1/k²) is best possible for first-order on smooth convex.
10. FISTA: apply prox at an extrapolated point y_k (not x_{k-1}). 
    x_k = p_L(y_k); t_{k+1}=(1+√(1+4t_k²))/2, t_1=1; y_{k+1}=x_k + ((t_k-1)/t_{k+1})(x_k - x_{k-1}).
    Same per-iteration cost (one prox = one soft-threshold + one A,Aᵀ). Rate O(1/k²) (Theorem 4.4).
11. Why the specific t_k recursion? From Lemma 4.1: need t_k² = t_{k+1}² - t_{k+1} (so the telescoping
    of (2/L)(t_k² v_k - t_{k+1}² v_{k+1}) ≥ ‖u_{k+1}‖²-‖u_k‖² closes via Lemma 4.2). Solve
    t_{k+1}²-t_{k+1}-t_k²=0 → t_{k+1}=(1+√(1+4t_k²))/2. Then t_k≥(k+1)/2 (Lemma 4.3) → 
    v_k = F(x_k)-F* ≤ 2L‖x0-x*‖²/(k+1)². DERIVE Lemma 4.1, 4.2, 4.3 and Theorem 4.4.
    The extrapolation weight ω_k=(t_k-1)/t_{k+1} → k/(k+3) asymptotically (Vandenberghe form).

## Lemma 2.3 (the central inequality — derive fully)
If F(p_L(y)) ≤ Q_L(p_L(y),y) (holds for L≥L(f)), then ∀x:
  F(x) - F(p_L(y)) ≥ (L/2)‖p_L(y)-y‖² + L⟨y-x, p_L(y)-y⟩.
Proof: convexity of f,g gives lower linear bounds; subtract Q; use Lemma 2.2 optimality
∇f(y)+L(z-y)+γ(y)=0 with γ∈∂g(z). Worked in reasoning.

## Code (grounded in pyproximal / Vandenberghe)
- soft-threshold: np.maximum(|x|-thresh,0)*sign(x)
- ISTA loop: x = soft(x - tau*A.T@(A@x - y), tau*lam), tau=1/L
- FISTA loop: prox at extrapolated y; t-recursion; momentum (t_old-1)/t_new.
- L = power-iteration estimate of ‖A‖² (λmax(AᵀA)), or backtracking.
pyproximal ProximalGradient: x_{k+1}=y_k+η(prox_{τεg}(y_k-τ∇f(y_k))-y_k); y_{k+1}=x_k+ω(x_k-x_{k-1});
acceleration="fista": ω=(t_{k-1}-1)/t_k, t_k=(1+√(1+4t_{k-1}²))/2.

## Design-decision → why
- ℓ1 not ℓ2 penalty: ℓ2 (Tikhonov/ridge) shrinks but never zeroes → not sparse; ℓ1 ball corners → exact zeros. 
- ℓ1 not ℓ0: ℓ0 NP-hard/nonconvex; ℓ1 convex envelope, tractable, provably recovers (CRT/Donoho RIP).
- penalized (λ) vs constrained: equivalent by Lagrange; penalized convenient for prox-gradient & noise.
- prox-gradient not subgradient: subgradient O(1/√k) and no exact sparsity per step; prox handles
  nonsmooth g exactly, keeps smooth f cheap (matrix-vector only).
- prox-gradient not interior-point: IPM second-order, dense large-scale infeasible; first-order scales.
- step τ=1/L: largest step for which quadratic majorizer dominates f (descent guaranteed). Backtracking
  if L unknown.
- FISTA momentum vs heavy-ball: Nesterov extrapolation provably O(1/k²) optimal; plain momentum not.
- t_k recursion specific form: forced by the telescoping identity t_k²=t_{k+1}²-t_{k+1}.
