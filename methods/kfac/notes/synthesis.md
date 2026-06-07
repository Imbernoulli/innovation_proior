# K-FAC synthesis notes (Phase 1.5)

## Pain point / research question
SGD(+momentum) is the workhorse for NN training. Curvature-aware methods (HF, Hessian-free)
make MUCH more progress per update (~10² updates vs 10⁴–10⁵ for diagonal methods), because NN
curvature is highly *non-diagonal* and updates that respect it are powerful. But HF is expensive:
each update runs CG for many iterations, each iteration = one curvature-matrix-vector product (≈ cost
of a stochastic gradient), and the curvature estimate must be *frozen* while CG iterates → HF sees
little data per update → bad in the stochastic regime. Diagonal/low-rank preconditioners are cheap
and stochastic-friendly but only give limited speedups (they throw away the off-diagonal curvature
that makes 2nd-order methods powerful).

Goal: a method whose updates are big and powerful like HF's (full, non-diagonal, non-low-rank
curvature) yet (a) directly invertible without CG, and (b) summarizable in a compact data structure
whose size is independent of the amount of data used to estimate it (so it can use an online
exponentially-decayed average over many minibatches, like diagonal methods do).

## Load-bearing ancestors (the lineage)
- **Amari, natural gradient (1998), info geometry (Amari–Nagaoka 2000).** Steepest descent in
  KL-metric: direction is F⁻¹∇h, F = E[∇log p ∇log pᵀ] = E[∇θ ∇θᵀ] the Fisher. Invariant to
  reparameterization. Limitation: F is (#params)² — millions² — totally intractable to form/invert.
- **Gauss-Newton / Fisher equivalence (Schraudolph 2002; Martens 2014 "new insights"; Martens &
  Sutskever 2012; Pascanu & Bengio 2014).** When the loss is −log r(y|z) for an exponential-family
  R_{y|z} with z = natural params, the Fisher = the Generalized Gauss-Newton matrix = a PSD
  approximation of the Hessian. So natural-gradient methods ARE 2nd-order methods; brings in the
  whole damping/trust-region toolbox.
- **Hessian-free (Martens 2010; Martens & Sutskever 2012; Vinyals & Povey KSD 2012).** The thing to
  beat. Uses exact F implicitly via mat-vec products + CG. The two drawbacks above motivate K-FAC.
- **Block-diagonal / per-unit approximations: TONGA (Le Roux et al. 2008), Ollivier (2013).** Blocks
  = weights of one *unit*. K-FAC's blocks are MUCH bigger: one whole *layer*. Gap: per-unit blocks
  ignore within-layer cross-unit structure; many small inverses.
- **Heskes (2000): closest prior art.** Already proposed a Kronecker-factored *block-diagonal* Fisher
  approximation for MLPs and used (A⊗B)⁻¹=A⁻¹⊗B⁻¹ for an approximate natural-gradient method. K-FAC's
  deltas vs Heskes (all crucial in practice): adaptive γ (Heskes fixes γ by hand) + exact-F rescaling
  (Heskes has none; basic factored Tikhonov alone can't make good updates — Fig damping_rescaling),
  stochastic MC estimate of G (Heskes computes G exactly → doesn't scale to many outputs), the
  block-tridiagonal inverse, parameter-free momentum, online estimation.
- **FANG (Grosse & Salakhutdinov 2015).** Approximates the gradient distribution with a directed
  graphical model — basis for the block-tridiagonal "inverse is tree-structured" justification.
- **Pourahmadi (1999, 2011).** Inverse-covariance ↔ linear-regression: row i of Σ⁻¹ = optimal linear
  predictor coefficients of var i from the rest (up to scale). Σ⁻¹ = D⁻¹(I−B). Justifies why F⁻¹ is
  ≈ block-diagonal/tridiagonal even though F is dense.
- **Povey et al. 2015 (concurrent).** Similar Kronecker block-diag, but empirical Fisher + basic
  Tikhonov + online low-rank+diag factor estimate.
- **Damping toolbox: Levenberg–Marquardt (Moré 1978), trust regions (Nocedal & Wright 2006).**
- **Momentum (Polyak 1964; Plaut/Nowlan/Hinton 1986; Sutskever et al. 2013); Matrix Momentum
  (Scarpetta et al. 1999).** K-FAC's momentum solves α,μ to jointly minimize the exact-F quadratic.
- **Stein/Sylvester solvers (Chu 1987; Gardiner et al. 1992; Smith 1968; Simoncini 2014).** For
  inverting A⊗B ± C⊗D (factored Tikhonov on tridiag, conditional covariances).
- **Newton-Schulz matrix inversion (Pan & Schreiber 1991).** Optional cheap inverse.

## Core derivation chain (verified inline)
Notation: layer i: s_i = W_i ā_{i-1}, a_i = φ_i(s_i), ā = [a;1] (homogeneous → bias is last column).
Backprop: g_i = Ds_i, DW_i = g_i ā_{i-1}ᵀ, Da_{i-1} = W_iᵀ g_i. θ = stacked vec(W_i).

1. **vec rule.** Column-stacking vec: vec(uvᵀ) = v⊗u, and vec(BXAᵀ) = (A⊗B)vec(X).
   ⇒ vec(DW_i) = vec(g_i ā_{i-1}ᵀ) = ā_{i-1}⊗g_i.
2. **Exact Fisher block.** F_{i,j} = E[vec(DW_i)vec(DW_j)ᵀ] = E[(ā_{i-1}⊗g_i)(ā_{j-1}⊗g_j)ᵀ].
   Using (A⊗B)(C⊗D) = AC⊗BD and (P⊗Q)ᵀ = Pᵀ⊗Qᵀ:
   = E[ā_{i-1}ā_{j-1}ᵀ ⊗ g_i g_jᵀ]. (CHECK: ✓)
3. **THE Kronecker approximation.** E[X⊗Y] ≈ E[X]⊗E[Y] (expectation of Kron ≈ Kron of expectations):
   F_{i,j} ≈ Ā_{i-1,j-1}⊗G_{i,j} =: F̃_{i,j}, where Ā_{i,j}=E[ā_iā_jᵀ], G_{i,j}=E[g_ig_jᵀ].
   This is NOT exact and won't become exact asymptotically. Justified two ways:
   (a) equiv. to assuming products ā^(1)ā^(2) of activities ⟂ products g^(1)g^(2) of pre-act grads;
   (b) (Appendix A) the scalar error E[āā gg] − E[āā]E[gg] equals a sum of cumulants
       κ(ā¹,ā²,g¹,g²) + E[ā¹]κ(ā²,g¹,g²) + E[ā²]κ(ā¹,g¹,g²) — all order-≥3 cumulants, which vanish if
       (ā,g) jointly Gaussian. Uses **Lemma (forward⟂backward):** E[u·Dv]=0 when u⟂y | f(x,θ) and
       expectation is under the MODEL's P_{y|x} (score has zero mean). This kills 10 of the 15
       moment→cumulant terms; the surviving κ(ā¹,ā²)κ(g¹,g²)+κ(ā¹)κ(ā²)κ(g¹,g²) reassemble into
       E[ā¹ā²]E[g¹g²]. (CHECK: ✓ rederived; the y must be SAMPLED FROM THE MODEL, not the data —
       else it's the empirical Fisher, lemma breaks, GGN-equiv lost.)
4. **Why sample y from the model.** G_{i,j}=E[g_ig_jᵀ] must average over P_{y|x} (the model), via MC:
   sample ŷ~model, rerun the backward pass with ŷ as target. Ā doesn't depend on y (forward only).
5. **Cheap inverse.** F̃ is a Khatri–Rao (block matrix of Kron products) — no general efficient
   inverse. So approximate F̃⁻¹ as block-DIAGONAL or block-TRIDIAGONAL.
   - **Block-diag** F̆ = diag(Ā_{i-1,i-1}⊗G_{i,i}). (A⊗B)⁻¹=A⁻¹⊗B⁻¹ ⇒ F̆⁻¹=diag(Ā⁻¹⊗G⁻¹). Solve:
     u=F̆⁻¹v ⇒ via (A⊗B)vec(X)=vec(BXAᵀ): **U_i = G_{i,i}⁻¹ V_i Ā_{i-1,i-1}⁻¹** (Ā,G symmetric so no
     transpose visible). (CHECK ✓ — matches reference code: v = Q_g (v1/(d_g⊗d_a+λ)) Q_aᵀ.)
   - **Block-tridiag** F̂: defined so F̂⁻¹ is block-tridiagonal and F̂ matches F̃ on tridiagonal blocks.
     Equiv. to a tree-structured UGGM over Dθ ⇔ a linear-Gaussian DGGM (edges high→low layers).
     Ψ_{i,i+1}=F̃_{i,i+1}F̃_{i+1,i+1}⁻¹ = (Ā_{i-1,i}Ā_{i,i}⁻¹)⊗(G_{i,i+1}G_{i+1,i+1}⁻¹)=Ψ^Ā⊗Ψ^G.
     Σ_{i|i+1}=F̃_{i,i} − Ψ F̃_{i+1,i+1} Ψᵀ (difference of Kron products). Block-Cholesky:
     F̂⁻¹ = Ξᵀ Λ Ξ, Λ=diag(Σ_{i|i+1}⁻¹,…,Σ_ℓ⁻¹), Ξ = bidiagonal with −Ψ_{i,i+1} on super-diag.
     Mat-vecs with Ξ, Ξᵀ via the vec rule; mat-vec with Λ needs inverting a difference-of-Kron
     (Appendix B Stein solver). (CHECK ✓.)
6. **Why F⁻¹ is ≈ block-tridiag though F isn't (Pourahmadi).** Predicting an entry of DW_i from all
   of Dθ: most useful predictors are the other entries of DW_i (⇒ block-diag dominates), then DW_{i±1}
   (info only flows from adjacent layers in fwd/bwd) ⇒ tridiag is the mild, principled relaxation.
   Empirically (paper figs) F⁻¹ is ~block-tridiag, F itself is not.

## Damping (non-optional for any powerful 2nd-order method)
- Quadratic model M(δ)=½δᵀFδ+∇hᵀδ+h; minimizer = −F⁻¹∇h. With ℓ2 (η/2‖θ‖²): use F+ηI.
- Exact-F Tikhonov adds (λ+η)I → trust region. But for the *approximate* F, no good single λ exists
  (must stay large to mask the Fisher-approx error → washes out small eigenvalues / low-curv dirs).
- **Factored Tikhonov.** Adding (λ+η)I = (λ+η)I⊗I to Ā⊗G breaks the single-Kron structure. Instead
  add to each factor: (Ā + π√(λ+η) I) ⊗ (G + (1/π)√(λ+η) I). Expanding gives Ā⊗G + (λ+η)I⊗I plus
  residual π√·I⊗G + (1/π)√·Ā⊗I. Choosing π to minimize a norm bound on the residual:
  **π_i = sqrt( [tr(Ā)/(d_{i-1}+1)] / [tr(G)/d_i] )** (ratio of avg eigenvalues). π balances the
  damping between the two factors. Often works BETTER than exact (λ+η)I in practice.
- **Exact-F rescaling.** Raw Δ = approx-F⁻¹(−∇h) is a poor update. Rescale δ=α*Δ with α from the
  exact-F quadratic: α* = −∇hᵀΔ / (ΔᵀFΔ + (λ+η)‖Δ‖²). Only ONE exact-F mat-vec (cheap via Jv trick,
  App C). Makes K-FAC = HF with approx-F preconditioner + 1 CG step from 0.
- **Adapt λ (Levenberg–Marquardt):** ρ=(h(θ+δ)−h(θ))/(M(δ)−M(0)); ρ>3/4→λ←ω₁λ; ρ<1/4→λ←λ/ω₁.
- **Separate γ for the factored Tikhonov** (init √(λ+η)), adjusted by greedy 3-point search on M(δ).
  γ's job: best Δ; λ's job: trust region. Decoupling them matters for robustness.

## Momentum / estimation / efficiency / invariance
- **Momentum:** δ = αΔ + μδ₀; solve 2×2 for (α,μ) minimizing exact-F M(δ). Parameter-free. If h
  quadratic & deterministic ⇒ equiv to preconditioned CG. 4 scalars via 2 fwd passes (App C Jv).
- **Estimation:** online exp-decay of Ā,G: new = ε·old + (1−ε)·batch, ε=min{1−1/k,0.95}.
  Compact (size independent of #data) — the feature HF can't have.
- **Efficiency:** per-update ≈ several× SGD. Inverses every T₃=20 iters; subsets τ₁=1/8 (stats),
  τ₂=1/4 (exact-F mat-vec). Low-rank gradient trick when d>m: U_i=(1/m)(G⁻¹𝒢_i)(𝒜̄_{i-1}ᵀĀ⁻¹).
- **Invariance (Thm):** with negligible damping, block-diag/tridiag K-FAC updates are invariant to
  the network transform āᵢ†=Ωᵢφ̄(Φᵢsᵢ) (affine input transforms, sigmoid↔tanh, centering/whitening).
  Proof: J_ζ=diag(Ω_{i-1}ᵀ⊗Φᵢ); G†=ΦᵢᵀGΦⱼ, Ā†=ΩᵢĀΩⱼᵀ ⇒ F̃†_{i,j}=(Ω⊗Φᵀ)F̃(Ωᵀ⊗Φ) ⇒ J_ζᵀF̆J_ζ=F̆†.
  **Corollary (whitening):** block-diag K-FAC = plain GD on a network whose aᵢ,gᵢ are centered &
  whitened (Φ=G⁻¹ᐟ², Ω=Ā⁻¹ᐟ² make F̆†=I).

## Reference implementation (code/kfac.py, kfac_utils.py — alecwangcq/KFAC-Pytorch)
Per-layer hooks cache A=E[āāᵀ] (forward pre-hook) and G=E[ggᵀ] (backward hook), running avg
(update_running_stat). Every TInv: eigendecompose A=Q_a d_a Q_aᵀ, G=Q_g d_g Q_gᵀ. Natural grad:
v1=Q_gᵀ·(grad mat)·Q_a; v2 = v1 / (d_g d_aᵀ + damping); v = Q_g v2 Q_aᵀ — this is exactly
G⁻¹·grad·A⁻¹ with eigenvalue-space damping (factored Tikhonov realized as +λ on each Kron-eigenvalue
product). bias = last column of the matricized weight. kl_clip rescales the whole step
(ν=min(1,√(kl_clip/Σ v·g·lr²))) — a cheap stand-in for the exact-F rescaling. Then SGD-momentum.
Decoupled implementation: forward pre-hook builds ā (cat a with ones for bias); G uses the
model-sampled / batch-scaled grad_output.
