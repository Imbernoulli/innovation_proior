# NTK synthesis (pre-Phase-2)

## Pain point / research question
Wide nets generalize well despite overparametrization and a highly non-convex
parameter loss surface (Choromanska 2015, Dauphin 2014). Mystery: a big net can
fit random labels yet generalize on real labels (Zhang 2017); kernel methods do
the same (Belkin 2018). At init, infinite-width nets are GPs (Neal 1996; Lee
2018; Matthews 2018) — connecting nets to kernels, but only at init. Open: what
does **gradient descent** converge to? What governs the dynamics of the FUNCTION
during training? Track the function f_θ in function space, not the weights.

## Load-bearing ancestors
- **Neal 1996 / Lee 2018 / Matthews 2018 (NNGP)**: single hidden layer (Neal),
  deep (Lee/Matthews) — at init, f_θ → centered GP with covariance Σ^(L) given by
  layerwise recursion Σ^(1)=x·x'/n0+β², Σ^(L+1)=E_{f~N(0,Σ^L)}[σ(f(x))σ(f(x'))]+β².
  CLT/LLN over width. Gap: only describes init (Bayesian prior), says nothing
  about gradient-descent training dynamics.
- **Random features (Rahimi & Recht 2007)**: approximate a kernel K by P random
  functions f^(p) with E[f^(p)(x)f^(p)(x')]=K; linear param f_θ=1/√P Σ θ_p f^(p);
  GD on θ is EXACTLY kernel gradient descent w.r.t. tangent kernel K̃=1/P Σ f^(p)⊗f^(p),
  which →K as P→∞. This is the toy model that motivates the NTK — but it is
  linear in θ, so K̃ is constant; the net is nonlinear so its tangent kernel
  moves. Gap to bridge: show the nonlinear net's tangent kernel also freezes.
- **Daniely, Frostig, Singer 2016 (dual activations)**: Hermite-expansion dual
  μ̂(ρ)=Σ a_i² ρ^i for μ=Σ a_i h_i, (X,Y) Gaussian corr ρ. Used for the
  positive-definiteness proof on the sphere.
- **Gneiting 2013 / Schoenberg**: f(ρ)=Σ b_n ρ^n gives PD kernel on S^{n0-1} for
  all n0 iff infinitely many even and infinitely many odd b_n>0.
- **Kernel methods**: kernel PCA (Schölkopf 1998), kernel ridge regression
  (Shawe-Taylor 2004) — convergence/generalization machinery once we are in
  function space.
- **Mei/Montanari 2018, Karakida 2018**: mean-field 2-layer / Fisher geometry —
  contemporaneous large-width analyses; deep dynamics open.

## Setup / parametrization (the "why")
- Layers 0..L, widths n0..nL. α^(0)=x; α̃^(ℓ+1)=1/√(nℓ) W^(ℓ)α^(ℓ)+β b^(ℓ);
  α^(ℓ)=σ(α̃^(ℓ)). f_θ=α̃^(L). Params iid N(0,1).
- **1/√(nℓ) factor (NTK parametrization)**: keeps preactivations O(1) as width
  grows (consistent asymptotics) AND, crucially, makes ∂_W F scaled by 1/√n so
  individual weights move negligibly — this is exactly what freezes the kernel.
  Same representable functions as LeCun init, but the JACOBIAN scales differently.
- **β (bias multiplier)**: side-effect of 1/√n is that connection-weight
  gradients shrink relative to bias gradients; β rebalances bias vs connection
  influence. They use β=0.1, lr=1.0 (≈ classical width-100 net at lr 0.01).
  β too big (β=1) inflates the gap between 1st and 2nd kernel PCA → harder
  training (their MNIST footnote).

## Core derivations (all to be WORKED inline in reasoning.md)
1. **df/dt = kernel gradient.** θ̇_p=-∂_{θp}(C∘F)=-⟨d, ∂_{θp}F⟩_{pin} (d = dual of
   ∂_f C). Then ∂_t f = Σ_p ∂_{θp}F · θ̇_p = -Σ_p ⟨d,∂_{θp}F⟩ ∂_{θp}F = -∇_Θ C,
   with NTK Θ^(L)=Σ_p ∂_{θp}F ⊗ ∂_{θp}F. For MSE, d=f-f*, so
   ∂_t f = -Θ·(f-f*) on data (the linear ODE). Cost decreases: ∂_t C = -‖d‖²_Θ ≤0,
   so PD kernel ⇒ global convergence (C convex in function space even though
   non-convex in θ).
2. **Random-features warmup** (Rahimi): linear param ⇒ tangent kernel exactly K̃,
   →K by LLN. Bridge: net is the same picture but nonlinear ⇒ Θ random + moving.
3. **Init limit (Prop 1 / Thm 1).** Induction on depth, taking n1→∞,…,n_{L-1}→∞
   sequentially.
   - GP recursion Σ (Prop 1): L=1 affine ⇒ Σ^(1). Step: condition on α^(L),
     outputs are iid Gaussian with cov Σ̃^(L+1)=1/nL α^(L)(x)·α^(L)(x')+β²; LLN
     ⇒ Σ^(L+1)=E[σ(f(x))σ(f(x'))]+β², deterministic ⇒ conditional=unconditional.
   - NTK recursion (Thm 1): L=1, Θ=Σ^(1)δ. Step: split θ into θ̃ (first L) + last
     layer (W^L,b^L). Last-layer contribution → Σ^(L+1)δ. Lower-layer via chain
     rule ∂_{θ̃p}f_k=1/√nL Σ_i ∂_{θ̃p}α̃^L_i σ̇(α̃^L_i) W^L_{ik}; the cross term
     1/nL Σ_{ii'} Θ^L_{ii'} σ̇σ̇ W W → (induction Θ^L→Θ^L_∞ δ_{ii'}) → 1/nL Σ_i
     Θ^L_∞ σ̇σ̇ W² → LLN → Θ^L_∞ · Σ̇^(L+1) δ. Sum:
     Θ^(L+1)_∞ = Θ^L_∞ Σ̇^(L+1) + Σ^(L+1), with Σ̇^(L+1)=E[σ̇σ̇].
4. **Constancy during training (Thm 2).** Generalized training direction d_t,
   θ̇_p=⟨∂_{θp}F, d_t⟩. Need ∫‖d_t‖dt stochastically bounded (true for MSE since
   ‖d‖=‖f-f*‖ decreasing). Induction:
   - L=1 NTK param-independent ⇒ constant.
   - Smaller net sees back-propagated direction d'_t=σ̇(α̃^L)(1/√nL W^L)^T d_t.
     ‖d'_t‖≤c‖1/√nL W^L‖_op ‖d_t‖. Lemma (Appendix 3): ‖1/√nℓ (W^ℓ(t)-W^ℓ(0))‖_op→0,
     and ‖1/√nL W^L(0)‖_op bounded by LLN (rows bounded; n_{L+1} fixed). So apply
     induction hyp to smaller net.
   - Weight/activation drift: ∂_t‖W^L_i(t)-W^L_i(0)‖₂ ≤ 1/√nL ‖α^L_i‖‖d_t‖;
     ∂_t‖α̃^L_i(t)-α̃^L_i(0)‖ ≤ 1/√nL ‖Θ^L_∞‖_op ‖σ̇‖_∞ ‖W^L_i‖₂ ‖d_t‖.
     Define A(t)=‖α^L_i(0)‖+c‖α̃^L_i(t)-α̃^L_i(0)‖+‖W^L_i(0)‖₂+‖W^L_i(t)-W^L_i(0)‖₂;
     ∂_t A ≤ max{c²‖Θ^L_∞‖,1}/√nL ‖d_t‖ A ⇒ Grönwall ⇒
     A(t)≤A(0)exp(const/√nL ∫‖d‖) → A(0); drift = O(1/√nL).
   - NTK variation: top-layer summands ∂_{W^L}f⊗∂_{W^L}f vary at n_L^{-3/2} (each
     ∂=1/√nL α^L, α drifts at 1/√nL) ⇒ NTK varies at 1/√nL. Lower-layer term:
     W^L_{ij} drift 1/√nL; σ̇(α̃^L) drift 1/√nL via bounded σ̈
     (∂_t σ̇(α̃)=O(∂_t α̃)). ⇒ whole NTK frozen.
   - Lemma proof (Appendix 3): backprop directions d^(ℓ), subnet NTKs Θ^(ℓ)
     recursion; recursive op-norm bounds ‖d^ℓ‖≤c^{L+1-ℓ}∏w^k ‖d‖,
     ‖Θ^{ℓ+1}‖≤c²(w^ℓ)²‖Θ^ℓ‖+(a^ℓ)²+β² ⇒ polynomial P; A(t) drift bound
     ≤ 1/√min{n} Q̃(A)‖d_t‖ ⇒ nonlinear Grönwall, τ→T, A(t)→A(0), w^ℓ→0.
   - Remark: individual activation drift→0 but collective (×n neurons) is
     significant ⇒ lower layers DO contribute (the Θ^L_∞ Σ̇ term = lower-layer
     learning; Σ^(L+1) = last-layer learning). Counterintuitive: hidden reps
     barely move per-neuron yet net function learns.
5. **Least-squares (linear ODE in function space).** ∂_t f=Φ_K(⟨f*-f,·⟩);
   solution f_t=f*+e^{-tΠ}(f0-f*), Π(f)_k=1/N Σ_i Σ_{k'} f_{k'}(x_i)K_{kk'}(x_i,·).
   Diagonalize Π by kernel-PCA eigenfunctions f^(i), eigenvalues λ_i; component i
   decays e^{-λ_i t}. Fast directions = top kernel PCs ⇒ early stopping = fit top
   PCs, skip noisy low-λ (high-freq for RBF). t→∞: f_∞,k(x)=κ^T K̃^{-1}y* +
   (f0(x)-κ^T K̃^{-1}y0); mean = kernel ridge regression as λ→0 = ridgeless /
   MAP under GP prior f_k~N(0,Θ^L_∞); residual centered Gaussian vanishing on data.
6. **Positive-definiteness on sphere (Prop 2, App 4).** Θ^(L+1)=Σ̇^L Θ^L+Σ^(L+1);
   Σ̇Θ is PSD (product of PSD kernels) so PD of Σ^(L+1) ⇒ PD of Θ^(L+1). Σ^(L+1)
   PD ⇐ Σ^(L) PD (Gaussian non-degenerate, σ non-constant) ⇒ induct down to Σ^(2).
   Σ^(2)(x,x')=ν(x^Tx') with ν a power series whose coeffs come from Hermite
   coeffs a_i² of μ(x)=σ(x√(1/n0+β²)); σ non-polynomial ⇒ infinitely many a_i≠0
   ⇒ infinitely many even AND odd terms ⇒ Schoenberg/Gneiting ⇒ PD on S^{n0-1},
   L≥2. (Converse: polynomial σ ⇒ not PD for some n0.)

## Code grounding (Phase 1.4)
- Canonical: google/neural-tangents — `(init_fn, apply_fn, kernel_fn)`,
  `kernel_fn(x1,x2,'ntk')` computes the infinite-width NTK by the SAME layerwise
  Σ/Σ̇/Θ recursion (Dense→×, Relu/Erf→dual nonlinearity). JAX only (not installed).
- I'll ground final code in numpy+torch (both available):
  (a) **analytic** infinite-width NTK = the Σ, Σ̇, Θ recursion in closed form for
      ReLU (arccos kernels of Cho & Saul 2009) — mirrors neural-tangents' Dense+Relu.
  (b) **empirical** NTK via torch autograd Jacobian: Θ(x,x')=Σ_p ∂_{θ}f(x)·∂_θ f(x')
      — the finite-width object the theorems take the limit of.
  (c) closed-form kernel-regression predictor f_∞ = κ^T K̃^{-1} y (ridgeless).
- ReLU dual closed forms (κ0,κ1 arccos kernels), used by neural-tangents Relu:
  Σ^{ℓ+1}=  (1/2π) √(Σxx Σx'x') (sinθ+(π-θ)cosθ)+β², θ=arccos(Σxx'/√(ΣxxΣx'x'));
  Σ̇^{ℓ+1}= (1/2π)(π-θ).

## Scaffold ↔ final code (piece-for-piece)
final code modules → context scaffold stubs:
- relu_dual(Λ) {κ0,κ1}  ←  `def activation_dual(cov): # TODO`
- nngp_and_derivative recursion  ←  `def layer_kernel_recursion(...): # TODO`
- infinite_ntk(X,X')  ←  `def tangent_kernel(...): # TODO` (the contribution slot)
- empirical_ntk(net,X,X') via autograd  ←  `def finite_tangent_kernel(...): # TODO`
- kernel_regression(K,Ktest,y) ridgeless  ←  generic `def kernel_predict(...): pass`
- WideMLP(nn.Module) with 1/√n scaling  ←  generic `class Net: # TODO scaling`

## In-frame reminders
Never name the paper/method-as-paper. May name "Neural Tangent Kernel" as the
thing being built (mainly answer.md). Cite ancestors (Neal 1996, Rahimi & Recht
2007, Daniely 2016, Cho & Saul 2009, Gneiting 2013) freely. No eval results in
context/reasoning (the convergence/inflation MNIST figures are the proposed
method's own experiments — out of scope).
