# Synthesis — Slow SDE for Adam / AGMs near the minimizer manifold

## Method short-name
`adam-slow-sde` — the slow SDE characterizing the long-horizon implicit bias of Adam (and a general class of adaptive gradient methods, AGMs) near a manifold of minimizers, plus the consequence that Adam's implicit regularizer under label noise is tr(Diag(H)^{1/2}) (vs SGD's tr(H)). Companion variant AdamE-λ → tr(Diag(H)^{1-λ}).

## Pain point / research question
- Implicit bias = which of the many global minimizers does an optimizer pick. Sharpness (flatness) correlates with generalization.
- For SGD near a minimizer manifold, gradient noise drives a slow drift that reduces sharpness; this is captured by a *slow SDE* valid for O(η^{-2}) steps (Li-Wang-Arora 2021; Gu-Lyu-Huang-Arora 2023 termed "slow SDE"). Under label noise, SGD ≈ gradient flow on tr(H).
- Adam dominates practice but its implicit bias near the manifold was unknown. Existing attempts: Ma 2023 (2D, non-rigorous quasistatic); Cattaneo 2024 IGR (only O(η^{-1}) horizon, claims Adam *anti*-regularizes sharpness when β1<β2); Wang 2021 (Adam=SGD regularizer but needs grad entries < ε, unrealistic); Zhang 2024 (linearly separable only). None give a rigorous O(η^{-2}) characterization.

## Background facts (pre-method, knowable before)
- Minimizer manifold assumption: Γ is a C∞ (d-m)-dim compact submanifold; each ζ∈Γ local min; rank(∇²L(ζ))=m. Motivated by mode connectivity / river-valley loss landscapes (Garipov 2018, Wen 2024).
- Two-phase dynamics: O(η^{-1} log) convergence phase onto Γ, then O(η^{-2}) manifold phase of slow implicit regularization. Conventional SDE (Li 2017/2019, Malladi 2022) tracks only convergence phase; error blows up on manifold. Slow SDE = peel off convergence (project onto Γ), track only the slow motion.
- Gradient-flow projection Φ(x)=lim gradient flow; ∂Φ(ζ) = orthogonal projection onto T_ζΓ.
- Slow SDE for SGD (Li 2021, Gu 2023 form): dζ = P_ζ( Σ_∥^{1/2} dW − ½ ∇³L(ζ)[Σ̂_◇] dt ). Drift = negative semi-gradient of μ(ζ)=⟨∇²L(ζ), Σ̂_◇(ζ)⟩ (semi = differentiate only first arg). Σ_∥=tangent noise, Σ_◇=normal-space noise weighted by 1/(λi+λj).
- Label noise: Σ(ζ)=α∇²L(ζ) on Γ (from ℓ2 regression with fresh ±δ label noise; Σ=E[ζ² ∇h∇h^T]=δ²∇²L). Under label noise slow SDE for SGD → ODE → fixed points satisfy ∇_Γ tr(H)=0. (Blanc 2020; Damian 2021; Li 2021)
- Diagonal net (Woodworth 2020): θ=(u,v), ŵ=u⊙²−v⊙², loss ½(⟨z_i,ŵ⟩−y_i)². Overparam d≫n. On Γ, diag(∇²L)=4θ⊙². tr(Diag(H)^{e0}) ∝ Σ(|u_i|^{2e0}+|v_i|^{2e0}). Minimizing it forces u_i=0 or v_i=0 ⇒ equals ‖ŵ‖_{e0}^{e0}. So SGD↔ℓ1, Adam↔ℓ0.5. lasso(ℓ1)≻ridge: ℓ0.5 even sparser ⇒ Adam recovers sparse ground truth with less data.
- Matrix factorization (Gatmiry 2023): minimizing tr(H) ≈ minimizing nuclear norm of W* ⇒ favors low rank. Adam's tr(Diag(H)^{1/2}) does NOT favor low rank ⇒ Adam generalizes WORSE here (diagnostic counter-case). Adam drives tr(Diag(H)^{1/2}) down while tr(H) stays high/non-monotone.

## The method (AGM framework + slow SDE)
AGM step:
  m_{k+1}=β1 m_k+(1-β1) g_k
  v_{k+1}=β2 v_k+(1-β2) V(g_k g_k^T)
  θ_{k+1}=θ_k − η S(v_{k+1}) m_{k+1}
where V:R^{d×d}→R^D linear, V(gg^T)≥0; S:R^D_{≥0}→S^d_{++}, ρ_s-smooth, S(v)⪰ I/R0. Adam: V=diag, S=Diag(1/(√v+ε)). RMSProp=Adam β1=0. AdamE-λ: S=Diag(1/(v^{⊙λ}+ε)). Adam-mini/Adalayer: block/layer-shared v. Shampoo: V=(V_L,V_R), S=Kronecker inverse sqrt.

Preconditioner-flow projection Φ_S(x)=lim of dx/dt=−S∇L(x). Adaptive projection P_{ζ,S}.

Slow SDE for AGMs:
  dζ = P_{ζ,S(t)}( Σ_∥^{1/2}(ζ;S) dW − ½ S(t) ∇³L(ζ)[Σ_◇(ζ;S)] dt )
  dv = c (V(Σ(ζ)) − v) dt,   c=(1-β2)/η²
  Σ_◇(ζ;S)=S Σ S − Σ_∥;  Σ_∥(ζ;S)=∂Φ_S S Σ S ∂Φ_S.
Interpretation: adaptive *semi-gradient* descent on μ(ζ,v)=⟨∇²L(ζ), Σ_◇⟩, preconditioned by S(t), with the preconditioner itself an O(η^{-2})-timescale OU-like state.

Main theorem: weak approximation, for any C³ test g, max_k |E[g(X̄_k)]−E[g(X(kη²))]| = Õ(η^{0.25}) over K=⌊Tη^{-2}⌋ steps, after K0=O((1/η)log(1/η)) convergence steps.

## Design decisions → why (rejected alternatives + failure modes)
| Decision | Why / alternative rejected |
|---|---|
| Slow SDE (project onto Γ) not conventional SDE | conventional SDE tracks only convergence phase O(η^{-1}); approximation error unbounded on the manifold where the implicit bias lives. |
| 2-scheme: 1-β2=Θ(η²) | If β2 far from 1 (1-β2 large), preconditioner moves too fast → moments untrackable; if β2 too close to 1, preconditioner frozen → no adaptiveness (≈SGD). η² is the *unique* rate matching the O(η^{-2}) manifold timescale: preconditioner changes Θ(1) over exactly the SDE timescale, slow enough to track yet fast enough to matter. Also β2<β1² ⇒ Adam may diverge (Reddi 2018). |
| Preconditioner-flow projection Φ_S (not gradient-flow Φ) | S breaks SGD's rotational equivariance; can't diagonalize Hessian/treat Γ as coordinate subspace. Reparameterize x'=P^{-1}x with P=S0^{1/2}: in x'-space the preconditioned flow becomes a *plain* gradient flow, so reuse Li2021/Gu2023 moment formulas (Lemmas I.36/I.37), then map back. |
| Semi-gradient (freeze Σ_◇'s ζ-dependence) | The drift is literally −½∇_{ζ1}μ(ζ1,ζ2)|_{ζ1=ζ2=ζ}; the noise covariance's own ζ-dependence does not enter the drift (it's a second-order moment effect). Same as SGD case. |
| β1 (momentum) of constant order, β1≤0.9, doesn't affect bias | After convergence ∇L moves slowly; momentum averages only O(log 1/η) past steps so E[m]≈E[g] to O(η^{1.5} log). Matches Wang2023 "momentum marginal". Threshold 0.9 only for clean constants; any const<1 works. |
| C5 smoothness of L, C4 of S | need Φ_S to be C4 (so ∂²Φ_S exists & is continuous for the drift), which needs ∇L∈C4 ⇒ L∈C5; weak approximation needs C³ test functions ⇒ drift/diffusion ∈ C4. |
| High-prob convergence to L(θ_K)-L*=Õ(η) directly (not avg-grad, not in-expectation) | the manifold analysis needs each step to be near Γ w.h.p.; prior Adam bounds are loose / only in expectation / only average gradient norm / don't →0 as η→0. Built via descent lemma + Azuma on martingale X_i + μ-PL (proved to hold in a tube around Γ via Lyu2022). |
| μ-PL only local → proxy loss L̃ with quadratic "wall" | μ-PL only holds in Γ^{ε3}; build L̃=L+½C(dist−ε1)² outside ε1 making it (μ,L̄)-PL globally; sublevel set {L̃<Lm}⊆Γ^{ε1} where L̃=L. Tubular neighborhood theorem gives smooth dist/normal; nonobtuse angle ⟨∇L,n⟩≥0 near Γ. |
| label noise Σ=αH | makes slow SDE → ODE (diffusion Σ_∥^{1/2} vanishes since ∂Φ_S S Σ^{1/2}=0: with Σ=αH, Σ^{1/2}x∈normal space, killed by ∂Φ_S S). Fixed point ⇒ S P_∥ ∇³L[S]=0. With S=Diag((αdiag H)^{-1/2}): ∇³L[S]=Σ_j (αH_jj)^{-1/2}∇H_jj = (2/√α)∇tr(Diag(H)^{1/2}). ⇒ ∇_Γ tr(Diag(H)^{1/2})=0. |
| AdamE-λ knob | S=Diag(1/(v^λ+ε)); ∇³L[S]=Σ (αH_jj)^{-λ}∇H_jj=(1/((1-λ)α^λ))∇tr(Diag(H)^{1-λ}) ⇒ regularizer tr(Diag(H)^{1-λ}); λ=0→SGD(ℓ1/tr H), λ=1/2→Adam(ℓ0.5). interpolates implicit bias. |
| Shampoo: no explicit regularizer | the vector field A(ζ)=∇³L[S(V(Σ))] is non-conservative (nonzero curl, by Stokes-Cartan) even for diagonal H ⇒ no potential ψ with ∇ψ=A ⇒ no closed-form regularizer. |

## Canonical code grounding
No official repo found for the target paper (theory paper). Code is grounded in: (a) standard PyTorch Adam single-step update (m,v EMAs; θ-=lr·m/(√v+ε)), generalized to the AGM (V,S) form and AdamE-λ exponent; (b) the diagonal-net sparse-regression-with-label-noise experiment exactly as specified (θ=(u,v), ŵ=u²−v², fresh ±δ label noise each step, d≫n, test on clean labels). State this grounding explicitly.

## Ancestors (load-bearing)
- Blanc-Gupta-Valiant-Valiant 2020: SGD+label noise locally (O(η^{-1.6})) decreases tr(∇²L) via OU-like process near zero-loss points; moves off ζ iff tr(H) not locally minimal. Limit: local, short horizon, label-noise-specific.
- Damian-Ma-Lee 2021: constant-LR, any KL-smooth loss, also SGDM; global. Limit: still tied to SGD noise structure.
- Li-Wang-Arora 2021 "What happens after SGD reaches zero loss": Katzenberger/giant-step slow SDE, O(η^{-2}), arbitrary noise covariance, general drift with ∇³L and Lyapunov-weighted normal covariance. Limit: SGD-specific (additive gradient noise, rotational structure); can't extend to preconditioned methods.
- Gu-Lyu-Huang-Arora 2023 "Why does local SGD generalize": termed "slow SDE", derived for local SGD via a structure generalizable to other optimizers; supplies the reusable moment lemmas (I.36/I.37). 
- Wang et al 2023 "marginal value of momentum": SGD and SGDM share the same slow SDE.
- Kingma-Ba 2014 Adam; Hinton 2012 RMSProp; Gupta 2018/Morwani 2024 Shampoo; Woodworth 2020 diagonal net; Gatmiry 2023 deep matrix factorization flatness↔nuclear norm; Reddi 2018 Adam non-convergence; Malladi 2022 SDEs for adaptive (conventional).
