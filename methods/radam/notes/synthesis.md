# RAdam synthesis

## Pain point
Adaptive optimizers (Adam, RMSprop) empirically need a learning-rate **warmup** (small lr in first few
epochs) or they converge to bad local optima / blow up. Example: Transformer on De-En IWSLT'14 — no
warmup → train loss stuck ~10; with warmup → <3. No theory for why warmup helps, no guidance on how to
set it (T_w). Trial-and-error per task.

## Diagnostic finding (pre-method, about existing systems)
- Without warmup, the absolute-gradient histogram is **distorted within ~10 updates** (mass shifts to
  small values). The optimizer is trapped early.
- Controlled experiments (knowable diagnostics):
  - **Adam-2k**: freeze params + first moment, only update v_t (the ψ second-moment) for first 2000
    iters → then run Adam. Fixes the convergence problem. ⇒ root cause is *lack of samples to estimate
    the adaptive denominator*.
  - **Adam-eps**: use ε=1e-4 instead of 1e-8. Reduces variance of ψ̂ (if ψ̂~uniform, Var=1/(12ε²)).
    Avoids the blow-up but underperforms (large ε biases the rate). ⇒ need a *principled* variance
    control, not a crude floor.

## Adam in the generic framework (Reddi et al. 2019 framing)
θ_t = θ_{t-1} − α_t m_t l_t, with
- φ (momentum): m_t = ((1−β₁)Σ β₁^{t−i} g_i)/(1−β₁^t)  [bias-corrected EMA of g]
- ψ (adaptive rate): l_t = sqrt( (1−β₂^t) / ((1−β₂)Σ β₂^{t−i} g_i²) )  [= sqrt(bias-corrected 1/v_t)]
  numerically ψ̂ = sqrt(1−β₂^t)/(ε + sqrt((1−β₂)Σβ₂^{t−i}g_i²)).

## The variance argument
Model g_i ~ iid N(0,σ²) (valid early: weights init mean-0). 
- t=1 special case: ψ(g₁)=sqrt(1/g₁²). 1/g₁² ~ scaled-inv-χ²(1, 1/σ²). Var[sqrt(1/g₁²)] ∝ ∫₀^∞ x^{-1}e^{-x}dx
  which **diverges**. So the adaptive ratio is unboundedly variable at the start. Small lr scales variance
  by α² (Var[αx]=α²Var[x]) → warmup = variance reduction. This is the whole intuition.
- General t: EMA of g_i². Approximate EMA by SMA (Nau 2014): ψ²(.) ≈ t/Σg_i². Since g_i~N(0,σ²),
  t/Σg_i² ~ scaled-inv-χ²(t, 1/σ²). Assume ψ²(.) ~ scaled-inv-χ²(ρ, 1/σ²) generally.

### Theorem (Var monotone decreasing in ρ). Let x=ψ²~SInvχ²(ρ,τ²), τ²=1/σ².
PDF p(x) = (τ²ρ/2)^{ρ/2}/Γ(ρ/2) · exp(−ρτ²/2x) / x^{1+ρ/2}.
E[x] = ρ/((ρ−2)σ²)  (ρ>2).
E[√x] = τ√ρ Γ((ρ−1)/2) / (√2 Γ(ρ/2))  (ρ>1).
Var[ψ]=Var[√x]=E[x]−E[√x]² = τ²( ρ/(ρ−2) − ρ·2^{2ρ−5}/π · B((ρ−1)/2,(ρ−1)/2)² )  (finite for ρ>2; paper analyzes/plots it for ρ>4).
Appendix proof: d/dt of the bracket <0 for t≥4, via Legendre duplication + digamma bounds
(ln x − 1/(2x) > Ψ(x) > ln(x+0.5) − 1/x) + Gautschi's inequality. Net: Var decreases in ρ.
The exact variance is finite for ρ>2. The practical ρ>4 threshold comes from the first-order approximation below, because it uses Var[x], which is finite only for ρ>4.

## Estimating ρ via SMA length (ρ_t)
EMA≈SMA with length f(t,β₂) matched by **center of mass**:
(1−β₂)Σ_{i=1}^t β₂^{t−i} i / (1−β₂^t) = (Σ_{i=1}^{f} (t+1−i))/f.
Solve → f(t,β₂) = 2/(1−β₂) − 1 − 2tβ₂^t/(1−β₂^t).  [VERIFIED numerically, exact]
ρ_∞ := lim = 2/(1−β₂) − 1  (max SMA length).
ρ_t := ρ_∞ − 2tβ₂^t/(1−β₂^t).
Since SMA(f)/g of g_i² ~ SInvχ²(f,1/σ²), treat ρ ≈ ρ_t = f(t,β₂). [VERIFIED ρ_t solves CoM eq exactly]
ρ_∞ derivation: as t→∞, normalized EMA weight on age k=t−i is (1−β₂)β₂^k; E[age+1]=β₂/(1−β₂)+1=1/(1−β₂);
SMA(f) average index from new = (f+1)/2; set equal → f=2/(1−β₂)−1. [VERIFIED]

## First-order (delta-method) variance approx
Var[√x] ≈ Var[x]/(4E[x]). For SInvχ²(ρ,τ²): E[x]=ρτ²/(ρ−2), Var[x]=2ρ²τ⁴/((ρ−2)²(ρ−4)).
⇒ Var[ψ] ≈ ρ_t / (2(ρ_t−2)(ρ_t−4)σ²).  [VERIFIED matches delta method exactly]
Decays ~ O(1/ρ_t). Numerically stable substitute for the Beta-function analytic form.
Var(ρ=5) ≈ 100× Var(ρ=500).

## Rectification term
Want consistent variance across t: pin to the minimum (achieved at ρ_∞), C_var = Var[ψ]|_{ρ_∞}.
Var[r_t ψ] = C_var ⇒ r_t = sqrt(C_var / Var[ψ]|_{ρ_t}) = sqrt(Var|_{ρ_∞} / Var|_{ρ_t}).
Plug approx Var(ρ)=ρ/(2(ρ−2)(ρ−4)) (σ²,τ² cancel):
r_t = sqrt( (ρ_t−4)(ρ_t−2)ρ_∞ / ((ρ_∞−4)(ρ_∞−2)ρ_t) ).  [VERIFIED = sqrt(V(ρ_∞)/V(ρ_t)), r_t≤1, →1]

## The algorithm (RAdam)
ρ_∞ = 2/(1−β₂)−1 precomputed.
Each step: update v_t (EMA 2nd moment), m_t (EMA 1st moment), m̂_t=m_t/(1−β₁^t).
ρ_t = ρ_∞ − 2tβ₂^t/(1−β₂^t).
If ρ_t > 4 (variance tractable):
  l_t = sqrt((1−β₂^t)/v_t)  [bias-corrected inverse sqrt second moment]
  r_t = sqrt((ρ_t−4)(ρ_t−2)ρ_∞ / ((ρ_∞−4)(ρ_∞−2)ρ_t))
  θ_t = θ_{t−1} − α_t r_t m̂_t l_t   (rectified adaptive step)
Else (ρ_t ≤ 4, delta-method rectifier outside its valid region):
  θ_t = θ_{t−1} − α_t m̂_t   (un-adapted momentum SGD step)
If β₂ ≤ 0.6 then ρ_∞ ≤ 4, so the adaptive branch never activates. With `degenerated_to_sgd=True`, this is SGD with momentum; with the canonical default `False`, the reference implementation skips these inactive updates.
Code note: canonical impl uses ρ_t ≥ 5 ("more conservative since approximated") rather than >4, and defaults `degenerated_to_sgd=False`.

## Comparison to warmup
r_t increases from small toward 1 with t — same shape as linear warmup min(t,T_w)/T_w. So warmup ≈
heuristic variance reduction; r_t is the principled version. No T_w hyperparameter; auto-adapts to β₂.
RAdam additionally deactivates the adaptive rate while the stable rectifier is outside its valid region (ρ_t≤4).
Appendix "Downgrading to SGDM": replacing the first 1–4 RADAM steps with (divergent-variance) Adam
hurts more than replacing steps 5–8 → first updates most damaging, consistent with theory.

## Ancestors (cite by author/year)
- SGD / momentum.
- Adagrad (Duchi et al. 2011): per-coord lr ∝ 1/sqrt(Σ g²) — accumulate all past squared grads.
- RMSprop (Tieleman & Hinton 2012): EMA of squared grads instead of full sum.
- Adam (Kingma & Ba 2014): EMA first+second moments + bias correction.
- Nadam (Dozat 2016), Adadelta (Zeiler 2012): variants.
- Reddi et al. 2019 (AMSGrad / "On the convergence of Adam"): generic φ,ψ framework.
- Nau 2014: EMA-as-SMA, center-of-mass equivalence.
- Wolter 2007: first-order (delta-method) Taylor variance approx.
- Warmup: Vaswani et al. 2017 (Transformer), Goyal et al. 2017 (large-batch), Popel & Bojar 2018,
  Gotmare et al. 2018 (warmup stabilizes deep layers).
- Balduzzi et al. 2017 (shattered gradients; mean-0 init justification).

## Code framework / canonical impl
LiyuanLucasLiu/RAdam radam.py — torch Optimizer subclass, step() loop. Uses N_sma (=ρ_t), N_sma_max
(=ρ_∞), buffered per (step%10). `degenerated_to_sgd` defaults to `False`; if it is `True`, it controls the ρ_t≤4 momentum fallback.
