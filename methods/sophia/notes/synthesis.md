# Sophia — synthesis notes (pre-Phase-2)

## Pain point / research question
LLM pre-training is dominated by optimizer step count. Adam has been SOTA for years; a 2x reduction in steps-to-loss = 2x less compute/wall-clock/$. Can we cheaply use curvature to beat Adam *in wall-clock*, not just in steps? Constraint: any extra per-step cost (Hessian compute/memory) must be amortizable — it has to be offset by the step-count speedup. K-FAC / Shampoo / AdaHessian compute curvature too often → per-step overhead kills the wall-clock win. No prior second-order method had shown wall-clock speedup on decoder-only LLMs.

## The diagnostic: heterogeneous curvature
- Hessian diagonal of GPT-2 125M: positive entries are *dispersed* over orders of magnitude (Fig "lt"). Loss landscape has very different curvature per coordinate.
- 2D toy: L(θ1,θ2)=L1(θ1)+L2(θ2), L1 sharp (curvature h1), L2 flat (h2), h1≫h2.
  - L1(θ1)=8(θ1−1)²(1.3θ1²+2θ1+1) — note this is NON-convex (has the saddle/max region), L2(θ2)=½(θ2−4)².
- GD: optimal lr ∝ 1/curvature. Per-coord optimal lr is 1/h1 vs 1/h2; shared lr capped at 1/h1 (else blow up in sharp dim) ⇒ flat dim crawls. Convergence depends on condition number h1/h2.
- Adam ≈ SignGD (drop EMA: η·∇L/|∇L| = η·sign(∇L)). Update size = η in every coordinate. Same step everywhere ⇒ in flat dim the loss decrease per step (≈ ½h2·η²) is tiny; in sharp dim it overshoots/bounces (must decay η to converge, which kills flat-dim progress). Adam's preconditioner is "first-order" (uses grad magnitude only), does NOT equalize loss decrease across heterogeneous curvature.
- Want: smaller relative step in sharp dims, bigger in flat dims, so loss decrease is EQUALIZED → Newton. Per-coord Newton: θi ← θi − η·L'i/hi. Loss decrease per coord ≈ ½ g²/h (uniform if we move each to its quadratic min).

## Why not vanilla Newton
- Non-convex: H can be indefinite. Negative curvature dim: −g/h points UPHILL (toward a max). In toy, Newton converges to the saddle (blue curve). Catastrophic.
- H changes rapidly along trajectory ⇒ −H⁻¹g extrapolated from stale/local quadratic can be wildly wrong, huge.
- Full H is d×d, infeasible for LLMs. Even forming/inverting per step is out.
Three problems: (1) cost/size, (2) indefiniteness → blow-up to max/saddle, (3) non-stationary H → unreliable.

## Resolution sketch (what Sophia is)
1. Only DIAGONAL of H (per-coordinate curvature), cheap to store (same size as params).
2. Only POSITIVE curvature (clamp/estimate PSD), so update is always descent.
3. Estimate H infrequently — every k=10 steps — and EMA-denoise it. Possible *only because* of clipping (safeguard).
4. Per-coordinate CLIP of the update at ρ: bounds worst-case step. When h tiny/neg/mis-estimated, update would blow up; clip caps it at ρ → falls back to SignSGD-sized step (descent-lemma safety). Where h is well-estimated and not clipped, you get the full Newton step. ⇒ uniform progress across heterogeneous curvature, robust to bad H.

Update: m_t = β1 m_{t-1} + (1−β1) g_t (EMA grad, numerator).
h_t = β2 h_{t-k} + (1−β2) ĥ_t every k steps, else carry (EMA Hessian, denominator).
θ ← θ − η_t λ θ (decoupled WD); θ ← θ − η_t · clip( m_t / max{γ h_t, ε}, 1 ).
Note identity: η·clip(m/max{γh,ε},1) = (η/γ)·clip(m/max{h,ε/γ}, γ). So γ controls the *fraction clipped*; clipped entries all become ±η_t (≡ SignSGD step). Choose γ so 50–90% of coords are clipped in practice (tune to clip fraction; γ=0.01 Sophia-H, 0.05 Sophia-G).
Neg-h case: m/max{γh,ε} = m/ε (huge, sign of m) → clip → ±1 → step = η·sign(m) = SignSGD. So Sophia degrades gracefully to momentum-SignSGD exactly where curvature is unusable.
Worst-case update size ηρ; because most coords are clipped-or-auto-adjusted, can pick ηρ > η of plain SignSGD.

## Two diagonal Hessian estimators
### (a) Hutchinson (unbiased, needs HVP)
u ~ N(0, I_d). ĥ = u ⊙ (∇²ℓ(θ) u). E[ĥ] = diag(∇²ℓ).
Proof: E[u (Hu)]_i = E[u_i Σ_j H_ij u_j] = Σ_j H_ij E[u_i u_j] = H_ii since E[u_i u_j]=δ_ij. ✓ (works for any u with E[uuᵀ]=I, Gaussian or Rademacher.)
Cost: one Hessian-vector product = ∇(⟨∇L,u⟩) = double backward, run-time ~ a constant × gradient. No full Hessian. Can be indefinite (ĥ entries can be negative) — that's fine, clip/ε handle it.
Algorithm: compute L(θ); draw u; return u ⊙ ∇(⟨∇L(θ), u⟩).

### (b) Gauss-Newton-Bartlett (biased, gradient-only, always PSD)
Setup: ℓ(θ,(x,y)) = ce(f(θ,x), y), f = logits ∈ R^V, ce = cross-entropy.
Gauss-Newton decomposition (chain rule):
∇²_θ ℓ = J_θf · S · J_θfᵀ + J_θθf[q]
where J_θf ∈ R^{d×V} Jacobian, S = ∂²ce/∂t² |_{t=f} ∈ R^{V×V} (Hessian of loss wrt logits), q = ∂ce/∂t first deriv wrt logits, J_θθf[q] = second deriv of f contracted with q.
Drop second term (small for NNs, Sankar et al. 2021) → GN matrix G = J_θf S J_θfᵀ, always PSD (S PSD because ce convex in logits). PSD ⇒ preconditioned update always descent direction.
Key fact 1: S depends only on logits t=f, NOT on label y. For softmax+ce: S = diag(p) − p pᵀ where p = softmax(f). (General exponential-family property: Hessian of neg-log-likelihood depends only on natural params.)
⇒ S = E_{ŷ~Cat(t)}[ ∂²ce(t,ŷ)/∂t² ] (expectation trivial since S independent of label).
Bartlett's 2nd identity (for neg-log-likelihood of a model, here Cat(t)):
E_{ŷ~Cat(t)}[ ∂²ce/∂t² ] = E_{ŷ~Cat(t)}[ (∂ce/∂t)(∂ce/∂t)ᵀ ].   (Fisher = expected Hessian of NLL.)
So S = E_{ŷ~Cat}[ (∂ce/∂t)(∂ce/∂t)ᵀ ].
Push through J_θf: J_θf S J_θfᵀ = E_{ŷ}[ ∇_θ ce(f,ŷ) ∇_θ ce(f,ŷ)ᵀ ]   (since J_θf ∂ce/∂t = ∇_θ ce by chain rule).
⇒ diag(G) = E_{ŷ~Cat}[ ∇_θ ce(f,ŷ) ⊙ ∇_θ ce(f,ŷ) ].  (one-example unbiased estimator of GN diagonal: ∇ce(f,ŷ)⊙∇ce(f,ŷ) with ŷ sampled from model.)
Mini-batch: want (1/B)Σ_b ∇ce_b ⊙ ∇ce_b. But autodiff gives only the AVERAGED gradient over batch, not per-example. Fix via Bartlett's 1st identity: E_{ŷ_b}[∇ce(f(x_b),ŷ_b)] = 0 (score has zero mean under model). Sampled labels independent across b ⇒ cross terms vanish in expectation:
Let L̂(θ) = (1/B)Σ_b ce(f(θ,x_b), ŷ_b) (sampled-label minibatch loss).
E[ B · ∇L̂ ⊙ ∇L̂ ] = E[ (1/B) (Σ_b ∇ce_b) ⊙ (Σ_b ∇ce_b) ] = E[ (1/B) Σ_b ∇ce_b ⊙ ∇ce_b ] (cross terms E[∇ce_b]⊙E[∇ce_{b'}]=0).
= diag(GN of minibatch loss). ✓
So estimator = B · ∇L̂ ⊙ ∇L̂, computed from ONE ordinary minibatch gradient on RESAMPLED labels ŷ_b ~ softmax(f). Always ≥ 0. Cost = one gradient. (This is the "B·(ĝ⊙ĝ)" form.)
Squared loss: S = I, GN = J Jᵀ, can use directly; equiv to y~N(f,σ²).
Comparison: Hutchinson assumes nothing but needs HVP, can be indefinite; GNB is gradient-only + always PSD (guaranteed descent), but biased (drops the J_θθf[q] term).

## Theory (simplified, convex) — descent lemma + clip + 2-phase + SignGD lower bound
Analyze deterministic clipped-Newton on strictly convex L (clipping done in eigenbasis since diagonal in coords needn't align):
θ+ = θ − η Vᵀ clip(V (∇²L)⁻¹ ∇L, ρ), where ∇²L = Vᵀ Σ V.
Assumptions: (A1) twice-cts-diff strictly convex, μ = λ_min(∇²L(θ*)). (A2) multiplicative Hessian Lipschitz: ‖θ−θ'‖≤R ⇒ ‖∇²L(θ')⁻¹ ∇²L(θ)‖ ≤ 2.
Descent Lemma: for ηρ ≤ R/√d,
L(θ+) − L(θ) ≤ −(η − η²) Σ_i min{ ρ|vᵢᵀ∇L|, σᵢ⁻¹|vᵢᵀ∇L|² }.
Proof: f(t)=L(tθ+ +(1−t)θ); A2 ⇒ f''(t)≤2f''(0); f(1)≤f(0)+f'(0)+f''(0).
- f'(0) = ⟨∇L, −ηVᵀu⟩ = −η Σ min{ρ|vᵢᵀ∇L|, |vᵢᵀ∇L|²/σᵢ} where u=clip(Σ⁻¹V∇L,ρ).
  (per coord: vᵢᵀ∇L · clip(σᵢ⁻¹ vᵢᵀ∇L, ρ) = min of the two by sign-matching.)
- f''(0) = η²⟨u,Σu⟩ = η² Σ |uᵢ|²σᵢ ≤ η² Σ min{|vᵢᵀ∇L|²/σᵢ, ρ|vᵢᵀ∇L|} (|uᵢ|=min{|vᵢᵀ∇L|/σᵢ, ρ}).
Two regimes per coordinate:
- Unclipped (small grad / safe Newton): decrease ≈ |vᵢᵀ∇L|²/σᵢ — the full Newton quadratic decrease, condition-number-free.
- Clipped (huge/neg/unreliable curvature): decrease ≈ ρ|vᵢᵀ∇L| — SignGD-like guaranteed decrease bounded by ρ. The clip turns a possibly-uphill Newton step into a bounded safe descent step. THIS is the safety.
Phase 1 (burn-in): drive L−minL to μρ²/8 in O((L0−minL)/(ημρ²)) steps. Phase 2: once close, no coord clips (small-loss ⇒ ‖H⁻¹∇L‖≤ρ), pure Newton ⇒ L−minL ≤ (1−η(1−η))^{t−T}(...), linear/exp convergence, log(1/ε).
Final: η=1/2, ρ=R/(2√d) ⇒ T ≲ d·(L0−minL)/(μR²) + ln(μR²/(32dε)). NO condition-number, NO smoothness (max-eigenvalue) dependence.
SignGD lower bound (Thm): on L=½μθ1²+½βθ2², SignGD needs T ≥ ½(√(Δ/ε)−√2)·√(β/μ) — DOES depend on √(condition number). So Adam-proxy provably worse on heterogeneous curvature.

## Design-decision → why table
- Diagonal (not full/block) Hessian: only affordable curvature for LLMs; per-coord curvature is what heterogeneity needs; same memory as params. Block-diag K-FAC too expensive.
- Positive-curvature only: negative curvature → Newton goes uphill to max/saddle; restrict to PSD so update is descent. GNB always PSD by construction; Hutchinson handled by max{γh,ε} (neg h → ε → SignSGD fallback).
- EMA of h (β2): single estimate noisy like minibatch grad; denoise across iters, mirror Adam's 2nd-moment EMA.
- k=10 (infrequent Hessian): amortize cost (5% overhead) so step-count win → wall-clock win. Safe ONLY because clipping tolerates stale/wrong h. k=1 no extra benefit but 2x cost; k=100 too stale. (Ablation: k=10 best speed/overhead balance.)
- Per-coordinate clip at ρ (the crux): bounds worst-case update; neutralizes indefinite/stale/mis-estimated curvature (descent-lemma); lets us be infrequent+stochastic. Simpler/cheaper than trust-region/line-search/cubic-regularization which solve the same Newton-safety problem.
- γ reparam (η·clip(m/max{γh,ε},1) form): γ controls fraction clipped (tune to 50–90% clipped / equivalently keep "win rate" — unclipped fraction — ~10–50%); the η/γ·clip(·,γ) identity makes overall scale ~independent of γ, so lr and γ decouple in tuning.
- EMA of gradient m (β1, numerator): momentum, same as Adam; denoise gradient.
- Decoupled weight decay (θ ← (1−ηλ)θ): AdamW-style, standard for LLMs.
- Clip the UPDATE (not the gradient): novel vs prior diagonal-Hessian methods (AdaHessian etc. estimate H every step, no update-clip, hence pay big overhead). Update-clip is what tames H's change-along-trajectory + estimate inaccuracy.
- GNB B-factor: B·∇L̂⊙∇L̂ — recovers per-example sum from averaged-gradient autodiff via Bartlett-1 (cross terms vanish). In code: bs = total_bs * block_size folded into denominator (rho*bs*h).

## Canonical code (Liuhong99/Sophia, nanoGPT base) — SophiaG
- optimizer SophiaG: state exp_avg (m), hessian (h), step. betas=(β1,β2). rho≈γ. weight_decay decoupled.
- update_hessian(): h.mul_(β2).addcmul_(p.grad, p.grad, 1−β2)  ← GNB: grad here is from sampled-label loss; B folded via bs in step.
- step(): param.mul_(1−lr·wd); exp_avg EMA; ratio = (|m| / (rho·bs·h + 1e-15)).clamp(max=1); param.addcmul_(sign(m), ratio, −lr). NB sign(m)·clip(|m|/(...),1) == clip(m/(...),1).
- training loop GNB branch (every hess_interval=k): logits=model(X); y_sample ~ Categorical(logits); loss=CE(logits, y_sample); backward; optimizer.update_hessian(); zero_grad. Non-hess steps: normal forward/backward on real Y, optimizer.step(bs=total_bs*block_size).
- Hutchinson variant (Sophia-H): not in this minimal repo's optimizer (it ships SophiaG); HVP via torch.autograd HVP + u⊙Hu would feed the same update_hessian slot. Provide Sophia-H estimator in answer as the HVP form.

## In-frame discipline
Never name "Sophia paper"/authors/arXiv. May name method "Sophia" in answer.md. Ancestors (Adam/Kingma2014, RProp/Braun, RMSProp/Hinton, AdaHessian/Yao2021, K-FAC/Martens2015, Hutchinson1989, Bartlett1953, Schraudolph2002, Wei2020, Lion/Chen2023, AdamW/Loshchilov) cited freely. context.md scaffold: generic optimizer harness, no "Sophia"/method names; Background/Baselines includes SGD/Adam/AdaHessian/Newton + indefinite-Hessian & cost diagnostic.
