# LAMB synthesis (pre-Phase-2)

## Pain point / research question
Training large nets (BERT, ResNet-50) takes days. The only knob to parallelize SGD across many accelerators without async staleness is the batch size: bigger batch b → variance of stochastic gradient drops by ~1/b → fewer, larger steps. But naively scaling b:
- **Fewer steps**: at fixed #epochs, #updates T = (#examples × epochs)/b falls linearly. To keep progress per epoch you must raise the LR.
- **A single global LR fails**: linear LR scaling (Krizhevsky 2014, Goyal 2017) works only up to a batch-size ceiling and needs hand-tuned warmup; beyond it, training is unstable / generalization gap opens (Keskar 2016 sharp minima, Hoffer 2017). Shallue 2018: scaling heuristics don't transfer across problems/batch sizes.
- The instability is because **one global LR is wrong**: different layers have wildly different ratios ‖w‖/‖update‖. A global LR large enough for a layer with a tiny update-to-weight ratio blows up a layer with a large ratio.

## Lineage / load-bearing ancestors
- **SGD nonconvex (Ghadimi & Lan 2013)**: with b=T, E‖∇f‖² ≤ O((f(x1)−f*)L_∞/T + ‖σ‖²/T). Bound depends on **L_∞ = max_i L_i** (worst-layer smoothness) — pessimistic when curvature is uneven across layers.
- **Linear LR scaling + warmup (Goyal et al. 2017)**: LR ∝ b up to b=8192; gradual warmup to avoid early instability. sqrt scaling (Krizhevsky 2014) from variance ∝1/b argument.
- **LARS (You et al. 2017)**: layerwise trust ratio on top of momentum-SGD. Per layer i: x ← x − η·(φ(‖x^(i)‖)/‖m^(i)‖)·m^(i). Decouples per-layer effective step from the global LR; trained ResNet-50 at 32k batch. BUT: no theory, and **fails on attention models (BERT)** — diverges at 32k.
- **Adam / AdamW**: per-coordinate adaptive (m̂/(√v̂+ε)); AdamW = decoupled weight decay. Good for BERT but **plateaus / can't reach SGD-momentum accuracy on ResNet** and **stops scaling at 16k for BERT** (F1 88 vs 90.4), diverges at 64k.
- **signSGD (Bernstein et al. 2018)**: sign(g) update; convergence comparison via density quantities ψ; the sign-disagreement probability bound P(sign g ≠ sign ∇f) ≤ σ/(√b·|∇f|) (Markov/bounded-variance). LAMB's β2>0 proof imports this.

## General strategy (the unifying frame)
Take any base optimizer A producing layerwise update u_t (x←x+η u_t). Two modifications for large batch:
1. **Normalize the update per layer to unit ℓ2 norm**: u_t/‖u_t‖. Kills gradient magnitude (robust to explode/plateau); in large batch the *direction* is well-estimated so dropping magnitude is cheap. Small bias.
2. **Scale per-layer LR by φ(‖x^(i)‖)**: makes the update norm of the same order as the weight norm. φ(z)=min(max(z,γ_l),γ_u) clipped identity. When φ(z)=z the per-layer multiplier is ‖x^(i)‖/‖u^(i)‖ → interpretable as an estimate of 1/L_i.
- A = momentum-SGD ⇒ LARS.
- A = Adam ⇒ **LAMB**.

## LAMB update (Algorithm 2)
m_t = β1 m_{t-1} + (1−β1) g_t
v_t = β2 v_{t-1} + (1−β2) g_t²
m̂_t = m_t/(1−β1^t), v̂_t = v_t/(1−β2^t)   [debiasing; optional — same effect as warmup, can drop]
r_t = m̂_t/(√v̂_t + ε)                       [the Adam direction]
x_{t+1}^(i) = x_t^(i) − η_t · φ(‖x_t^(i)‖)/‖r_t^(i)+λ x_t^(i)‖ · (r_t^(i) + λ x_t^(i))
- λ = decoupled weight decay added INSIDE the trust-ratio numerator-direction and the norm (AdamW-style, but inside the layerwise normalization).
- β1=0,β2=0 ⇒ signSGD scaled by √(layer dim).
- Default β1=.9, β2=.999, ε=1e-6, λ=.01.

## Convergence (proofs worked in reasoning)
Setup: f L_i-smooth per layer, bounded variance σ_i² / σ̃ per-dim, gradients bounded by G, b=T, η constant, α_l ≤ φ ≤ α_u.

### LARS (β1=0, λ=0): update x^(i) ← x^(i) − η φ(‖x^(i)‖) g^(i)/‖g^(i)‖
Per-layer smoothness descent lemma:
f(x_{t+1}) ≤ f(x_t) + Σ_i ⟨∇_i f, Δx^(i)⟩ + Σ_i (L_i/2)‖Δx^(i)‖².
‖Δx^(i)‖ = η φ ≤ η α_u so curvature term ≤ (η²α_u²/2)‖L‖_1.
Add-subtract the true-gradient direction ∇_i f/‖∇_i f‖: the inner-product term splits into −η Σ φ ‖∇_i f‖ (signal) plus an error from g vs ∇f directions. With Δ_t = g−∇f, Cauchy-Schwarz on the error term bounds it by 2 η Σ φ ‖Δ^(i)‖. Take E, E‖Δ^(i)‖ ≤ σ_i/√b. Telescope, divide by ηTα_l:
(1/T)Σ_t Σ_i E‖∇_i f‖ ≤ (f(x1)−f*)/(Tηα_l) + 2α_u‖σ‖_1/(α_l√b) + ηα_u²‖L‖_1/(2α_l).
Pick η = √(2(f1−f*)/(α_u²‖L‖_1 T)), b=T ⇒ (E (1/√h)Σ‖∇_i f‖)² ≤ O((f1−f*)L_avg/T + ‖σ‖_1²/(Th)).
**Key win: L_avg = (1/h)Σ L_i instead of L_∞.**

### LAMB β2>0: update x^(i) ← x^(i) − η φ(‖x^(i)‖) r^(i)/‖r^(i)‖
Same descent lemma; curvature term again ≤ (η²α_u²/2)‖L‖_1 since ‖Δx^(i)‖=ηφ≤ηα_u.
Signal term T_1 = −η Σ_i Σ_j φ ∇_{i,j}f · r_{i,j}/‖r^(i)‖.
Bound ‖r^(i)‖ ≤ √(d_i/(1−β2)) (since |r_j| = |m̂_j|/(√v̂_j+ε) ≤ 1/√(1−β2) coordinatewise when β1=0: m̂=g, v̂=g², r_j=g_j/(|g_j|/√(1−β2))=sign·√(1−β2)... actually |r_j| ≤ 1/√(1−β2)), and √v̂ ≤ G.
Split r into agree/disagree-sign-with-∇f. On agreement coordinates ∇·r/‖r‖ ≥ √((1−β2)/(G²d_i))·∇·g (lower bound the contribution). On disagreement coords bound |contribution| ≤ α_u|∇_{i,j}f| and multiply by P(sign g ≠ sign ∇f) ≤ σ_{i,j}/(√b|∇_{i,j}f|) [signSGD-style]. Get
E[T_1] ≤ −η α_l √(h(1−β2)/(G²d)) ‖∇f‖² + η α_u ‖σ̃‖_1/√b.
Telescope, divide by ηTα_l, η as above ⇒
E‖∇f‖² ≤ O( √(G²d/(h(1−β2))) · [ √(2(f1−f*)‖L‖_1/T) + ‖σ̃‖_1/√T ] ).
### LAMB β2=0: reduces signal coefficient √(1/d_i)·φ·|∇_{i,j}f| ⇒ (E(1/√d)‖∇f‖_1)² ≤ O((f1−f*)L_avg/T + ‖σ̃‖_1²/(Th)).

### Density comparison (signSGD-style)
(Σ_i‖∇_i f‖)² = ψ_g·d‖∇f‖²/h, ‖L‖_1² ≤ ψ_L d²L_∞²/h², ‖σ‖_1²=ψ_σ d‖σ‖²/h. LARS rate becomes O((f1−f*)L_∞/T·ψ_L/ψ_g² + ‖σ‖²/T·ψ_σ²/ψ_g²). Beats SGD when gradient denser than curvature/noise (ψ_L≪ψ_g², ψ_σ≪ψ_g²).

## Design-decision → why
- **Layerwise (not global) adaptation**: layers differ enormously in ‖w‖/‖update‖; single LR can't serve all. Trust ratio gives each layer its own effective step ∝ its weight scale.
- **Adam as base (not momentum) ⇒ LAMB vs LARS**: BERT/attention needs per-coordinate adaptivity (Adam); LARS (momentum base) diverges at 32k on BERT. Two-fold adaptivity: per-dim (Adam) × per-layer (trust ratio).
- **Trust ratio = ‖w‖/‖r+λw‖ (φ=identity clipped)**: φ(z)=z makes update norm ≈ weight norm ⇒ effective step ≈ ‖x‖/‖r‖ ≈ 1/L_i estimate. Clip [γ_l,γ_u] (e.g. clamp ‖w‖ to ≤10) so a huge-weight or tiny-update layer can't get an unbounded step.
- **Weight decay inside the trust ratio (λx in numerator and norm)**: decoupled (AdamW-style) but placed so the normalization sees the full update direction r+λw; keeps decay scale-consistent with the layerwise step.
- **Normalize update to unit norm per layer**: magnitude of Adam update is unreliable across layers; direction is what matters at large batch. Robust to explode/plateau.
- **Debiasing optional**: adam-correction η√(1−β2^t)/(1−β1^t) ≈ LR warmup; can drop if warmup already used.
- **√ LR scaling + linear-epoch warmup across batch sizes**: variance ∝1/b ⇒ √b LR; warmup avoids early large-step instability. Lets batch scale 512→32k untuned.
- **L2 norm for the layer norm**: ablation shows <0.1% difference vs other norms; L2 default.

## Canonical code
- cybertronai/pytorch-lamb `Lamb(Optimizer)` (clean, widely-used; v3 no-debias): exp_avg/exp_avg_sq EMA; adam_step = exp_avg/(sqrt(exp_avg_sq)+eps); +wd*p; weight_norm=clamp(‖p‖,0,10); trust_ratio=weight_norm/adam_norm (=1 if either 0); p -= lr*trust_ratio*adam_step.
- TFA `LAMB`: same with debiasing + exclude_from_weight_decay/layer_adaptation lists. Final code mirrors pytorch-lamb, with debiasing toggle shown.
