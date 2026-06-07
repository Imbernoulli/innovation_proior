# LARS synthesis

## The pain point (derivation-time, 2017)
- Speed up CNN training by adding GPU nodes → data-parallel synchronous SGD → each worker gets a chunk of the global mini-batch. More nodes ⇒ larger global batch B.
- Same #epochs at larger B ⇒ fewer weight updates (T = examples·epochs / B falls linearly). Each step must do more.
- Compensate by raising LR. Linear scaling rule (Krizhevsky 2014): grow B by k ⇒ grow LR by k. Justification: 2 steps of batch B ≈ 1 step of batch 2B with double LR, **assuming ∇L(x_j, w_{t+1}) ≈ ∇L(x_j, w_t)** (weights barely move in one step). Holds early-ish, breaks when steps are big.
- But large LR makes optimization diverge, especially the **initial phase**.
- Warmup (Goyal 2017): start small "safe" LR, ramp to target over a few epochs. With linear scaling + warmup, ResNet-50 to B=8K matched baseline. This is the SOTA recipe.

## What actually breaks (the diagnostic — empirical-first heart)
- Tried linear-scaling + warmup on AlexNet/ImageNet. Scaling stopped at B=2K; for B=4K acc dropped 57.6→53.1%, B=8K → 44.8%. Diverges for LR > 0.06 even with warmup.
- Replaced Local Response Normalization with Batch Normalization (AlexNet-BN). BN allows large LR even without warmup, widens the good-LR range. B=8K gap shrank from 14% to 2.2%. (BN baseline B=512 acc 60.2%.)
- Checked the residual 2.2% gap: is it the Keskar "generalization gap" (sharp minima)? Measured train–test loss gap for B=256 vs B=8K — no significant difference. So it's NOT generalization; it's under-training / optimization.
- DIAGNOSTIC measurement: ratio ‖w‖/‖∇L(w)‖ per layer, AlexNet-BN after 1 iteration. conv1.w = 5.76, fc6.w = 1345. Spans orders of magnitude across layers, and differs between weights and biases.
- Standard SGD uses ONE λ for all layers: w_{t+1}=w_t−λ∇L. When λ large, ‖λ∇L(w)‖ can exceed ‖w‖ → that layer's update overshoots its own weights → divergence. The unsafe layer (smallest ratio) sets the ceiling.
- The ratio is high early, drops after a few epochs. Warmup's real job: start LR small enough that it's safe for ALL layers, then raise it as weights grow. But a single global LR is fundamentally the wrong knob — if it's safe for the worst layer it's too small for the others (they barely move), if it's right for the others it diverges the worst layer.

## The method (LARS)
- Use a separate LOCAL LR λ^l per layer l. Update direction:
  Δw^l_t = γ · λ^l · ∇L(w^l_t),  γ = global LR.
- Local LR via trust coefficient η<1:
  λ^l = η · ‖w^l‖ / ‖∇L(w^l)‖.    (Eq 6)
  η = how much we trust this layer to change in one update.
- Effect: update magnitude ‖Δw^l‖ ≈ γ·η·‖w^l‖ is a fixed FRACTION of the weight norm, INDEPENDENT of gradient magnitude. Kills vanishing/exploding-gradient sensitivity; each layer moves the same relative amount.
- Extend to include weight decay β (so the denominator accounts for the decay term added to the gradient):
  λ^l = η · ‖w^l‖ / (‖∇L(w^l)‖ + β·‖w^l‖).   (Eq 7)
- Contrast with Adam/RMSProp: (1) per-LAYER not per-weight (more stable); (2) update controlled relative to weight norm, not just gradient statistics.
- It's a special case of block-diagonal rescaling (Lafond et al. 2017) — one block per layer.

## Algorithm (with momentum, weight decay, poly decay)
Params: base LR γ_0, momentum m, weight decay β, LARS coeff η, steps T. Init t=0, v=0, w_0^l.
For each step, each layer l:
  g_t^l ← ∇L(w_t^l)
  γ_t ← γ_0 (1 − t/T)^2                          # global poly(2) decay
  λ^l ← ‖w_t^l‖ / (‖g_t^l‖ + β‖w_t^l‖)            # local LR (η folded into γ in box; Eq7 has η)
  v_{t+1}^l ← m·v_t^l + γ_{t+1}·λ^l·(g_t^l + β·w_t^l)   # momentum on the trust-scaled, decayed grad
  w_{t+1}^l ← w_t^l − v_{t+1}^l
Note: the box absorbs η into γ; canonical impls keep η explicit as trust_coeff (η≈0.001). Both equivalent up to where the constant lives.

## Canonical implementation (timm / NVIDIA APEX LARC), code/timm_lars.py
Per parameter tensor p (a "layer"):
  if weight_decay≠0 or always_adapt:
    w_norm = ‖p‖₂ ; g_norm = ‖grad‖₂
    trust_ratio = trust_coeff · w_norm / (g_norm + w_norm·weight_decay + eps)   # = η·‖w‖/(‖g‖+β‖w‖)
    if w_norm==0 or g_norm==0: trust_ratio = 1.0   # safety
    (optional LARC trust_clip: trust_ratio = min(trust_ratio/lr, 1))
    grad ← grad + weight_decay·p        # fold weight decay into grad
    grad ← trust_ratio · grad           # scale by trust ratio
  # then plain SGD+momentum with the GLOBAL lr:
  buf ← momentum·buf + (1−dampening)·grad   (init buf=grad)
  grad ← grad + momentum·buf  if nesterov else buf
  p ← p − lr·grad
This is exactly the algorithm box: lr is the global γ_t (set by scheduler), trust_coeff is η, weight_decay is β.

## Design-decision → why
- Per-layer (not per-weight): per-weight (Adam) destabilizes; layer norm is a robust aggregate; biases/weights and layers differ by 100s× so layer granularity is enough. 
- Ratio ‖w‖/‖grad‖ specifically: makes step a fixed fraction of weight norm → relative, scale-free, safe regardless of grad scale.
- η<1 (≈0.001): the "trust" — how big a fraction of its own weight a layer may move in one step; <1 keeps each step well inside the safe region; small because update is γ·η·‖w‖ and γ is large at large batch.
- +β‖w‖ in denominator: when you fold weight decay β·w into the gradient, the effective update grad is g+β·w; the denominator must measure the norm of what's actually being scaled, so add β‖w‖ for consistency.
- trust_ratio=1 when norms are 0: avoid div-by-zero / undefined adaptation (e.g. zero-init biases).
- Still uses warmup with LARS in experiments: warmup and LARS are complementary; LARS fixes the cross-layer mismatch, warmup smooths the very first steps where ratios are still settling. LARS shrinks how much warmup is needed and widens the safe-LR interval.
- Global poly(2) decay γ_t = γ_0(1−t/T)^2: standard global schedule on top of the per-layer adaptation; the two compose (per-layer trust × global decay).
- LARC trust_clip variant: clip trust_ratio so local LR never exceeds global LR (cap, not floor) — an extra safety used in some impls.

## In-frame discipline
- context.md: LARS/this paper NOT named; scaffold generic large-batch SGD (no trust ratio). Linear scaling (Krizhevsky 2014), warmup (Goyal 2017), SGD-momentum, BN, the per-layer ratio diagnostic all belong (pre-method facts / motivating findings).
- reasoning.md: discovery order, no headers, first person, no naming source paper, derive linear-scaling assumption + the divergence + the per-layer ratio measurement + the trust-ratio update inline.
- answer.md: name LARS, clean algorithm + faithful code, no citation header.
```
