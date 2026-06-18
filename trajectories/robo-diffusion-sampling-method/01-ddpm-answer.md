**Problem.** A diffusion policy is trained and frozen; the only inference freedom is which reverse-process solver runs and over how many steps. The score multiplies D4RL return quality by an NFE penalty that is flat to 10 steps and decays above it. The floor must be the model's own native sampler, run the model's own way, so every faster rung has a faithful quality-and-cost reference to beat.

**Key idea (the floor).** DDPM ancestral sampling over the full denoising chain. The actor is a noise-prediction model whose Gaussian per-step reverse transition is only valid because each forward step is small; the ancestral chain visits every trained level, so the per-step Gaussian assumption stays honest end to end. This is the most faithful unroll of what the loss assumed — the quality ceiling, not a target to beat — at the price of one network call per level.

**Why this start.** No design risk: no deterministic approximation, no skipped levels, no change of parameterization. The injected per-step noise is a non-issue on the full chain because the many remaining steps average it down. It establishes the per-env return ceiling and the wall-clock anchor; the headroom in this task is keeping that quality at a tenth of the steps.

**Hyperparameters.** `solver: ddpm`; `sampling_steps: 100` (equal to `diffusion_steps`). This is the unmodified template — a no-op edit. NFE penalty at 100 steps ≈ exp(−0.015·90) ≈ 0.26, so the floor's score is penalty-capped well below the fast rungs regardless of quality.

**What to watch.** Per-env returns set the quality ceiling; if a faster sampler *beats* DDPM's return, the ancestral noise was hurting and a deterministic sampler is justified on its own merits. Which env's sigmoid quality term is most sensitive flags where a fast sampler's quality loss would cost most in the geometric mean. Wall-clock anchors the speedup the next rung delivers.

```yaml
# EDITABLE region of CleanDiffuser/configs/custom/mujoco/mujoco.yaml — step 1: DDPM, 100 steps
# Default template fill (no-op edit). Only lines 15 and 17 are editable.
diffusion_steps: 100
solver: ddpm          # line 15 — native ancestral sampler
predict_noise: True
sampling_steps: 100   # line 17 — full chain: one network call per trained level
ema_rate: 0.9999
```
