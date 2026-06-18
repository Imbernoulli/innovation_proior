**Problem.** DDPM at 100 steps is penalty-capped (NFE penalty ≈0.26) *and* leaves return on the table (hopper 0.784, halfcheetah 0.442) — the most faithful unroll underperforms, which points at its injected per-step noise. The next rung must cut steps and remove that noise without retraining the frozen policy.

**Key idea.** DDIM: the deterministic (σ=0) member of the marginal-preserving non-Markovian inference family. The denoising loss only ever constrained the marginals q(a_t|a_0), not the joint, so any joint with the same marginals is solved by the same frozen network. The σ=0 reverse update,
`a_{t-1} = √ᾱ_{t-1}·f_θ(a_t) + √(1−ᾱ_{t-1})·ε_θ(a_t)` with `f_θ(a_t)=(a_t−√(1−ᾱ_t)ε_θ)/√ᾱ_t`,
is a deterministic pushforward of the prior; and since the objective is blind to chain length, it runs on a short sub-sequence — 20 steps instead of 100.

**Why it wins twice.** Speed: 20 steps is a 5× cut at no retraining cost. Quality: with no injected noise there is nothing for a short chain to clean up, so cutting steps only coarsens a smooth map; and every one of the 50 candidate actions now sits at the model's mode rather than scattered by ancestral noise, so the critic's softmax-over-Q re-weighting filters a tighter set — expected to *raise* return where DDPM's was lowest.

**Why DDIM specifically.** It is the exact σ=0 limit of the family — the most conservative deterministic sampler, one network call per step, schedule carry in closed form, no order/derivative estimates. It adds the fewest new approximations on top of the two changes (stochastic→deterministic, 100→20).

**Hyperparameters.** `solver: ddim` (CleanDiffuser's deterministic σ=0 DDIM); `sampling_steps: 20`. NFE penalty at 20 ≈ exp(−0.015·10) ≈ 0.861 — a 3.3× multiplier over the floor before quality.

**What to watch.** Expect returns to hold or rise (hopper toward ~1.0, halfcheetah up from 0.442). A meaningful drop in any env means 20 steps is too few for that env's action geometry — forcing a higher-order solver that holds quality at the same budget (next rung).

```yaml
# EDITABLE region of CleanDiffuser/configs/custom/mujoco/mujoco.yaml — step 2: DDIM, 20 steps
diffusion_steps: 100
solver: ddim          # line 15 — deterministic (sigma=0) marginal-preserving sampler
predict_noise: True
sampling_steps: 20    # line 17 — 5x fewer steps than the DDPM floor
ema_rate: 0.9999
```
