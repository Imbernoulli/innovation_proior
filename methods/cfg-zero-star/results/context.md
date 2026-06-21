# Context: classifier-free guidance for flow-matching and diffusion samplers

## Research question

Flow-matching and diffusion image/video generators sample by integrating a learned velocity (or
score) field from a source distribution to the data distribution. A frozen model is queried twice
per step — once with the condition and once with it dropped — yielding `v_cond` and `v_uncond`, and
the sampler integrates a linear mix of the two. Classifier-free guidance (CFG) is the standard mix.
The question is how to set the per-step guidance update from the two available predictions `v_cond`
and `v_uncond`, as a drop-in rule that keeps the same two model calls per step — no retraining, no
extra network evaluations.

## Background

Conditional flow matching uses a linear path `x_t = (1 - t) x_0 + t x_1` with `x_0 ~ p(x|y)`,
`x_1 ~ q(x|y)`, and regresses the velocity to the displacement, so the population minimizer is the
conditional mean `v_t^*(x) = E[x_1 - x_0 | x_t = x]`. On real data `v^*` is unknown, but for a
Gaussian source and target (`p = N(0,I)`, `q = N(mu, I)`) it is closed form:

```text
v_t^*(x) = ((2t - 1) / ((1 - t)^2 + t^2)) (x - t mu) + mu,
```

from the Gaussian conditional-mean identity with `Cov(x_t, x_1 - x_0) = (2t - 1)I` and
`Var(x_t) = ((1 - t)^2 + t^2)I`. This gives a setting where a learned velocity can be compared to the
optimal one across timesteps. Diffusion DDIM/CFG++ samplers in the noise-prediction (epsilon) frame
are the same object under a change of variables: a step is "Tweedie-denoise to the clean manifold,
then renoise."

## Baselines

**Classifier-free guidance, CFG (Ho & Salimans, 2022).** Combine the two predictions as
`v_guided = (1 - w) v_uncond + w v_cond`, with `w` a single global hand-tuned scalar.

**Classifier guidance (Dhariwal & Nichol, 2021).** Adds a separately scaled classifier gradient to
an unconditional score, using an extra classifier on noisy inputs, so the baseline direction and the
condition-improving direction are set by separate scalars rather than locked to one.

**Manifold/reverse-step fixes (CFG++ — Chung et al. 2024/2025; adaptive, characteristic, rectified
CFG).** Change *how guidance is injected into the reverse step* to reduce oversaturation and
off-manifold drift at high scale (CFG++ renoises with the unconditional prediction).

**Guidance-interval heuristics (Kynkäänniemi et al. 2024).** Restrict guidance to a middle interval
of the schedule, applying guidance over part of the noise schedule rather than the whole trajectory.

## Evaluation settings

The generator is frozen; only the per-step guidance computation changes. Diagnostics on a Gaussian
flow-matching toy problem where `v^*` is closed form; class-conditional ImageNet-256 with a DiT/SiT
flow model at a fixed solver/step budget (IS, FID, sFID, precision, recall); text-to-image with large
frozen flow models (SD3/3.5, Lumina-Next, Flux) at fixed prompts/seeds/solvers/scales; text-to-video
(Wan-style) at fixed prompts/seeds. The transplant target here is an eps-prediction DDIM/CFG++
text-to-image sampler scored by FID.

## Code framework

A guidance rule plugs into the ordinary sampling loop; the frozen model gives both predictions per
step within a fixed evaluation budget. The open slot is the per-step velocity/noise update.

```python
import torch


@torch.no_grad()
def sample(pipeline, cond, uncond, guidance_scale, num_steps):
    x = pipeline.initialize_sample()
    for step, t in enumerate(pipeline.schedule(num_steps)):
        v_uncond, v_cond = pipeline.predict_velocity(x, t, uncond, cond)
        # TODO: fill in the per-step update (the guidance rule).
        v = ...
        x = pipeline.ode_step(x, t, v)
    return x
```
