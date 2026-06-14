# CFG-Zero* (Optimized Scale + Zero-Init)

CFG-Zero* is a drop-in classifier-free guidance rule for flow-matching
samplers. It keeps the same conditional and unconditional model calls as CFG, but changes the
per-step guidance update: compute a per-sample projection scale for the unconditional prediction,
and set the velocity to zero for the first `K` ODE steps.

## Problem

A flow-matching model samples by integrating `dx/dt = v_t^theta(x | y)`. Standard CFG uses

```text
v_guided = (1 - w) v_uncond + w v_cond.
```

When the learned velocity is underfitted, this mix amplifies velocity error as well as
conditional signal. The failure is strongest near the source end of the trajectory, where the
sample is still close to noise.

## Core Formulas

For a Gaussian source and target with linear path
`x_t = (1 - t)x_0 + t x_1`, `x_0 ~ N(0, I)`, `x_1 ~ N(mu, I)`, the optimal flow-matching velocity
is

```text
v_t^*(x) = ((2t - 1) / ((1 - t)^2 + t^2)) (x - t mu) + mu.
```

This follows from
`Cov(x_t, x_1 - x_0) = (2t - 1)I`,
`Var(x_t) = ((1 - t)^2 + t^2)I`, and the Gaussian conditional-mean formula for
`E[x_1 - x_0 | x_t = x]`.

The optimized scale is the least-squares projection coefficient:

```text
s* = (v_cond^T v_uncond) / ||v_uncond||^2.
```

It minimizes `||v_cond - s v_uncond||^2`, so `v_cond - s* v_uncond` is the conditional residual
orthogonal to the unconditional prediction.

The guided velocity after the inert prefix is

```text
v = v_uncond * s* + w * (v_cond - v_uncond * s*)
  = (1 - w) s* v_uncond + w v_cond.
```

The zero-init condition comes from the first-step diagnostic in the underfitted regime:

```text
||v_guided(t = 0) - v_0^*||_2^2 >= ||0 - v_0^*||_2^2.
```

At those initial steps, the zero velocity is no farther from the optimal velocity than the guided
prediction, so the sampler uses `v = 0` and leaves `x` unchanged.

## Algorithm

```text
Input: trained velocity model, initial sample x, guidance weight w, zero-init length K.

for each solver step step, t:
    compute v_uncond, v_cond
    s* = (v_cond^T v_uncond) / ||v_uncond||^2
    if step < K:
        v = 0
    else:
        v = v_uncond * s* + w * (v_cond - v_uncond * s*)
    x = ODEStep(x, t, v)

return x
```

`K = 1` is the default in the flow-ODE statement. In the implementation convention
`i <= zero_steps`, `zero_steps = 0` also zeros exactly the first step; the DDIM transplant below
uses `step < K` because `K` is a count of skipped updates.

## Faithful Velocity-Space Code

```python
import torch


def optimized_scale(positive_flat, negative_flat, eps=1e-8):
    dot_product = torch.sum(positive_flat * negative_flat, dim=1, keepdim=True)
    squared_norm = torch.sum(negative_flat ** 2, dim=1, keepdim=True) + eps
    return dot_product / squared_norm


@torch.no_grad()
def sample(pipeline, cond, uncond, guidance_scale, num_steps, zero_steps=0, use_zero_init=True):
    x = pipeline.initialize_sample()

    for i, t in enumerate(pipeline.schedule(num_steps)):
        v_uncond, v_cond = pipeline.predict_velocity(x, t, uncond, cond)

        batch_size = v_cond.shape[0]
        positive_flat = v_cond.view(batch_size, -1)
        negative_flat = v_uncond.view(batch_size, -1)
        alpha = optimized_scale(positive_flat, negative_flat)
        alpha = alpha.view(batch_size, *([1] * (v_cond.dim() - 1))).to(v_cond.dtype)

        if (i <= zero_steps) and use_zero_init:
            v = v_cond * 0.0
        else:
            v = v_uncond * alpha + guidance_scale * (v_cond - v_uncond * alpha)

        x = pipeline.ode_step(x, t, v)

    return x
```

The velocity-space pipeline block uses the same structure with `noise_pred_uncond` and
`noise_pred_text`; the per-step zero branch is the pipeline convention:

```python
noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)

positive_flat = noise_pred_text.view(batch_size, -1)
negative_flat = noise_pred_uncond.view(batch_size, -1)
alpha = optimized_scale(positive_flat, negative_flat)
alpha = alpha.view(batch_size, *([1] * (noise_pred_text.dim() - 1)))
alpha = alpha.to(noise_pred_text.dtype)

if (i <= zero_steps) and use_zero_init:
    noise_pred = noise_pred_text * 0.0
else:
    noise_pred = noise_pred_uncond * alpha + guidance_scale * (
        noise_pred_text - noise_pred_uncond * alpha
    )
```

## DDIM-Style Zero-Init Transplant

For an eps-prediction DDIM or CFG++ sampler, zeroing the initial velocity is implemented by
skipping the initial denoise-and-renoise updates:

```python
import torch


@torch.no_grad()
def sample_ddim_zeroinit(pipeline, prompt, cfg_guidance=7.5, K=2):
    uc, c = pipeline.get_text_embed(null_prompt=prompt[0], prompt=prompt[1])
    zt = pipeline.initialize_latent()

    for step, t in enumerate(pipeline.scheduler.timesteps):
        if step < K:
            continue

        at = pipeline.alpha(t)
        at_prev = pipeline.alpha(t - pipeline.skip)

        eps_uc, eps_c = pipeline.predict_noise(zt, t, uc, c)
        eps_pred = eps_uc + cfg_guidance * (eps_c - eps_uc)

        z0t = (zt - (1 - at).sqrt() * eps_pred) / at.sqrt()
        zt = at_prev.sqrt() * z0t + (1 - at_prev).sqrt() * eps_uc

    return pipeline.decode(zt)
```

This is the same zero-init operation in the noise-prediction frame: `if step < K: continue`.
