# CFG-Zero* (optimized scale + zero-init), distilled

CFG-Zero* is a drop-in classifier-free guidance rule
for flow-matching and diffusion samplers. It keeps the same two model calls per step as CFG but
changes the per-step guidance update in two ways: (a) an **optimized scale** — a per-sample,
per-step scalar on the unconditional prediction, chosen as the least-squares projection coefficient,
so guidance amplifies only the conditional *residual* the unconditional prediction cannot explain;
and (b) **zero-init** — zeroing the velocity (skipping the update) for the first `K` solver steps,
where the underfitted field is least reliable. No retraining, no extra network evaluation.

## Problem it solves

CFG combines the two predictions as `v_guided = (1 - w) v_uncond + w v_cond` with a single global,
hand-tuned scale `w`. When the learned field is inaccurate, the same `w` amplifies prediction error
as readily as conditional signal, and the fixed coefficients ignore the current sample, timestep, and
the agreement between the two predictions. The damage is worst at the high-noise source end, where the
prediction carries the least semantic information.

## Key idea

A Gaussian linear path gives the exact velocity in closed form,

```text
v_t^*(x) = ((2t - 1) / ((1 - t)^2 + t^2)) (x - t mu) + mu,
```

from `Cov(x_t, x_1 - x_0) = (2t - 1)I`, `Var(x_t) = ((1 - t)^2 + t^2)I`, locating where the learned
field's error is worst.

**Optimized scale.** Introduce a scalar `s` on the unconditional prediction,
`v_s = (1 - w) s v_uncond + w v_cond = v_cond + (w - 1)(v_cond - s v_uncond)`. The true loss
`||v_s - v^*||^2` is invisible, but a Young's-inequality bound has only one `s`-dependent term, a
positive constant times `||v_cond - s v_uncond||^2`. Minimizing it is a one-line least squares:

```text
s* = (v_cond^T v_uncond) / ||v_uncond||^2.
```

`s* v_uncond` is the orthogonal projection of `v_cond` onto `v_uncond`; `v_cond - s* v_uncond` is the
residual the unconditional prediction cannot explain — the direction guidance should amplify. The
guided velocity is `s* v_uncond + w (v_cond - s* v_uncond)`. It is per-sample/per-step, costs one dot
product and one norm, and reduces to CFG exactly when the predictions are collinear (`s* = 1`). It is
a new degree of freedom orthogonal to `w`: `w` sets how hard to push along the residual, `s*` sets what
the residual is. (When `v_cond = c v_uncond` are collinear the projection gives `s* = c`, so the mix
reduces to plain CFG, since the residual `v_cond - s* v_uncond = 0`; in particular `s* = 1` when the two
predictions are equal.)

**Zero-init.** The first-step diagnostic in the underfitted regime can satisfy
`||v_guided(t = 0) - v_0^*||^2 >= ||0 - v_0^*||^2` — the guided move is no better than the zero move at
the source end. So zero the velocity (leave `x` unchanged) for the first `K` steps. `K` stays small:
the inequality is about the unreliable source end, not the whole trajectory.

## Algorithm

```text
Input: trained model, initial sample x, guidance weight w, zero-init length K.
for each solver step step, t:
    compute v_uncond, v_cond
    s* = (v_cond^T v_uncond) / ||v_uncond||^2     # per-sample, flattened
    if step < K:
        v = 0
    else:
        v = v_uncond * s* + w * (v_cond - v_uncond * s*)
    x = ODEStep(x, t, v)
return x
```

## Working code (velocity / flow frame)

The per-sample `optimized_scale` (dot-product over squared-norm with an eps floor, flattened
per batch element), and the per-step zero branch:

```python
import torch


def optimized_scale(positive_flat, negative_flat, eps=1e-8):
    dot_product = torch.sum(positive_flat * negative_flat, dim=1, keepdim=True)
    squared_norm = torch.sum(negative_flat ** 2, dim=1, keepdim=True) + eps
    return dot_product / squared_norm


@torch.no_grad()
def sample(pipeline, cond, uncond, guidance_scale, num_steps,
           zero_steps=0, use_zero_init=True):
    x = pipeline.initialize_sample()
    for i, t in enumerate(pipeline.schedule(num_steps)):
        v_uncond, v_cond = pipeline.predict_velocity(x, t, uncond, cond)

        batch_size = v_cond.shape[0]
        positive_flat = v_cond.view(batch_size, -1)
        negative_flat = v_uncond.view(batch_size, -1)
        alpha = optimized_scale(positive_flat, negative_flat)
        alpha = alpha.view(batch_size, *([1] * (v_cond.dim() - 1))).to(v_cond.dtype)

        if (i <= zero_steps) and use_zero_init:
            v = v_cond * 0.0                                  # zero-init prefix
        else:
            v = v_uncond * alpha + guidance_scale * (v_cond - v_uncond * alpha)

        x = pipeline.ode_step(x, t, v)
    return x
```

`v_uncond * alpha + w (v_cond - v_uncond * alpha)` collects to `(1 - w) alpha v_uncond + w v_cond`;
at `alpha = 1` with the zero branch off, this is standard CFG. With `i <= zero_steps`, `zero_steps = 0`
zeros exactly the first step.

## DDIM / CFG++ epsilon-space transplant

For an eps-prediction DDIM/CFG++ sampler, the optimized scale acts on `(noise_uc, noise_c)`, the
renoise stays unconditional (CFG++ manifold step), and zero-init is the `if step < K: continue` prefix:

```python
import torch


@torch.no_grad()
def sample_ddim_cfg_zero_star(pipeline, prompt, cfg_guidance=0.6, K=2):
    uc, c = pipeline.get_text_embed(null_prompt=prompt[0], prompt=prompt[1])
    zt = pipeline.initialize_latent()

    for step, t in enumerate(pipeline.scheduler.timesteps):
        if step < K:                                          # zero-init prefix
            continue

        at = pipeline.alpha(t)
        at_prev = pipeline.alpha(t - pipeline.skip)

        noise_uc, noise_c = pipeline.predict_noise(zt, t, uc, c)

        # optimized scale: s* = <noise_c, noise_uc> / ||noise_uc||^2  (per sample)
        bsz = noise_c.shape[0]
        c_flat = noise_c.reshape(bsz, -1)
        uc_flat = noise_uc.reshape(bsz, -1)
        dot = (c_flat * uc_flat).sum(dim=1, keepdim=True)
        sq_norm = (uc_flat ** 2).sum(dim=1, keepdim=True) + 1e-8
        alpha = (dot / sq_norm).reshape(bsz, *([1] * (noise_c.dim() - 1))).to(noise_c.dtype)

        # mix on the rescaled unconditional baseline
        noise_pred = noise_uc * alpha + cfg_guidance * (noise_c - noise_uc * alpha)

        z0t = (zt - (1 - at).sqrt() * noise_pred) / at.sqrt()         # Tweedie denoise (mixed)
        zt = at_prev.sqrt() * z0t + (1 - at_prev).sqrt() * noise_uc   # CFG++ renoise (uncond)

    return pipeline.decode(z0t)  # return the clean Tweedie estimate, not the renoised latent
```

Cost: the same two predictions per step plus one dot product and one norm; no extra NFE, no retraining.

## Convention note

`K = 1` is the default in the flow-ODE statement (zero exactly the first update); the implementation
convention `i <= zero_steps` with `zero_steps = 0` also zeros the first step, and the DDIM transplant
uses `step < K` because `K` counts skipped updates (`K = 2` in the DDIM/CFG++ transplant). The
optimized scale and zero-init are independent and can be applied separately.
