MeanFlow is a self-contained framework for one-step generative modeling. Instead of learning the
*instantaneous* velocity of a flow (as in flow matching) and then numerically integrating it over
many steps at sampling time, MeanFlow learns the *average* velocity over a time interval directly,
so the whole flow from noise to data can be covered in a single network evaluation. The training
target comes from a closed-form identity between average and instantaneous velocity — the
**MeanFlow Identity** — not from a self-consistency heuristic imposed on the network. The `mse-base`
form trains this with a pure squared-L2 (MSE) loss on the predicted mean velocity.

## Problem it solves

Flow matching / diffusion learn `v(z_t, t)` and sample by integrating `dz_t/dt = v` over a curved
marginal field; high quality needs many function evaluations (NFE) and one step is inaccurate.
MeanFlow targets few-step — especially **1-NFE** — generation, trained from scratch, with no
distillation, no pre-trained teacher, and no discretization curriculum, by making the optimum a
genuine network-independent ground-truth field.

## Key idea

Define the **average velocity** over `[r, t]` as displacement / interval:

```
u(z_t, r, t) = (1 / (t - r)) * integral_{r}^{t} v(z_tau, tau) d tau.
```

Then sampling over any interval is one step, `z_r = z_t - (t - r) u(z_t, r, t)`, and one-step
generation is `z_0 = z_1 - u(z_1, 0, 1)` with `z_1 = eps ~ prior`. No time integral at inference.

Properties straight from the definition:
- **Boundary:** `lim_{r -> t} u = v` (average over a zero interval is the instantaneous velocity).
- **Consistency for free:** by additivity of the integral,
  `(t - r) u(z_t, r, t) = (s - r) u(z_s, r, s) + (t - s) u(z_t, s, t)` for any `s in [r, t]`. A net
  that fits `u` inherits this — no self-consistency loss is required.

## The MeanFlow Identity

`u` is intractable to use directly (it is an integral). Write the displacement form
`(t - r) u = integral_{r}^{t} v d tau` and differentiate in `t` (with `r` independent of `t`):
the product rule on the left gives `u + (t - r) d/dt u`, the fundamental theorem of calculus on
the right gives `v`. Rearranging:

```
u(z_t, r, t) = v(z_t, t) - (t - r) * d/dt u(z_t, r, t).      [MeanFlow Identity]
```

Integral-free, with the instantaneous `v` as the only ground-truth signal. At `r = t` the second
term vanishes and `u = v` — MeanFlow reduces exactly to flow matching.

**Sufficiency (not just necessity).** Differentiating loses an additive constant in general. With
`S = (t - r) u`, `d/dt S = v` gives `S = integral_{r}^{t} v d tau + C`. But `S|_{t=r} = 0` (the
`(t-r)` factor) and `integral_{r}^{t} v|_{t=r} = 0`, so `C = 0` and `S = integral v d tau` exactly.
The boundary is automatic *because* `u` (not the displacement `S`) is the modeled object;
parameterizing `S` directly would require enforcing `S|_{t=r} = 0` by hand.

## Computing the time derivative (JVP)

`d/dt u` is a total derivative along the trajectory. By the chain rule, with `dz_t/dt = v`,
`dr/dt = 0`, `dt/dt = 1`:

```
d/dt u(z_t, r, t) = v(z_t, t) * d_z u + d_t u.
```

This is a Jacobian-vector product of `[d_z u, d_r u, d_t u]` with the tangent `(v, 0, 1)`, computed
by forward-mode autodiff (`torch.func.jvp`, `jax.jvp`) in one extra pass — no Jacobian formed. If
the implementation orders the scalar inputs as `(z, t, r)`, the same derivative is the tangent
`(v, 1, 0)`.

## Training objective

Parameterize `u_theta(z_t, r, t)` and regress it onto the Identity's right-hand side:

```
L(theta) = E || u_theta(z_t, r, t) - sg(u_tgt) ||^2,
u_tgt    = v_t - (t - r) ( v_t * d_z u_theta + d_t u_theta ),
v_t      = eps - x.                                  (conditional velocity, default schedule)
```

- **stop-gradient (`sg`) on `u_tgt`:** the target contains derivatives of `u_theta`, so
  back-propagating through it would mean second-order ("double") backprop through the JVP. Freezing
  the target makes it ordinary first-order regression; it does not move the optimum (at zero loss,
  `u_theta` satisfies the Identity, hence equals the true `u`).
- **conditional `v_t` for marginal `v`:** the Identity uses the intractable marginal
  `v = E[v_t | z_t]`. With the target stopped, the target is affine in the velocity for fixed
  `(z_t, r, t)`, so the conditional-flow-matching expectation argument replaces `v` by the
  per-sample `v_t = eps - x` in both the leading term and the JVP tangent without changing the
  expected gradient.
- **`mse-base` form:** plain squared-L2, `loss = mean(|| u_theta - sg(u_tgt) ||^2)` — no adaptive
  reweighting. (Generalization: a powered loss `|| Delta ||^{2*gamma}` equals MSE weighted by
  `w = 1/(|| Delta ||^2 + c)^p`, `p = 1 - gamma`, with `w` stop-gradiented; `p = 0` is this floor.)

## Design choices

- **Model `u`, not the displacement `S`:** makes the boundary `S|_{t=r}=0` automatic and the
  Identity sufficient; gives the clean sampling rule `z_r = z_t - (t-r)u`.
- **Mix `r = t` and `r != t`:** `r = t` is the flow-matching slice (learn instantaneous velocity,
  an anchor); `r != t` trains the interval field via the correction term.
- **Logit-normal `(r, t)` sampler:** concentrates resolution in the curved mid-times; draw two,
  assign larger to `t`, smaller to `r`.
- **Condition on `(t, t - r)`:** equivalent reparameterization (current time + interval width);
  the JVP is still taken w.r.t. `u_theta(., r, t)`.

## Sampling

```
z_r = z_t - (t - r) * u_theta(z_t, r, t).
one step:  z_0 = z_1 - u_theta(z_1, 0, 1),   z_1 = eps ~ prior.
```

## Working code (PyTorch)

Pure-MSE MeanFlow training step and one-step sampler, faithful to the canonical implementation
(single `jvp`, stop-gradient on the target, conditional velocity `eps - x`):

```python
import torch
from torch.func import jvp


def sample_t_r(batch_size, device, p_mean=-0.4, p_std=1.0, fm_proportion=0.75):
    """Logit-normal (t, r); larger -> t, smaller -> r; fm_proportion keeps r = t.
    r = t is the flow-matching slice; r != t trains interval behavior."""
    def logit_normal(n):
        return torch.sigmoid(torch.randn(n, device=device) * p_std + p_mean)
    t = logit_normal(batch_size)
    r = logit_normal(batch_size)
    t, r = torch.maximum(t, r), torch.minimum(t, r)          # orient interval: t >= r
    use_fm = torch.rand(batch_size, device=device) < fm_proportion
    r = torch.where(use_fm, t, r)                            # r = t gives the FM slice
    return t.view(-1, 1, 1, 1), r.view(-1, 1, 1, 1)


def meanflow_loss(net, x):
    """MeanFlow training step, pure squared-L2 objective (the mse-base floor)."""
    B, device = x.shape[0], x.device
    t, r = sample_t_r(B, device)

    eps = torch.randn_like(x)
    z_t = (1 - t) * x + t * eps                              # conditional path
    v = eps - x                                              # conditional velocity v_t

    def u_fn(z, t_, r_):
        return net(z, t_, t_ - r_)                          # network conditioned on (t, t - r)

    # JVP: u_theta and d/dt u = v * d_z u + d_t u  via tangent (v, 1, 0) on (z, t, r)
    primals = (z_t, t, r)
    tangents = (v, torch.ones_like(t), torch.zeros_like(r))
    u, dudt = jvp(u_fn, primals, tangents)                  # one forward-mode pass

    interval = (t - r).clamp(0.0, 1.0)                      # no-op for valid sampled times
    u_tgt = (v - interval * dudt).detach()                  # MeanFlow Identity, stop-gradient
    return (u - u_tgt).flatten(1).pow(2).mean(dim=1).mean() # pure MSE on mean velocity


@torch.no_grad()
def sample_one_step(net, shape, device):
    """1-NFE generation: z_0 = z_1 - u_theta(z_1, r=0, t=1)."""
    z = torch.randn(shape, device=device)
    t = torch.ones(shape[0], 1, 1, 1, device=device)
    r = torch.zeros(shape[0], 1, 1, 1, device=device)
    interval = t - r
    u = net(z, t, interval)
    return z - interval * u
```

The only machinery beyond ordinary flow matching is the single `jvp` call and the stop-gradient on
its product; setting `r = t` everywhere recovers flow matching exactly.
