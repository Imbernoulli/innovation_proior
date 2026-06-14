# Lagrangian Self-Distillation (LSD), distilled

LSD is a single-phase, teacher-free training objective for a **two-time flow map**
`X_{s,t}` — the integrated map that jumps a point along a probability-flow ODE trajectory from
time `s` to time `t`, enabling one- or few-step generation instead of a many-step ODE solve. A
single network output `v_{s,t}(x)` parameterizes the map as `X_{s,t}(x) = x + (t-s) v_{s,t}(x)`;
its diagonal `v_{t,t}` is the velocity field `b_t`, trained by ordinary flow matching, and its
off-diagonal is trained to satisfy the **Lagrangian flow-map relation** by self-distillation,
using the diagonal as an implicit teacher. LSD is chosen over the Eulerian and progressive
variants because its residual contains no spatial Jacobian and no composition of learned jumps —
the two operations that destabilize the alternatives.

## Problem it solves

Flows/diffusions sample by integrating `xdot_t = b_t(x_t)` from noise to data, costing tens to
hundreds of network evaluations. The goal is to learn the integrated jump map `X_{s,t}` so a
single call `X_{0,1}(x_0)` produces a sample (compose a few for higher quality) — directly, from
scratch, without a pre-trained teacher and without the optimization instability of prior
direct-training schemes.

## Key idea

**Stochastic interpolant / flow.** `I_t = alpha_t x_0 + beta_t x_1`; the probability flow
`xdot_t = b_t(x_t)` has `b_t(x) = E[Idot_t | I_t = x]`, learned by
`L_b(vhat) = int_0^1 E |vhat_{t,t}(I_t) - Idot_t|^2 dt`.

**The map and its structural identities.** The flow map satisfies `X_{s,t}(x_s) = x_t`.
Differentiating this in `t`, in `s`, and composing gives three equivalent characterizations the
true map must obey:

```
Lagrangian:  partial_t X_{s,t}(x) = b_t(X_{s,t}(x))
Eulerian:    partial_s X_{s,t}(x) + grad X_{s,t}(x) . b_s(x) = 0
semigroup:   X_{u,t}(X_{s,u}(x)) = X_{s,t}(x)
```

**Tangent condition.** Pushing the Lagrangian identity to `s -> t` and using `X_{t,t}(x)=x`:

```
lim_{s -> t} partial_t X_{s,t}(x) = b_t(x).
```

The velocity is the map's diagonal time-derivative — one network encodes both.

**Parameterization.** `X_{s,t}(x) = x + (t-s) v_{s,t}(x)` (a shift-rescale of `X`, no loss of
expressivity). It makes `X_{s,s}(x) = x` exact and yields `lim_{s->t} partial_t X = v_{t,t}`, so

```
v_{t,t}(x) = b_t(x).
```

The diagonal trains by flow matching; the off-diagonal is distilled from it.

**Self-distillation objective.** `L_sd = L_b + L_dist`, where `L_dist` is the squared Lagrangian
residual:

```
L_LSD(vhat) = int_0^1 int_0^t E | vhat_{t,t}(Xhat_{s,t}(I_s)) - partial_t Xhat_{s,t}(I_s) |^2 ds dt.
```

The minimizer is the true flow map and it is unique: `L_b >= L_b(b)` (convex, min at the
conditional mean `b`) and `L_LSD >= 0`, so `L_sd >= L_b(b)`; the true map attains the bound, and
any minimizer has the correct diagonal and zero residual, which by one-sided-Lipschitz ODE
uniqueness (plus continuity) forces it to be the flow map.

## Why Lagrangian (vs. Eulerian and Progressive)

All three characterizations share the same minimizer, so the choice is about optimization:

- **ESD (Eulerian)** residual contains `grad X` — the **spatial Jacobian** of the map; its
  parameter gradient backpropagates through a spatial JVP of a large image network, which is
  empirically unstable.
- **PSD (progressive / semigroup)** target composes two learned jumps `Xhat_{u,t}(Xhat_{s,u})` —
  **bootstrapping** small steps into large ones: errors compound and the target's input
  distribution shifts; no Wasserstein guarantee.
- **LSD (Lagrangian)** mismatch `v_{t,t}(X_{s,t}) - partial_t X_{s,t}` has only a cheap
  **time**-derivative (1-D `jvp`, ~1.5x forward, returns `X` for free) and a single teacher call
  — no spatial Jacobian, no bootstrapping. Cost `(1 + C)B` with `C ~ 1.5`.

## Stopgradient (teacher placement)

Both residual terms share parameters; to keep information flowing from the diagonal (which has
the external `Idot` signal) to the off-diagonal, the teacher is stop-gradiented, emulating the
frozen teacher of distillation:

```
r = sg[ vhat_{t,t}(Xhat_{s,t}(I_s)) ] - partial_t Xhat_{s,t}(I_s).
```

For high-dimensional images, the transported point is also stop-gradiented before the teacher
call (`vhat_{t,t}(sg[Xhat_{s,t}(I_s)])`) so no spatial Jacobian leaks back through the argument
(the `convex` configuration). The implementation uses instantaneous teacher parameters
`phi = theta` (`delta = 0`); EMA parameters are kept for sampling/evaluation rather than for this
teacher.

## Adaptive weight and sampling

Each `(s,t)` is weighted by a learned scalar `w_{s,t}` (EDM2 uncertainty weighting, generalized
to two times): the per-term loss is `e^{-w_{s,t}} |r|^2 + w_{s,t}`. Minimizing over `w` gives
`w = log L`, so the model receives the scale-normalized gradient `∇L / L`, equalizing the
heterogeneous gradient norms across time pairs and allowing larger learning rates. Times are
drawn from `p_{s,t} = eta U_diag + (1-eta) U_offtriangle` with `eta = 0.75`: most of the (cheap)
budget on flow matching, the rest on (expensive) distillation; `eta` doubles as a compute knob.

## Guarantee

If `L_b(vhat) + L_LSD(vhat) <= eps`, with `Lhat` the spatial Lipschitz constant of `vhat_{t,t}`,
then the one-step generated distribution `rhohat_1 = Xhat_{0,1} # rho_0` satisfies

```
W_2^2(rhohat_1, rho_1) <= 4 e^{1 + 2 Lhat} eps.
```

Sketch: `L_b <= eps` implies `int E |vhat_{t,t} - b_t|^2 dt <= eps` (the leftover term is a
conditional variance, `>= 0`, by the tower property); a flow-matching Grönwall bound gives
`W_2^2(rho_1, rhohat^v_1) <= e^{1+2Lhat} eps` for the flow of `vhat_{t,t}`; the Lagrangian
flow-map-matching bound gives `W_2^2(rhohat^v_1, rhohat_1) <= e^{1+2Lhat} eps`; triangle +
`(a+b)^2 <= 2a^2 + 2b^2` combine to `4 e^{1+2Lhat} eps`. (ESD has an analogous
`2e(1 + e^{2Lhat}) eps`; PSD has none — consistent with its compounding error.)

## Implementation details

Linear interpolant `alpha_t = 1-t`, `beta_t = t` (so `Idot_t = x_1 - x_0`), Gaussian base scaled
to the data variance. EDM2 UNet with preconditioning `v_{s,t}(x) = sigma_data * UNet(x/sigma_data,
s, t-s)` — note it embeds `s` and the gap `dt = t-s`, with positional (not Fourier) time
embeddings and FiLM conditioning. RAdam, warmup then sqrt LR decay, gradient clipping at 1.0,
parameter EMA for sampling, trained from random init with no pre-training.

## Working code

The loss terms (JAX/Flax), filling the off-diagonal slot of the interpolant harness. The
flow-map module exposes `method="calc_b"` (`v_{t,t}`), `method="partial_t"` (`(X_{s,t},
partial_t X_{s,t})` via `jvp` in `t`), and `method="calc_weight"` (`w_{s,t}`):

```python
import jax
import jax.numpy as jnp


def diagonal_term(params, x0, x1, label, t, rng, *, interp, X):
    """Flow matching on the diagonal s = t: v_{t,t} = b_t, regressed onto Idot."""
    It = interp.calc_It(t, x0, x1)              # I_t = alpha_t x0 + beta_t x1
    It_dot = interp.calc_It_dot(t, x0, x1)      # Idot_t = alpha_dot_t x0 + beta_dot_t x1
    bt = X.apply(params, t, It, label, train=True, method="calc_b", rngs=rng)  # v_{t,t}(I_t)
    velocity_loss = jnp.sum((bt - It_dot) ** 2)
    weight_tt = X.apply(params, t, t, method="calc_weight")          # w_{t,t}
    return jnp.exp(-weight_tt) * velocity_loss + weight_tt           # e^{-w} L + w


def lsd_term(params, teacher_params, x0, x1, label, s, t, rng, *, interp, X, stopgrad_type):
    """Lagrangian self-distillation mismatch on the off-diagonal s < t."""
    Is = interp.calc_It(s, x0, x1)              # start point at time s

    # one jvp in t returns BOTH X_{s,t}(I_s) and partial_t X_{s,t}(I_s)
    Xst_Is, dt_Xst = X.apply(
        params, s, t, Is, label, train=False, method="partial_t", rngs=rng
    )

    if stopgrad_type == "convex":
        # frozen-teacher emulation for images: stop gradient through the transported point
        # (no spatial Jacobian) and through the teacher velocity (info flows diag -> offdiag)
        Xst_Is = jax.lax.stop_gradient(Xst_Is)
        b_eval = jax.lax.stop_gradient(
            X.apply(teacher_params, t, Xst_Is, label, train=False, method="calc_b", rngs=rng)
        )
    elif stopgrad_type == "none":
        # full gradient (low-dimensional / small-net regime)
        b_eval = X.apply(params, t, Xst_Is, label, train=False, method="calc_b", rngs=rng)
    else:
        raise ValueError(f"Invalid stopgrad_type: {stopgrad_type}")

    weight_st = X.apply(params, s, t, method="calc_weight")          # w_{s,t}
    error = b_eval - dt_Xst                     # v_{t,t}(X_{s,t}(I_s)) - partial_t X_{s,t}(I_s)
    lsd_loss = jnp.sum(error ** 2)
    return jnp.exp(-weight_st) * lsd_loss + weight_st               # e^{-w} |r|^2 + w
```

The training driver vmaps these over the batch, routes a fraction `eta = 0.75` of samples through
`diagonal_term` (diagonal `s=t`) and the rest through `lsd_term` on uniformly-sampled upper-
triangle pairs `(s,t)`, sums the two contributions, and takes one gradient step on both the model
`v` and the weight `w`. Sampling composes `X_{t_i, t_{i+1}}` over a uniform time grid for the
chosen NFE.
