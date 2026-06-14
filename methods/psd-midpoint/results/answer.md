# Progressive Self-Distillation (PSD), midpoint variant — distilled

PSD trains a two-time **flow map** `X_{s,t}` for one-/few-step generation directly from data,
with no pre-trained teacher, by enforcing the **semigroup** (composition) property of the map:
a big jump must equal two smaller jumps composed along the same trajectory. The **midpoint**
variant fixes the intermediate split at the center of the interval, `u = (s+t)/2`, so the
training target is the even average of the two equal half-jumps. It pairs this off-diagonal
self-consistency term with a diagonal flow-matching term, uses a stopgradient teacher to keep
the signal flowing from the anchored diagonal outward, and works entirely in velocity-space so
the loss is preconditioned (no dependence on gap size) and needs no spatial Jacobian.

## Problem it solves

Flow/diffusion models sample by integrating `dot x_t = b_t(x_t)`, costing many network
evaluations. Learn the integrated map `X_{s,t}` instead — one application `X_{0,1}(x_0)` is a
sample — trained as directly and stably as flow matching, without a separate teacher/distill
phase.

## Key objects

- **Interpolant:** `I_t = alpha_t x_0 + beta_t x_1` (linear: `alpha_t = 1-t`, `beta_t = t`),
  with drift `b_t(x) = E[dot I_t | I_t = x]`.
- **Flow map / parameterization:** `X_{s,t}(x) = x + (t-s) v_{s,t}(x)`. This enforces
  `X_{t,t}(x) = x`, and the **tangent condition** `lim_{s->t} partial_t X_{s,t}(x) = b_t(x)`
  becomes `v_{t,t}(x) = b_t(x)` — the network `v` on the diagonal *is* the flow-matching
  velocity. Off the diagonal, `v_{s,t}(x)` is the chord slope between `x_s` and `x_t`.
- **Semigroup property:** `X_{u,t}(X_{s,u}(x)) = X_{s,t}(x)` for any `u in [s,t]`.

## The objective

Diagonal flow-matching term plus off-diagonal semigroup-residual term, each with an EDM2-style
learned `(s,t)` uncertainty weight `w` (Gaussian NLL form `exp(-w)·loss + w`):

```
L_b^t   = exp(-w_{t,t}) |v_{t,t}(I_t) - dot I_t|^2 + w_{t,t}                         (s = t)
L_psd^{s,t} = exp(-w_{s,t}) |v_{s,t}(I_s) - sg[teacher]|^2 + w_{s,t}                 (s < t)
```

The off-diagonal residual is taken in **velocity-space**: expanding the composition under
`X = x + (t-s)v` gives
`v_{s,t}(x) = ((u-s)/(t-s)) v_{s,u}(x) + ((t-u)/(t-s)) v_{u,t}(X_{s,u}(x))`. With
`u = gamma s + (1-gamma) t` the coefficients are `(1-gamma, gamma)`, so

```
teacher = (1 - gamma) v_{s,u}(I_s) + gamma v_{u,t}(X_{s,u}(I_s)),   X_{s,u}(I_s) = I_s + (u-s) v_{s,u}(I_s)
```

- **PSD-uniform:** `gamma ~ U[0,1]` (the code's `h`), length-weighted mix, swept node.
- **PSD-midpoint:** `gamma = 1/2`, `u = (s+t)/2` deterministically, so `teacher = 1/2 (v_{s,u} + v_{u,t}(X_{s,u}))` — equal half-jumps, even average, and a lower-variance deterministic target (the continuous-time analogue of the dyadic shortcut-model bootstrap).

Working in `v`-space divides out the `(t-s)^2` factor that the `X`-space residual carries, so the
effective learning rate no longer depends on the gap. The `sg[·]` (stopgradient) on the teacher
makes the composed two-jump term a frozen target (instantaneous-parameter teacher, EMA factor
`delta = 0`) and the full jump `v_{s,t}` the student, so the externally-anchored diagonal truth
flows outward into the off-diagonal rather than being corrupted by it.

## Correctness and the known caveat

With a full-support proposal over the intermediate point `u`, minimizing `L_b + L_psd` has the
flow map as its **unique** minimizer: the true map zeroes the semigroup residual and minimizes
`L_b`, achieving the lower bound `L_b(b)`; conversely any global minimizer (with continuous `v`)
satisfies the semigroup property and `v_{t,t} = b`, which forces it to be the flow map. The
midpoint branch uses the deterministic proposal `gamma = 1/2`; the true map still zeroes this
exact residual, but the full-support uniqueness proof does not automatically apply to that
single-split objective. Unlike the Lagrangian/Eulerian residuals, **no Wasserstein bound** relates
PSD's loss value to one-step accuracy — bootstrapping small steps into large ones compounds error
and shifts the input distribution, a structural difficulty of self-consistent bootstrap schemes.

## Training configuration

- Batch split `eta = 0.75` diagonal (flow matching) / `0.25` off-diagonal (PSD). Off-diagonal
  `(s,t)` drawn uniformly on the upper triangle `s < t`; midpoint `u = (s+t)/2`.
- Teacher params = current params (`delta = 0`). Cost: 3 network evals per off-diagonal sample
  (`v_{s,t}`, `v_{s,u}`, `v_{u,t}`), 1 per diagonal sample.
- Network: EDM2-style U-Net taking `(s, dt = t-s)`, `sigma_data` input/output preconditioning,
  a `logvar` head for `w_{s,t}`; square-root LR schedule, gradient clipping, EMA for evaluation.
  Benchmark: unconditional CIFAR-10 (linear interpolant, Gaussian base), FID at 1/2/4/8/16 steps.

## Working code (JAX/Flax loss implementation)

`X.apply(..., return_X_and_phi=True) -> (X_{s,t}(x), v_{s,t}(x))`; `method="calc_b"` returns
`v_{t,t}`; `method="calc_weight"` returns `w_{s,t}`. The PSD term is the off-diagonal slot; the
diagonal term is the flow-matching slot.

```python
import jax
import jax.numpy as jnp


def diagonal_term(params, x0, x1, label, t, rng, *, interp, X):
    """Diagonal (interpolant) term: v_{t,t}(I_t) must match dot I_t."""
    It = interp.calc_It(t, x0, x1)
    It_dot = interp.calc_It_dot(t, x0, x1)
    bt = X.apply(params, t, It, label, train=True, method="calc_b", rngs=rng)   # v_{t,t}(I_t)
    velocity_loss = jnp.sum((bt - It_dot) ** 2)
    weight_tt = X.apply(params, t, t, method="calc_weight")                     # w_{t,t}
    return jnp.exp(-weight_tt) * velocity_loss + weight_tt


def psd_term(params, teacher_params, x0, x1, label, s, t, u, h, rng,
             *, interp, X, psd_type, stopgrad_type):
    """Progressive Self-Distillation term (midpoint when psd_type == 'midpoint')."""
    Is = interp.calc_It(s, x0, x1)

    # full jump (student): X_{s,t}(I_s) and its slope v_{s,t}(I_s)
    X_st, phi_st = X.apply(
        params, s, t, Is, label, train=False, rngs=rng, return_X_and_phi=True
    )

    # two composed sub-jumps (teacher)
    if stopgrad_type == "convex":
        X_su, phi_su = jax.lax.stop_gradient(
            X.apply(teacher_params, s, u, Is, label,
                    train=False, rngs=rng, return_X_and_phi=True)
        )
        X_ut, phi_ut = jax.lax.stop_gradient(
            X.apply(teacher_params, u, t, X_su, label,           # evaluated at X_{s,u}(I_s)
                    train=False, rngs=rng, return_X_and_phi=True)
        )
    elif stopgrad_type == "none":
        X_su, phi_su = X.apply(params, s, u, Is, label,
                               train=False, rngs=rng, return_X_and_phi=True)
        X_ut, phi_ut = X.apply(params, u, t, X_su, label,
                               train=False, rngs=rng, return_X_and_phi=True)
    else:
        raise ValueError(f"Invalid stopgrad_type: {stopgrad_type}")

    if psd_type == "uniform":
        student = phi_st
        teacher = (1 - h) * phi_su + h * phi_ut                  # h is gamma; u = h*s + (1-h)*t
    elif psd_type == "midpoint":
        student = phi_st
        teacher = 0.5 * (phi_su + phi_ut)                        # h unused; sampler sets u = 0.5*(s+t)
    else:
        raise ValueError(f"Invalid psd_type: {psd_type}")

    psd_loss = jnp.sum((student - teacher) ** 2)                 # v-space: (t-s)^2 divided out
    weight_st = X.apply(params, s, t, method="calc_weight")      # w_{s,t}
    return jnp.exp(-weight_st) * psd_loss + weight_st
```

The intermediate time is supplied as the midpoint `u = 0.5 * (s + t)` for the midpoint variant
(`h` unused); `(s,t)` are sampled by drawing two uniform times and ordering them; the teacher
parameters are the current parameters. The full loss averages `diagonal_term` over the `eta`
fraction of the batch on `s = t` and `psd_term` over the `1 - eta` fraction on `s < t`.
