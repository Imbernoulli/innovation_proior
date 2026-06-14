# Progressive Self-Distillation (PSD), Uniform Variant

Progressive Self-Distillation (PSD) is a teacher-free, single-model objective for training the
two-time **flow map** `X_{s,t}` of a probability-flow ODE, so that samples can be generated in
one (or a few) network evaluations instead of a full ODE solve. It learns the map directly from
scratch by pairing flow matching on the diagonal `s=t` with a *semigroup* (composition) residual
on the off-diagonal `s≠t`, using the model as its own teacher. The **uniform** variant samples
the intermediate split point uniformly, supervising self-consistency under every way of breaking
a jump into two.

## Problem it solves

Flow/diffusion samplers integrate `x_dot = b_t(x)` from noise to data, costing tens-to-hundreds
of sequential network evaluations. The flow map `X_{s,t}` (with `X_{s,t}(x_s) = x_t` along
trajectories) takes a finite jump in one call; `X_{0,1}` generates in one step. The goal is to
train `X_{s,t}` as cheaply and stably as ordinary flow matching — one network, no pretrained
teacher, a provably correct minimizer, and dynamics stable on image-scale UNets.

## Key idea

Parameterize the map to enforce the identity on the diagonal and expose the implicit velocity:

```
X_{s,t}(x) = x + (t - s) v_{s,t}(x)   =>   X_{s,s}(x) = x,   lim_{s->t} d_t X_{s,t}(x) = v_{t,t}(x) = b_t(x).
```

So `v_{t,t}` is the drift, trainable by flow matching, and only the off-diagonal `v_{s,t}` needs
a target. Use the semigroup identity `X_{u,t}(X_{s,u}(x)) = X_{s,t}(x)`. Substituting the
parameterization, the constant `x` cancels and (with `u = gamma s + (1-gamma) t`,
`gamma in [0,1]`) the composition reads off the slopes as a length-weighted convex combination:

```
v_{s,t}(x) = (1 - gamma) v_{s,u}(x) + gamma v_{u,t}( X_{s,u}(x) ),    u = gamma s + (1-gamma) t.
```

Reading the residual off the slopes (not the points) removes a `(t-s)^2` loss-scale factor that
would otherwise make the effective learning rate depend on the sampled jump size. Regress the
student `v_{s,t}` onto a **stopgradded** version of the right-hand combination so information
flows from the externally-supervised diagonal to the off-diagonal (and to avoid backprop through
the bootstrapped inner map). **PSD-uniform**: `gamma ~ U([0,1])`. (PSD-midpoint:
`gamma = 1/2`.)

## Final objective

`L_sd = L_b + L_psd`, each term wrapped in the EDM2-style learned adaptive weight
`e^{-w_{s,t}} L + w_{s,t}` (for positive `L`, minimizing over `w` gives `w* = log L`,
value `1 + log L`, equalizing loss scale across `(s,t)` and enabling larger learning rates):

```
L_b   = E_t E_{x0,x1} [ e^{-w_{t,t}} | v_{t,t}(I_t) - I_dot_t |^2 + w_{t,t} ]            (diagonal s=t)
L_psd = E_{(s,t)} E_gamma E_{x0,x1} [ e^{-w_{s,t}} | v_{s,t}(I_s)
          - sg{ (1-gamma) v_{s,u}(I_s) + gamma v_{u,t}( X_{s,u}(I_s) ) } |^2 + w_{s,t} ]  (off-diagonal)
```

with `I_t = alpha_t x0 + beta_t x1` (linear: `alpha_t=1-t, beta_t=t`, so `I_dot_t = x1 - x0`),
`u = gamma s + (1-gamma) t`, `gamma ~ U([0,1])`. A fraction `eta` of each batch is drawn on the
diagonal (`t ~ U([0,1])`, `s=t`) and `1-eta` in the upper triangle `s<t` (two uniforms, take min
and max); `eta = 0.75`. The teacher uses the current (instantaneous) parameters with the
gradient stopped; `v` and `w` are trained jointly.

**Correctness.** For the unweighted objective, or for fixed positive weights, `L_sd >= L_b(b)`;
the true flow map attains this lower bound (semigroup => `L_psd = 0`, diagonal =>
`L_b = L_b(b)`). Conversely, any global minimizer has `L_psd = 0` and the correct diagonal, and
assuming `v` is continuous (which rules out the trivial discontinuous `X=id` solution), it must
be the flow map. The target is correct for any full-support proposal over `gamma`; the learned
`w` head is a practical loss-scale normalizer, not a separate flow-map target.

## Training settings (CIFAR-10, unconditional)

Linear interpolant, Gaussian base with adaptive variance; EDM2 config-G UNet (128 base
channels, multipliers `[2,2,2]`, 4 residual blocks/resolution, attention at `16x16`, dropout
0.13, positional time embeddings, embed `s` and `(t-s)`); batch size 512; 204.8M training samples
(400,000 optimizer steps); RAdam, gradient clipping 1.0, square-root learning-rate decay after
35,000 steps; EMA of parameters for sampling; online FID over 10,000 generated samples and
FID over 50,000 generated samples at `NFE in {1,2,4,8,16}`.

## Working code

JAX/Flax implementation structure. The network wrapper exposes `calc_b(t,x) =
v_{t,t}(x)` (i.e. `calc_phi(t,t,x)`) and `apply(..., return_X_and_phi=True)` returning both
`X_{s,t}(x) = x + (t-s) v_{s,t}(x)` and the slope `v_{s,t}(x)`; `calc_weight(s,t) = w_{s,t}`.

```python
import jax
import jax.numpy as jnp


def diagonal_term(params, x0, x1, label, t, rng, *, interp, X):
    """Diagonal (s=t) flow-matching loss: v_{t,t} = b_t = E[I_dot_t | I_t]."""
    It = interp.calc_It(t, x0, x1)                       # I_t = alpha_t x0 + beta_t x1
    It_dot = interp.calc_It_dot(t, x0, x1)               # I_dot_t (= x1 - x0 for linear)
    bt = X.apply(params, t, It, label, train=True, method="calc_b", rngs=rng)  # v_{t,t}(I_t)
    velocity_loss = jnp.sum((bt - It_dot) ** 2)
    weight_tt = X.apply(params, t, t, method="calc_weight")
    return jnp.exp(-weight_tt) * velocity_loss + weight_tt


def psd_term(
    params, teacher_params, x0, x1, label,
    s, t, u, h, rng, *, interp, X, psd_type, stopgrad_type,
):
    """Off-diagonal Progressive Self-Distillation loss via the semigroup (slope form).
    h is the split fraction gamma; u = h*s + (1-h)*t."""
    Is = interp.calc_It(s, x0, x1)

    # student: slope of the full jump v_{s,t}(I_s)   (gradients flow)
    X_st, phi_st = X.apply(
        params, s, t, Is, label, train=False, rngs=rng, return_X_and_phi=True
    )

    # teacher: slopes of the two composed segments [s,u] and [u,t]
    if stopgrad_type == "convex":                         # frozen teacher, no bootstrap backprop
        X_su, phi_su = jax.lax.stop_gradient(
            X.apply(teacher_params, s, u, Is, label, train=False,
                    rngs=rng, return_X_and_phi=True)
        )
        X_ut, phi_ut = jax.lax.stop_gradient(
            X.apply(teacher_params, u, t, X_su, label, train=False,  # seg 2 starts at X_{s,u}
                    rngs=rng, return_X_and_phi=True)
        )
    elif stopgrad_type == "none":
        X_su, phi_su = X.apply(params, s, u, Is, label, train=False,
                               rngs=rng, return_X_and_phi=True)
        X_ut, phi_ut = X.apply(params, u, t, X_su, label, train=False,
                               rngs=rng, return_X_and_phi=True)
    else:
        raise ValueError(f"Invalid stopgrad_type: {stopgrad_type}")

    # v_{s,t} = (1-gamma) v_{s,u} + gamma v_{u,t}(X_{s,u})
    if psd_type == "uniform":
        student = phi_st
        teacher = (1 - h) * phi_su + h * phi_ut          # gamma ~ U([0,1])
    elif psd_type == "midpoint":
        student = phi_st
        teacher = 0.5 * (phi_su + phi_ut)                # gamma = 1/2
    else:
        raise ValueError(f"Invalid psd_type: {psd_type}")

    psd_loss = jnp.sum((student - teacher) ** 2)             # convex mode freezes teacher above
    weight_st = X.apply(params, s, t, method="calc_weight")
    return jnp.exp(-weight_st) * psd_loss + weight_st    # e^{-w} L + w
```

Times are sampled as `t ~ U([0,1])` on the diagonal and `(s,t)` by drawing two uniforms and
taking `(min,max)` for the upper-triangle rows. For the uniform variant, `h ~ U([0,1])` and
`u = h*s + (1-h)*t`; in the code `h` is exactly `gamma`, so the teacher is
`(1-h)*phi_su + h*phi_ut`. The loss wrapper concatenates diagonal rows first and off-diagonal
rows second, vmaps/mean-reduces each term separately, and returns
`(diag_loss * diag_bs + offdiag_loss * offdiag_bs) / total_bs` with diagonal fraction
`eta = 0.75`. `teacher_params` are `train_state.params`; gradient flow is stopped inside
`psd_term` for `stopgrad_type == "convex"`.
