## Research question

A flow-based generative model trained by flow matching produces excellent CIFAR-10 samples, but
drawing one sample means integrating the probability-flow ODE `xdot_t = b_t(x_t)` from a Gaussian at
`t=0` to data at `t=1` — tens to hundreds of full UNet evaluations per image. The escape is to learn
the **two-time flow map** `X_{s,t}`, the operator that takes a point on a trajectory at time `s`
straight to the same trajectory at time `t`: `X_{0,1}(x_0)` then turns noise into a sample in a single
call (compose a handful of jumps over a coarse time grid to buy back fidelity). The one thing being
designed here is the **self-distillation training objective** that supervises this map off the
diagonal `s != t`, where there is no external regression target — only the model's own internal
consistency. Everything else (network, interpolant, optimizer, sampler, budget) is fixed. The question
is which off-diagonal residual, dropped into a fixed flow-map harness and trained from scratch under a
short fixed budget, lands the lowest CIFAR-10 **FID**.

## Prior art before the first rung (one-/few-step generation lineage)

The first rung reacts to the methods the field already had for fast generation, each of which it is
trying to avoid the cost of.

- **Flow matching / stochastic interpolants (Lipman et al. 2022; Albergo–Vanden-Eijnden 2023).** Build
  `I_t = alpha_t x_0 + beta_t x_1` between Gaussian and data and learn the drift by the square loss
  `L_b(bhat) = int E |bhat_t(I_t) - Idot_t|^2 dt`, whose minimizer is exactly `b_t = E[Idot_t | I_t]`.
  Trains like clockwork and samples beautifully — but sampling *is* a many-step ODE solve. Gap: the
  velocity gives only the instantaneous direction, so a finite move costs many sequential evaluations.
- **Progressive / knowledge distillation of a sampler (Salimans–Ho 2022; Hinton et al. 2015).** Train a
  fast few-step student to imitate a pre-trained many-step teacher, halving the step count repeatedly.
  Stable and effective — but a two-phase pipeline (train the slow model, then the fast one), two models
  to keep, and a student that can never beat the teacher it copies. Gap: the teacher tax.
- **Consistency models and their trajectory extension (Song et al. 2023; Kim et al. 2023).** Train a
  one-/few-step map *directly*, no teacher, by enforcing consistency of points along a trajectory. This
  is the regime the first rung lives in — but the objectives are observed to be touchy on large image
  UNets (they hinge on a spatial-Jacobian / Eulerian residual) and need dataset-specific babysitting to
  converge. Gap: direct training, but optimization is fragile at image scale.

## The fixed substrate

A flow-map training harness (JAX/Flax, EDM2-style) is frozen and must not be touched. It provides: a
linear stochastic interpolant `I_t = alpha_t x_0 + beta_t x_1` (`interp.calc_It`, `interp.calc_It_dot`,
with `Idot_t` constant for the linear schedule); a flow-map module `X` whose single network output
parameterizes `X_{s,t}(x) = x + (t-s) v_{s,t}(x)` (so `X_{s,s}(x) = x` is exact and the diagonal
`v_{t,t}` is the velocity `b_t`), preconditioned EDM2-style and conditioned on `s` and the gap
`dt = t-s`; the loop's per-baseline `slurm_id` selects which loss term is active and trains it with
RAdam, gradient clipping at 1.0, warmup-then-sqrt LR decay, and parameter EMA used for sampling. The
module exposes exactly these calls a loss term may use:

- `X.apply(params, t, x, label, method="calc_b", ...)` — the diagonal velocity `v_{t,t}(x)`.
- `X.apply(params, s, t, x, label, method="partial_t", ...)` — returns `(X_{s,t}(x), partial_t X_{s,t}(x))`
  in one `jax.jvp` along **time** (no spatial Jacobian).
- `X.apply(params, s, t, x, label, method="partial_s", ...)` — returns `(X_{s,t}(x), partial_s X_{s,t}(x))`.
- `X.apply(params, s, t, x, label, return_X_and_phi=True, ...)` — returns `(X_{s,t}(x), v_{s,t}(x))`,
  the mapped point and its implicit slope.
- `X.apply(params, s, t, method="calc_weight")` — a learned per-`(s,t)` scalar weight head `w_{s,t}`.

The harness passes both the student `params` and a `teacher_params` copy, the start/end times `(s,t)`,
an intermediate time `u` and a split fraction `h` for composition terms, and a `stopgrad_type` /
`psd_type` selector. `setup_loss`, imports, and `scripts/bench_env.sh` are off-limits.

## The editable interface

Exactly one region is editable — the four loss-term bodies in `flow-maps/py/common/losses.py`, lines
**36–178** (`diagonal_term`, `psd_term`, `lsd_term`, `esd_term`); the same span as `losses_template.py`.
The contract: `diagonal_term` returns the per-sample loss on the diagonal `s=t` (flow matching on
`v_{t,t}`); `psd_term`, `lsd_term`, `esd_term` each return the per-sample off-diagonal loss for one
self-distillation residual; the driver `vmap`s them over the batch and routes a fraction `eta` of
samples through the diagonal term and the rest through the active off-diagonal term, selected per
baseline by `FLOWMAPS_BENCH_SLURM_ID` (0=LSD, 1=PSD-uniform, 2=PSD-midpoint) in `bench_env.sh`. The
**default** scaffold fill below is deliberately bare: raw squared residuals with **no** adaptive
weighting and a single hard-wired stopgradient placement — a non-paper starting point each baseline
then replaces.

```python
def diagonal_term(
    params: Parameters,
    x0: jnp.ndarray,
    x1: jnp.ndarray,
    label: jnp.ndarray,
    t: float,
    rng: jnp.ndarray,
    *,
    interp: interpolant.Interpolant,
    X: flow_map.FlowMap,
) -> float:
    """Starter diagonal term for agent edits."""
    It = interp.calc_It(t, x0, x1)
    It_dot = interp.calc_It_dot(t, x0, x1)
    bt = X.apply(params, t, It, label, train=True, method="calc_b", rngs=rng)
    return jnp.sum((bt - It_dot) ** 2)


def psd_term(
    params: Parameters,
    teacher_params: Parameters,
    x0: jnp.ndarray,
    x1: jnp.ndarray,
    label: jnp.ndarray,
    s: float,
    t: float,
    u: float,
    h: float,
    rng: jnp.ndarray,
    *,
    interp: interpolant.Interpolant,
    X: flow_map.FlowMap,
    psd_type: str,
    stopgrad_type: str,
) -> float:
    """Starter PSD term for agent edits."""
    del stopgrad_type
    Is = interp.calc_It(s, x0, x1)
    _, phi_st = X.apply(
        params, s, t, Is, label, train=False, rngs=rng, return_X_and_phi=True
    )
    X_su, phi_su = X.apply(
        teacher_params, s, u, Is, label, train=False, rngs=rng, return_X_and_phi=True
    )
    _, phi_ut = X.apply(
        teacher_params,
        u,
        t,
        X_su,
        label,
        train=False,
        rngs=rng,
        return_X_and_phi=True,
    )
    phi_su = jax.lax.stop_gradient(phi_su)
    phi_ut = jax.lax.stop_gradient(phi_ut)
    if psd_type == "uniform":
        teacher = (1 - h) * phi_su + h * phi_ut
    elif psd_type == "midpoint":
        teacher = 0.5 * (phi_su + phi_ut)
    else:
        raise ValueError(f"Invalid psd_type: {psd_type}")
    return jnp.sum((phi_st - teacher) ** 2)


def lsd_term(
    params: Parameters,
    teacher_params: Parameters,
    x0: jnp.ndarray,
    x1: jnp.ndarray,
    label: jnp.ndarray,
    s: float,
    t: float,
    rng: jnp.ndarray,
    *,
    interp: interpolant.Interpolant,
    X: flow_map.FlowMap,
    stopgrad_type: str,
) -> float:
    """Starter LSD term for agent edits."""
    Is = interp.calc_It(s, x0, x1)
    Xst_Is, dt_Xst = X.apply(
        params, s, t, Is, label, train=False, method="partial_t", rngs=rng
    )
    if stopgrad_type == "none":
        b_eval = X.apply(
            params, t, Xst_Is, label, train=False, method="calc_b", rngs=rng
        )
    elif stopgrad_type == "convex":
        b_eval = jax.lax.stop_gradient(
            X.apply(
                teacher_params,
                t,
                jax.lax.stop_gradient(Xst_Is),
                label,
                train=False,
                method="calc_b",
                rngs=rng,
            )
        )
    else:
        raise ValueError(f"Invalid stopgrad_type: {stopgrad_type}")
    return jnp.sum((b_eval - dt_Xst) ** 2)


def esd_term(
    params: Parameters,
    teacher_params: Parameters,
    x0: jnp.ndarray,
    x1: jnp.ndarray,
    label: jnp.ndarray,
    s: float,
    t: float,
    rng: jnp.ndarray,
    *,
    interp: interpolant.Interpolant,
    X: flow_map.FlowMap,
    stopgrad_type: str,
) -> float:
    """Starter ESD term for agent edits."""
    Is = interp.calc_It(s, x0, x1)
    _, ds_Xst = X.apply(
        params, s, t, Is, label, train=False, method="partial_s", rngs=rng
    )
    if stopgrad_type == "none":
        b_eval = X.apply(
            params, s, Is, label, train=False, method="calc_b", rngs=rng
        )
    elif stopgrad_type in ("convex", "full"):
        b_eval = jax.lax.stop_gradient(
            X.apply(
                teacher_params, s, Is, label, train=False, method="calc_b", rngs=rng
            )
        )
    else:
        raise ValueError(f"Invalid stopgrad_type: {stopgrad_type}")
    _, grad_Xst_b = jax.jvp(
        lambda x: X.apply(params, s, t, x, label, train=False, rngs=rng),
        primals=(Is,),
        tangents=(b_eval,),
    )
    return jnp.sum((ds_Xst + grad_Xst_b) ** 2)
```

## Evaluation settings

CIFAR-10 (TensorFlow Datasets; the scripts ensure `cifar_stats.npz` exists). Three model scales run as
separate benchmarks — `train_small`, `train_medium`, `train_large` — each setting `FLOWMAPS_UNET_SIZE`
to the small / medium / large EDM2 preset in `configs.cifar10_bench`. Fixed training hyperparameters
(do not change): `batch_size=128`, `max_training_step=50000`, `num_eval_step=8` (the few-step sampler's
NFE at evaluation), `num_fid_samples=50000`. The metric is **Fréchet Inception Distance (FID)** —
**lower is better** — recorded per benchmark as `fid_<label>` and `best_fid_<label>` (e.g.
`best_fid_train_small`). Seed: 42.
