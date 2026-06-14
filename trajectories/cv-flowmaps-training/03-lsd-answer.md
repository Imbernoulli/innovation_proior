**Problem (from step 2).** The midpoint semigroup variant fixed the split-variance deficit (14.99 ->
10.79) but stalled there: its off-diagonal target `0.5 (v_{s,u} + v_{u,t}(X_{s,u}))` still composes two
learned jumps — a network at another network's output — so the bootstrap (compounding error,
distribution shift, no clean accuracy bound) caps it. Going lower means a target that is *not* a
composition.

**Key idea (LSD).** Supervise the off-diagonal with the **Lagrangian** identity
`partial_t X_{s,t}(x) = b_t(X_{s,t}(x))` instead of the semigroup. The residual
`v_{t,t}(X_{s,t}(I_s)) - partial_t X_{s,t}(I_s)` uses a single frozen-teacher velocity at the
transported point and a *scalar-time* JVP (`partial_t`, ~1.5 fwd, returns `X` for free) — no composition
(unlike PSD) and no spatial Jacobian (unlike ESD). It threads exactly between the ladder's two observed
failure modes.

**Why it works.** The minimizer is the same true map (zero residual + correct diagonal = the unique
solution of the Lagrangian IVP under one-sided Lipschitz `b`), but now with a certificate PSD lacked: if
the diagonal loss and the LSD residual are each `<= eps`, then `W_2^2(rhohat_1, rho_1) <= 4 e^{1+2Lhat}
eps` (flow-matching Grönwall + flow-map-matching bound + triangle), so driving the loss down provably
improves the one-step sampler. The `convex` stopgrad freezes the teacher velocity *and* the transported
point (info flows diagonal -> off-diagonal, no spatial-Jacobian leak). Per off-diagonal sample it is also
cheaper than PSD: ~2.5 forward-equivalents vs PSD's 3.

**Scaffold edit / hyperparameters.** Identical reference fill of `losses.py` (lines 36–178); the only
change from step 2 is `bench_env.sh` -> `FLOWMAPS_BENCH_SLURM_ID=0` (LSD), so the driver routes the
off-diagonal batch through `lsd_term` with `stopgrad_type="convex"` instead of `psd_term`. Unchanged:
diagonal flow-matching term, uncertainty weight `e^{-w}L + w`, instantaneous teacher, batch split
`eta = 0.75` diagonal / `0.25` off-diagonal. Fixed: `batch_size=128`, `max_training_step=50000`,
`num_eval_step=8`, `num_fid_samples=50000`.

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
    """Compute the diagonal (interpolant) term of the loss."""
    It = interp.calc_It(t, x0, x1)
    It_dot = interp.calc_It_dot(t, x0, x1)
    bt = X.apply(params, t, It, label, train=True, method="calc_b", rngs=rng)
    velocity_loss = jnp.sum((bt - It_dot) ** 2)
    weight_tt = X.apply(params, t, t, method="calc_weight")
    return jnp.exp(-weight_tt) * velocity_loss + weight_tt


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
    """Compute the PSD (Progressive Self-Distillation) term of the loss."""
    Is = interp.calc_It(s, x0, x1)
    X_st, phi_st = X.apply(
        params, s, t, Is, label, train=False, rngs=rng, return_X_and_phi=True
    )
    if stopgrad_type == "convex":
        X_su, phi_su = jax.lax.stop_gradient(
            X.apply(
                teacher_params,
                s,
                u,
                Is,
                label,
                train=False,
                rngs=rng,
                return_X_and_phi=True,
            )
        )
        X_ut, phi_ut = jax.lax.stop_gradient(
            X.apply(
                teacher_params,
                u,
                t,
                X_su,
                label,
                train=False,
                rngs=rng,
                return_X_and_phi=True,
            )
        )
    elif stopgrad_type == "none":
        X_su, phi_su = X.apply(
            params,
            s,
            u,
            Is,
            label,
            train=False,
            rngs=rng,
            return_X_and_phi=True,
        )
        X_ut, phi_ut = X.apply(
            params,
            u,
            t,
            X_su,
            label,
            train=False,
            rngs=rng,
            return_X_and_phi=True,
        )
    else:
        raise ValueError(f"Invalid stopgrad_type: {stopgrad_type}")
    if psd_type == "uniform":
        student = phi_st
        teacher = (1 - h) * phi_su + h * phi_ut
    elif psd_type == "midpoint":
        student = phi_st
        teacher = 0.5 * (phi_su + phi_ut)
    else:
        raise ValueError(f"Invalid psd_type: {psd_type}")
    psd_loss = jnp.sum((student - teacher) ** 2)
    weight_st = X.apply(params, s, t, method="calc_weight")
    return jnp.exp(-weight_st) * psd_loss + weight_st


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
    """Compute the LSD term of the loss."""
    Is = interp.calc_It(s, x0, x1)
    Xst_Is, dt_Xst = X.apply(
        params, s, t, Is, label, train=False, method="partial_t", rngs=rng
    )
    if stopgrad_type == "convex":
        Xst_Is = jax.lax.stop_gradient(Xst_Is)
        b_eval = jax.lax.stop_gradient(
            X.apply(
                teacher_params,
                t,
                Xst_Is,
                label,
                train=False,
                method="calc_b",
                rngs=rng,
            )
        )
    elif stopgrad_type == "none":
        b_eval = X.apply(
            params,
            t,
            Xst_Is,
            label,
            train=False,
            method="calc_b",
            rngs=rng,
        )
    else:
        raise ValueError(f"Invalid stopgrad_type: {stopgrad_type}")
    weight_st = X.apply(params, s, t, method="calc_weight")
    error = b_eval - dt_Xst
    lsd_loss = jnp.sum(error**2)
    return jnp.exp(-weight_st) * lsd_loss + weight_st


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
    """Compute the ESD term of the loss."""
    Is = interp.calc_It(s, x0, x1)
    _, ds_Xst = X.apply(
        params, s, t, Is, label, train=False, method="partial_s", rngs=rng
    )
    if stopgrad_type == "full":
        b_eval = jax.lax.stop_gradient(
            X.apply(
                teacher_params,
                s,
                Is,
                label,
                train=False,
                method="calc_b",
                rngs=rng,
            )
        )
        _, grad_Xst_b = jax.lax.stop_gradient(
            jax.jvp(
                lambda x: X.apply(
                    teacher_params, s, t, x, label, train=False, rngs=rng
                ),
                primals=(Is,),
                tangents=(b_eval,),
            )
        )
    elif stopgrad_type == "convex":
        b_eval = jax.lax.stop_gradient(
            X.apply(
                teacher_params,
                s,
                Is,
                label,
                train=False,
                method="calc_b",
                rngs=rng,
            )
        )
        _, grad_Xst_b = jax.jvp(
            lambda x: X.apply(params, s, t, x, label, train=False, rngs=rng),
            primals=(Is,),
            tangents=(b_eval,),
        )
    elif stopgrad_type == "none":
        b_eval = X.apply(
            params,
            s,
            Is,
            label,
            train=False,
            method="calc_b",
            rngs=rng,
        )
        _, grad_Xst_b = jax.jvp(
            lambda x: X.apply(params, s, t, x, label, train=False, rngs=rng),
            primals=(Is,),
            tangents=(b_eval,),
        )
    else:
        raise ValueError(f"Invalid stopgrad_type: {stopgrad_type}")
    esd_loss = jnp.sum((ds_Xst + grad_Xst_b) ** 2)
    weight_st = X.apply(params, s, t, method="calc_weight")
    return jnp.exp(-weight_st) * esd_loss + weight_st
```

The loss bodies are byte-for-byte the step-1/step-2 fill; LSD differs only by
`FLOWMAPS_BENCH_SLURM_ID=0`, which routes the off-diagonal batch through `lsd_term`. `psd_term` and
`esd_term` are present but inactive for this baseline.
