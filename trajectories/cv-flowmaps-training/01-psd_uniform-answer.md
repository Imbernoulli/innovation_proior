**Problem.** A flow-map `X_{s,t}(x) = x + (t-s) v_{s,t}(x)` generates in one/few steps, but the
off-diagonal `v_{s,t}` (`s != t`) has no external target — only the model's own consistency. With the
diagonal pinned by flow matching (`v_{t,t} = b_t`), which self-distillation residual trains stably at
image scale and lands the lowest FID?

**Key idea (PSD, uniform).** Use the **semigroup** identity `X_{u,t}(X_{s,u}(x)) = X_{s,t}(x)`: one big
jump equals two composed small jumps. Substituting the parameterization and cancelling `x` turns it
into a scale-free relation among the *slopes* — `v_{s,t} = (1-gamma) v_{s,u} + gamma v_{u,t}(X_{s,u})`
with `u = gamma s + (1-gamma) t` — so the `(t-s)^2` that would make the effective learning rate depend
on jump size is divided out. The **uniform** variant draws the split `gamma ~ U([0,1])` (carried as `h`,
`u = h s + (1-h) t`), supervising self-consistency under *every* way of breaking a jump in two.

**Why it works / why start here.** Among the three off-diagonal residuals the harness exposes, the
semigroup one needs no spatial Jacobian (unlike ESD's `grad X`) and no map derivative (unlike LSD's
`partial_t`) — just three function evaluations — so it is the least likely to diverge on a large UNet,
the right floor to measure. The minimizer is the true map (full-support split + continuity force the
Lagrangian ODE). A stopgradient on the composed teacher (`convex`) makes the self-teacher behave like a
frozen distillation teacher (info flows diagonal -> off-diagonal) and avoids backprop through the inner
jump. The EDM2 uncertainty weight `e^{-w}L + w` (gradient `grad L / L`) equalizes the loss scale across
`(s,t)` and absorbs the extra variance the swept split injects.

**Scaffold edit / hyperparameters.** Fill all four loss bodies in `losses.py` (lines 36–178) with the
reference implementation below; the active term is selected by `bench_env.sh` —
`FLOWMAPS_BENCH_SLURM_ID=1` (PSD uniform) here. The driver routes `eta = 0.75` of the batch through
`diagonal_term` and `0.25` through `psd_term` on upper-triangle pairs with `psd_type="uniform"`,
`stopgrad_type="convex"`, instantaneous teacher (`teacher_params = params`, decay 0). Fixed:
`batch_size=128`, `max_training_step=50000`, `num_eval_step=8`, `num_fid_samples=50000`.

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

The four terms are filled identically for every baseline; only `bench_env.sh`'s
`FLOWMAPS_BENCH_SLURM_ID` differs (here `1` = PSD uniform), selecting which term the driver routes the
off-diagonal batch through. `esd_term` is present in the edit region but inactive for this baseline.
