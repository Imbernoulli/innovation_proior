**Problem.** R-SEG's mean was pinned by the bilinear half (`1.41 ≈ λ‖z0−z*‖`): the fixed-`z0`
Tikhonov anchor converged to the regularized solution `w*`, biased away from the true saddle by the
anchor pull toward `[10,10]ᵀ`. The cure is to remove the anchor and run the bare look-ahead step,
whose contraction comes from the operator's own curvature, not from an external pull.

**Key idea (SEG: stochastic extragradient — the scaffold default).** Imitate the unconditionally
stable implicit step `(I+τF)^{-1}` explicitly: a forward look-ahead `w = z_t − τF(z_t)`, then the
actual step from the *original* `z_t` using the field at `w`, `z_{t+1} = z_t − τF(w)`. Anchoring at
`z_t` (not at `w`) manufactures a `−τ²I` inward term.

**Why.** Simultaneous descent-ascent diverges on `f=xy` (`F=Jz` a rotation, update modulus
`√(1+τ²)>1`). The corrected step gives `z_{t+1}=(I−τJ−τ²I)z_t`, modulus `√(1−τ²(1−τ²))<1` for `τ<1`
— the contraction the forward step lacked, with **no anchor bias**, so on bilinear the iterate
contracts toward the true origin and the `1.41` floor vanishes. The `−τ²I` is the leading curvature
term of the resolvent; EG matches the implicit step to `O(τ²)` vs the forward step's `O(τ)` when
`τ<1/L`. Cost of dropping the anchor: `f` is merely monotone (`μ=0`), so no geometric rate and no
last-iterate gradient-norm contraction — only the `O(1/k)` ergodic gap and an `O(τσ²)` noise
neighborhood. `delta_nu` may lose R-SEG's stabilizing `λ` and drift up.

**Hyperparameters.** `bilinear`: `τ=0.1` (`L=1`, `τ<1/L` with margin). `delta_nu`: `τ=1.0`. Both
evaluations use the same deterministic operator (the same-sample rule); the harness noise is additive
update noise. Two operator evaluations, two noise draws per iteration; no anchor state.

```python
def init_state(
    problem: ProblemSpec,
    initial_z: np.ndarray,
    seed: int,
    hyperparameters: dict[str, Any],
) -> dict[str, Any]:
    return {
        "z": as_vector(initial_z, expected_dim=2 * problem.dim),
        "step_index": 0,
    }


def step(
    state: dict[str, Any],
    oracle: StochasticOracle,
    problem: ProblemSpec,
    hyperparameters: dict[str, Any],
    max_sfo_calls: int,
) -> StepOutput:
    tau = float(hyperparameters["tau"])
    z = as_vector(state["z"], expected_dim=2 * problem.dim)
    step_index = int(state.get("step_index", 0))

    g = oracle.grad(z)
    w = z - tau * g + oracle.noise()
    gw = oracle.grad(w)
    z_next = z - tau * gw + oracle.noise()
    metric_iterate = z_next if problem.name == "bilinear" else z
    return make_step_output(
        {"z": z_next, "step_index": step_index + 1},
        metric_iterate,
        2,
    )


def get_hyperparameters(problem_name: str, sigma: float) -> dict[str, Any]:
    if problem_name == "bilinear":
        return {"tau": 0.1}
    if problem_name == "delta_nu":
        return {"tau": 1.0}
    raise KeyError(f"Unknown problem: {problem_name}")
```
