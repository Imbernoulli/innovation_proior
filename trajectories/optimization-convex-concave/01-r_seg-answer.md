**Problem.** A noisy convex-concave saddle problem; the graded quantity is the operator gradient
norm `‖F(z)‖`. Plain stochastic extragradient (the scaffold default) defeats the rotation but, since
`f` is only convex-concave (`λ=0`), gets no contraction and parks at a noise-limited neighborhood:
its per-step contraction inequality is multiplied by the strong-monotonicity constant the problem
does not have.

**Key idea (R-SEG: Tikhonov-anchored stochastic extragradient).** Manufacture the missing strong
monotonicity by adding a quadratic penalty toward a fixed anchor `a=z0`: run extragradient on
`G(z) = F(z) + λ(z − z0)`, which is `λ`-strongly monotone by construction and still `(L+λ)`-Lipschitz.
Each half-step adds a fixed pull `τλ(z0 − ·)` toward `z0`, with the anchor re-evaluated at the look-
ahead in the corrector.

**Why.** SEG on `G` contracts geometrically (rate set by `λ`) toward `G`'s zero `w*` and has a smaller
noise floor `~ησ²/λ`. The transfer bound `‖F(z̃)‖ ≤ 2‖G(z̃)‖ + λ‖z0 − z*‖` (via the triangle
inequality, strong monotonicity, and the non-expansiveness `‖w*−z0‖ ≤ ‖z*−z0‖`) shows the only price
is an **irreducible bias** `λ‖z0 − z*‖`. Balancing bias against conditioning fixes `λ ~ ε/D`. The
weakness this rung exposes: when `z0` is far from `z*` (bilinear, `z0=[10,10]ᵀ`, `‖z0−z*‖≈14.1`), the
bias `λ‖z0−z*‖ ≈ 1.4` dominates and the iterate is dragged toward the worst point.

**Hyperparameters.** `bilinear`: `τ=0.1, λ=0.1`. `delta_nu`: `τ=1.0, λ=0.01`. Anchor `a=z0`, fixed
forever (the transfer bound needs only the single distance `‖z0−z*‖`). Two operator evaluations and
two noise draws per iteration; unconstrained, so no projection.

```python
def init_state(
    problem: ProblemSpec,
    initial_z: np.ndarray,
    seed: int,
    hyperparameters: dict[str, Any],
) -> dict[str, Any]:
    z0 = as_vector(initial_z, expected_dim=2 * problem.dim)
    return {
        "z": z0,
        "anchor_z": z0.copy(),
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
    lam = float(hyperparameters["lambda"])
    z = as_vector(state["z"], expected_dim=2 * problem.dim)
    anchor_z = as_vector(state["anchor_z"], expected_dim=2 * problem.dim)
    step_index = int(state.get("step_index", 0))

    g = oracle.grad(z)
    w = z - tau * g + tau * lam * (anchor_z - z) + oracle.noise()
    gw = oracle.grad(w)
    z_next = z - tau * gw + tau * lam * (anchor_z - w) + oracle.noise()
    metric_iterate = z_next if problem.name == "bilinear" else z
    return make_step_output(
        {"z": z_next, "anchor_z": anchor_z, "step_index": step_index + 1},
        metric_iterate,
        2,
    )


def get_hyperparameters(problem_name: str, sigma: float) -> dict[str, Any]:
    if problem_name == "bilinear":
        return {"tau": 0.1, "lambda": 0.1}
    if problem_name == "delta_nu":
        return {"tau": 1.0, "lambda": 0.01}
    raise KeyError(f"Unknown problem: {problem_name}")
```
