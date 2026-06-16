**Problem.** SEAG reaches its floor fast (AUC `−0.347→−1.107`) but cannot get below it: its single
fixed anchor at `z0` must keep the regularization strength small (it decays), and small strength is a
weak strong-monotonicity, which leaves the noise floor large (high-noise `delta_nu` `0.583`). Strength
(fights noise) and bias (from a far fixed anchor) are coupled through the anchor-to-`z*` distance.

**Key idea (RAIN: recursively/running-anchored stochastic extragradient).** Break the coupling by
re-anchoring toward an increasingly close point while raising the strength. The non-expansiveness
`‖w*−z*‖ ≤ ‖z0−z*‖` says the anchored solution is closer to `z*`, so anchor there next and crank `λ`
up geometrically. Recursive anchoring lemma: `‖F(z_S)‖ ≤ 16λ Σ_s (1+γ)^{s-1}‖z*_{s-1}−z_s‖`, a
geometrically-weighted sum of per-round errors; since the subproblem strong-monotonicity grows at the
same rate, the total cost hits the statistical floor `Õ(σ²ε^{-2}+κ)`. Collapsed to one loop
(`N_s=1, K_s=0`), the accumulated penalty becomes an extragradient step whose regularizer is a
geometrically-weighted running anchor over the stored trajectory.

**Why.** Unlike SEAG's *decaying* single anchor, the effective regularization toward the recent
trajectory *grows* (the geometric weights `(1+γ)^j` accumulate), so late iterations are increasingly
strongly monotone around where the trajectory has settled (near `z*`): the contraction keeps
tightening and the noise floor keeps shrinking, while the bias stays bounded because the anchor tracks
the moving trajectory, not the far `z0`.

**Implementation.** Maintain `weight_sum = Σ_j w_j` (scalar) and `weighted_flow_sum = Σ_j w_j z_j`
(vector); the anchor contribution is `τλ(weighted_flow_sum − weight_sum·z)`, `O(d)`. Each new iterate
`z_next` is inserted with weight `γ(1+γ)^{step_index+1}`. First step has empty buffers (plain EG).

**Hyperparameters.** `bilinear`: `τ=0.1, λ=0.1, γ=0.001`. `delta_nu`: `τ=1.0, λ=0.01, γ=0.0001`. `γ`
tiny so `(1+γ)^t` cannot overflow over 900/6000 iterations. Two operator evaluations, two noise draws,
`O(d)` buffer updates per iteration.

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
        "step_index": 0,
        "weight_sum": 0.0,
        "weighted_flow_sum": np.zeros_like(z0),
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
    gamma = float(hyperparameters["gamma"])
    z = as_vector(state["z"], expected_dim=2 * problem.dim)
    step_index = int(state.get("step_index", 0))
    weight_sum = float(state.get("weight_sum", 0.0))
    weighted_flow_sum = as_vector(state.get("weighted_flow_sum", np.zeros_like(z)), expected_dim=2 * problem.dim)

    g = oracle.grad(z)
    anchor_z = tau * lam * (weighted_flow_sum - weight_sum * z)
    w = z - tau * g + anchor_z + oracle.noise()
    gw = oracle.grad(w)
    anchor_w = tau * lam * (weighted_flow_sum - weight_sum * w)
    z_next = z - tau * gw + anchor_w + oracle.noise()

    current_weight = gamma * (1.0 + gamma) ** (step_index + 1)
    next_state = {
        "z": z_next,
        "step_index": step_index + 1,
        "weight_sum": weight_sum + current_weight,
        "weighted_flow_sum": weighted_flow_sum + current_weight * z_next,
    }
    metric_iterate = z_next if problem.name == "bilinear" else z
    return make_step_output(next_state, metric_iterate, 2)


def get_hyperparameters(problem_name: str, sigma: float) -> dict[str, Any]:
    if problem_name == "bilinear":
        return {"tau": 0.1, "lambda": 0.1, "gamma": 0.001}
    if problem_name == "delta_nu":
        return {"tau": 1.0, "lambda": 0.01, "gamma": 0.0001}
    raise KeyError(f"Unknown problem: {problem_name}")
```
