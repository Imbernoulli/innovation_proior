**Problem.** SEG fixed the bilinear bias (`bilinear_fgn` `1.41→0.174`) but, with no anchor, lost
contraction on the merely-monotone `(δ,ν)` field (`delta_nu_fgn` `0.093→0.190`, and `0.94` at high
noise). Want a form of anchoring that contracts the flat field — including its rotation — without
R-SEG's permanent fixed-`z0` bias.

**Key idea (SEAG: stochastic extra-anchored gradient).** Plant a *decaying* Halpern anchor inside
both half-steps of extragradient. Predictor `w = z − τF(z) + c_k(z0 − z) + noise`, corrector
`z_next = z − τF(w) + c_k(z0 − z) + noise`, with the same offset relative to the current point `z` in
both lines and a constant gradient step `τ`. The anchor coefficient decays as `c_k = 1/(k+3)`.

**Why.** The defect in R-SEG was a *constant* anchor weight, so its bias `λ‖z0−z*‖` never died. The
anchored flow `ż=−F(z)−β(t)(z−z0)` is fastest at `β(t)=1/t` (contracting and vanishing speeds
matched); on `f=xy` it gives `‖z(t)‖²∼1/t²` with the `z0` dependence in bounded oscillating
numerators over `t` — contraction *and* vanishing bias. The discrete `c_k=1/(k+3)` realizes this (the
`+3` keeps `c_0=1/3<1`); it is a **pure** convex pull `c_k(z0−z)`, no `τλ` factor, so it can be
order-one early yet die to nothing. A Lyapunov function `A_k‖F(z_k)‖²+B_k⟨F(z_k),z_k−z0⟩` with this
schedule has `B_k` linear, `A_k` quadratic, giving the optimal `O(1/k²)` last-iterate gradient-norm
rate — no bias. Under noise the `k²` weight amplifies accumulated noise (like stochastic Nesterov),
so expect a fast transient then a σ-set floor, not the deterministic `1/k²`.

**Hyperparameters.** `bilinear`: `τ=0.1`. `delta_nu`: `τ=1.0`. Anchor coefficient `c_k=1/(k+3)` with
`k` the zero-based `step_index`; anchor `z0` fixed. Two operator evaluations, two noise draws per
iteration.

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
    z = as_vector(state["z"], expected_dim=2 * problem.dim)
    anchor_z = as_vector(state["anchor_z"], expected_dim=2 * problem.dim)
    step_index = int(state.get("step_index", 0))
    coeff = 1.0 / (step_index + 3.0)

    g = oracle.grad(z)
    w = z - tau * g + coeff * (anchor_z - z) + oracle.noise()
    gw = oracle.grad(w)
    z_next = z - tau * gw + coeff * (anchor_z - z) + oracle.noise()
    metric_iterate = z_next if problem.name == "bilinear" else z
    return make_step_output(
        {"z": z_next, "anchor_z": anchor_z, "step_index": step_index + 1},
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
