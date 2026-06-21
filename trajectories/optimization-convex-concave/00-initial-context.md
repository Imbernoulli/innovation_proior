## Research question

Design an iteration map for `min_x max_y f(x,y)` that uses only a first-order saddle-gradient oracle `F(z)=[∇_x f; -∇_y f]` plus fixed-scale additive Gaussian update noise. The graded quantity is the operator norm `‖F(z)‖` at the reported iterate. Everything else — the two problem instances, the noise model, the iteration budgets, and the metric computation — is fixed. On the bilinear instance `f=xy`, the duality gap is infinite at every non-saddle point, so `‖F(z)‖` is the only usable metric.

## Prior art / Background / Baselines

- **Simultaneous gradient descent-ascent.** Descend in `x` and ascend in `y` at once: `z_{t+1} = z_t − τ F(z_t)`.
- **Proximal point / implicit step.** The resolvent `z_{t+1} = (I+τF)^{-1}(z_t)` is firmly nonexpansive for monotone `F` and contracts unconditionally; each step requires evaluating `F` at the unknown next point.
- **Extragradient.** Take a forward look-ahead point, evaluate the operator there, then step from the current iterate. It converges on monotone Lipschitz problems and under noise settles in an `O(τσ²)` ball, reducing the gradient norm at an `O(1/k)` rate.
- **Halpern anchoring.** Mix each iterate back toward a fixed reference point, with convergence tied to a vanishing step size.

## Fixed substrate / Code framework

The benchmark harness provides the two problem definitions and exact iteration counts, the deterministic operator `oracle.grad(z) = F(z)`, the fixed-scale Gaussian update perturbation `oracle.noise()` (one fresh draw per call), the fixed starting point passed to `init_state`, and the metric computation. `F` is monotone and `L`-Lipschitz. The bilinear field is a pure rotation (`L=1`); the `(δ,ν)` field is a clipped-monotone component plus a small skew coupling. The harness measures `‖F(z)‖` at a per-problem iterate (post-step on `bilinear`, where `‖F(z)‖=‖z‖`; pre-step otherwise).

## Editable interface

Only `init_state`, `step`, and `get_hyperparameters` in `RAIN/optimization_convex_concave/custom_strategy.py` are editable. The contract: `init_state(problem, initial_z, seed, hyperparameters)` must preserve the start in `state["z"]`; `step(state, oracle, problem, hyperparameters, max_sfo_calls)` runs one iteration and returns `make_step_output(next_state, metric_iterate, sfo_calls)` where `sfo_calls` must equal the number of `oracle.grad` calls consumed; `get_hyperparameters(problem_name, sigma)` returns the per-problem constants. The default fill is the plain stochastic extragradient step.

```python
"""Editable strategy scaffold for the optimization-convex-concave MLS-Bench task."""

from __future__ import annotations

from typing import Any

import numpy as np

from fixed_benchmark import (
    ProblemSpec,
    StepOutput,
    StochasticOracle,
    as_vector,
    make_step_output,
    run_cli,
)


# =====================================================================
# EDITABLE: init_state, step, get_hyperparameters
# =====================================================================


def init_state(
    problem: ProblemSpec,
    initial_z: np.ndarray,
    seed: int,
    hyperparameters: dict[str, Any],
) -> dict[str, Any]:
    """Initialize algorithm state from the fixed starting point."""
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
    """Default baseline: the official SEG / EG update from the MATLAB scripts."""
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
    """Return the official per-problem step size."""
    if problem_name == "bilinear":
        return {"tau": 0.1}
    if problem_name == "delta_nu":
        return {"tau": 1.0}
    raise KeyError(f"Unknown problem: {problem_name}")
```

## Evaluation settings

Two problem instances with additive Gaussian update noise, run for the per-problem iteration counts at seed 42:

- **`bilinear`**: scalar `f(x,y)=xy`, `n=900` iterations, `τ=0.1`, `z0=[10,10]ᵀ`, `σ=0.001`.
- **`delta_nu`**: `(δ,ν)` problem, `d=100`, `δ=1e-2`, `ν=5e-5`, `n=6000` iterations, `τ=1`, `σ=0.02`, `z0 ~ N(0,I)` under the script's fixed RNG.

Three noise regimes share the same code (`default-noise`, `low-noise`, `high-noise`; the σ above is the default). The primary metric is `final_gradient_norm` — the mean of the two official final gradient norms — **lower is better**; the harness also reports a per-iteration `gradient_norm` trace and `auc_log_iteration_log_grad`.
