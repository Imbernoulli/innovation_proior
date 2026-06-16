## Research question

A noisy convex-concave saddle problem, `min_x max_y f(x,y)`, reached only through a first-order
oracle: hand it a point `z=(x,y)`, get back the saddle-gradient operator `F(z)=[∇_x f ; -∇_y f]`,
and (separately) a fixed-scale additive Gaussian perturbation on each update. The single thing being
designed is the **iteration map** — how one step turns the current iterate into the next, using
operator evaluations and the update noise. The graded quantity is the **gradient (operator) norm**
`‖F(z)‖` at the reported iterate, not the duality gap: on the bilinear instance `f=xy` the gap is
`+∞` at every non-saddle point and cannot even rank two bad iterates, whereas `‖F(z)‖` is always
finite and is exactly zero at a solution. Everything else — the two problem instances, the noise
model, the iteration budgets, the metric computation — is frozen.

## Prior art before the first rung (saddle-point optimization lineage)

The first rung reacts to the failure of the obvious explicit method and to the operator-theoretic
fixes that precede the ladder.

- **Simultaneous gradient descent-ascent (Arrow–Hurwicz–Uzawa, 1958).** Descend in `x`, ascend in
  `y` at once: `z_{t+1} = z_t − τ F(z_t)`. On `f=xy`, `F(z)=[y,−x]=Jz` is a 90° rotation, so the
  step is orthogonal to the pull toward the saddle and `‖z_{t+1}‖² = (1+τ²)‖z_t‖²` — it spirals
  strictly outward for *every* `τ>0`. Gap: diverges on the simplest convex-concave instance.
- **Proximal point / implicit (backward) step (Martinet 1970; Rockafellar 1976).** The resolvent
  `z_{t+1} = (I+τF)^{-1}(z_t)` is firmly nonexpansive for any monotone `F` and contracts
  unconditionally (modulus `1/√(1+τ²)<1` on the bilinear field). Gap: implicit — needs `F` at the
  unknown next point, a nonlinear solve per step; an ideal to imitate, not to run.
- **Extragradient (Korpelevich 1976).** Imitate the implicit step explicitly: a forward look-ahead
  to `w`, evaluate `F(w)`, then step from the *original* `z_t`. The anchor at `z_t` manufactures the
  `−τ²I` inward term the forward step lacked. Gap: converges on monotone Lipschitz problems but,
  under noise, only to an `O(τσ²)` ball, and only `O(1/k)` on the merely-monotone gradient norm.
- **Halpern anchoring (Halpern 1967).** Mix each iterate back toward a fixed reference: pulling
  toward an anchor makes the *last* iterate converge and implicitly selects the nearest solution.
  Gap on its own: tied to a vanishing step, and anchoring at a far, fixed point biases the solution.

## The fixed substrate

The benchmark harness mirrors the official RAIN convex-concave scripts and must not be touched. It
provides: the two problem definitions and their exact iteration counts, the deterministic operator
`oracle.grad(z) = F(z)` and the fixed-scale Gaussian update perturbation `oracle.noise()` (one fresh
draw per call), the fixed starting point passed to `init_state`, and the metric computation. `F` is
monotone (`f` convex-concave) and `L`-Lipschitz. The bilinear field is a pure rotation (`L=1`); the
`(δ,ν)` field is a clipped-monotone component plus a small skew coupling. The harness measures
`‖F(z)‖` at a per-problem iterate (post-step on `bilinear`, where `‖F(z)‖=‖z‖`; pre-step otherwise).

## The editable interface

Exactly one region is editable — `init_state`, `step`, and `get_hyperparameters` in
`RAIN/optimization_convex_concave/custom_strategy.py` (lines 24–75). The contract:
`init_state(problem, initial_z, seed, hyperparameters)` must preserve the start in `state["z"]`;
`step(state, oracle, problem, hyperparameters, max_sfo_calls)` runs one official-style iteration and
returns `make_step_output(next_state, metric_iterate, sfo_calls)` where `sfo_calls` must equal the
number of `oracle.grad` calls consumed; `get_hyperparameters(problem_name, sigma)` returns the
per-problem constants. The default fill is the **plain stochastic extragradient (SEG) / EG** step
from the MATLAB scripts.

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

Two problem instances, both with additive Gaussian update noise, run for the official per-problem
iteration counts at seed 42:

- **`bilinear`**: scalar `f(x,y)=xy`, `n=900` iterations, `τ=0.1`, `z0=[10,10]ᵀ`, `σ=0.001`.
- **`delta_nu`**: `(δ,ν)` problem, `d=100`, `δ=1e-2`, `ν=5e-5`, `n=6000` iterations, `τ=1`,
  `σ=0.02`, `z0 ~ N(0,I)` under the script's fixed RNG.

Three noise regimes share the same code (`default-noise`, `low-noise`, `high-noise`; the σ above is
the default). The primary metric is `final_gradient_norm` — the mean of the two official final
gradient norms — **lower is better**; the harness also reports a per-iteration `gradient_norm` trace
and `auc_log_iteration_log_grad`.
