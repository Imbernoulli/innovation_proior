# Stochastic Extragradient (SEG), distilled

The Extragradient method solves convex-concave saddle points `min_x max_y f(x, y)` — and more
generally monotone variational inequalities, find `z*` with `F(z*) = 0` for the game vector field
`F(z) = [∇_x f ; -∇_y f]` — by a two-step **predictor-corrector** iteration that is fully explicit
yet imitates the unconditionally stable implicit (proximal-point) step. The stochastic variant
(SEG) is the same iteration under noisy updates, with the rule that **both** gradient evaluations
in a step use the **same** sample/operator.

## Problem it solves

Simultaneous gradient descent-ascent `z_{t+1} = z_t - τ F(z_t)` *diverges geometrically* on the
simplest saddle point `f(x, y) = x·y`: there `F(z) = [y; -x] = Jz` with `J` skew-symmetric
(eigenvalues `±i`), so the update operator `I - τJ` has eigenvalues `1 ∓ iτ` of modulus
`√(1 + τ²) > 1` for every `τ > 0` — an outward spiral. The field is monotone but with zero
strong-monotonicity margin (pure rotation), so a single forward evaluation has no contraction to
grab. We want a method that drives `‖F(z_t)‖ → 0` on monotone Lipschitz problems, using only
explicit operator evaluations (no inner solve, no matrix inverse), robust to update noise.

## Key idea

The implicit/proximal step `z_{t+1} = z_t - τ F(z_{t+1}) = (I + τF)^{-1}(z_t)` is firmly
nonexpansive for any `τ > 0` on a monotone operator (on bilinear, `(I + τJ)^{-1}` has modulus
`1/√(1+τ²) < 1`), but is
unrunnable — it needs `F` at the unknown next point. Extragradient **approximates that future
point explicitly**: take a forward "look-ahead" step to `w`, evaluate `F` there, and step from the
*original* `z_t`:

```
w       = z_t - τ F(z_t)        # predictor (look-ahead / leader iterate)
z_{t+1} = z_t - τ F(w)          # corrector: aim with F(w), anchor at z_t
```

Anchoring the actual step at `z_t` (not at `w`) is the defining feature — stepping from `w` would
just be two forward steps and would still diverge. On bilinear it produces an extra `-τ²I`:
`z_{t+1} = (I - τJ - τ²I) z_t`, eigenvalue modulus `√((1-τ²)² + τ²) = √(1 - τ²(1-τ²)) < 1` for
`τ < 1` — the contraction the forward step lacked, manufactured by one extra gradient evaluation.

This `-τ²` is the leading curvature term of the resolvent. On the scalar bilinear field,
`(I + τJ)^{-1} = (1/(1+τ²))(I - τJ) = I - τJ - τ²I + O(τ³)`, so the corrected explicit step keeps
the second-order inward term that the forward step drops. General statement, for `F` `L`-Lipschitz
with `w_imp` the true implicit point:

```
‖z_EG - w_imp‖ ≤ τ²L² ‖z_t - w_imp‖,
```

so extragradient matches the implicit step to `O(τ²)` versus the forward step's `O(τ)` — a genuine
improvement exactly when `τ < 1/L`, which is why extragradient uses small step sizes.

## Convergence (deterministic)

With `z* : F(z*) = 0`, `w = z_t - τF(z_t)`, `z_{t+1} = z_t - τF(w)`, completing the square gives
the one-step identity

```
‖z_{t+1} - z*‖² = ‖z_t - z*‖²
               - 2τ ⟨F(w), w - z*⟩            (≥0 dropped by monotonicity: progress)
               + τ²‖F(w) - F(z_t)‖² - ‖w - z_t‖²   (discretization error).
```

Monotonicity (`⟨F(w) - F(z*), w - z*⟩ ≥ 0`, `F(z*)=0`) makes the middle term a nonpositive
progress term. Lipschitzness bounds the error term: `τ²‖F(w)-F(z_t)‖² ≤ τ²L²‖w-z_t‖²`, so it
becomes `(τ²L² - 1)‖w - z_t‖² ≤ 0` for `τ ≤ 1/L`. Hence

```
‖z_{t+1} - z*‖² ≤ ‖z_t - z*‖² - 2τ⟨F(w), w-z*⟩ - (1 - τ²L²)‖w - z_t‖²,
```

Fejer-decreasing until the solution. **Step-size rule `τ ≤ 1/L`** is forced by this inequality.
- Merely monotone (`μ = 0`): the averaged iterate `ẑ_t = (1/t)Σ w_k` drives the VI gap/merit
  function down at `O(1/t)` in the deterministic bounded-domain setting.
- `μ`-strongly monotone: `-2τ⟨F(w), w-z*⟩ ≤ -2τμ‖w-z*‖²` gives linear (geometric) last-iterate
  convergence.

## Stochastic case (SEG) — same-sample is essential

Under noise, use the **same** sample `ξ` for both evaluations in a step:
`w = prox_{ηg}(z - ηF(z; ξ))`, `z_{t+1} = prox_{ηg}(z - ηF(w; ξ))`. Two *independent* samples
(the old stochastic Mirror-Prox choice) break the implicit-step approximation — predictor and
corrector then concern different operators — and the method diverges on stochastic bilinear.
Same-sample keeps the predictor-corrector logic intact. With `F(·;ξ)` a.s. monotone and
`L`-Lipschitz, `g` `μ`-strongly convex, variance at the optimum `E‖F(z*;ξ) - F(z*)‖² ≤ σ²`, and
`η ≤ 1/(2L)`:

```
E‖z_t - z*‖² ≤ (1 - 2ημ/3)^t ‖z_0 - z*‖² + 3ησ²/μ.
```

Geometric contraction down to an `O(ησ²/μ)` neighborhood (vanishing as `η → 0`; the clean linear
rate is recovered when `σ = 0`). The slightly tighter ceiling `η ≤ 1/(2L)` leaves enough of the
`-‖w - z_t‖²` surplus to absorb the noise cross-term `E⟨F(z*)-F(z*;ξ), w-z*⟩ ≤ ησ² +
(1/4η)E‖w-z_t‖²` (Young). On an unconstrained box (`prox = identity`) this reduces to the plain
two-step update with additive update noise.

## Final algorithm

```
input: z_0, step size τ (≤ 1/L); operator F (or stochastic F(·;ξ)); optional prox_{ηg}
for t = 0, 1, 2, ...:
    w        = prox_{τg}( z_t - τ F(z_t; ξ_t) )          # predictor / look-ahead
    z_{t+1}  = prox_{τg}( z_t - τ F(w;   ξ_t) )          # corrector, anchored at z_t, same ξ_t
return z_T   (or averaged ẑ_T = (1/T) Σ w_k in the merely-monotone case)
```

Two operator evaluations per iteration; fully explicit; no inner solve.

## Relation to prior methods

- **Sim-GDA** = drop the corrector (one forward step): diverges on rotational fields.
- **Proximal point / backward step** = the implicit ideal `(I + τF)^{-1}` it imitates: contracts
  unconditionally but requires a nonlinear solve per step. EG = its `O(τ²)` explicit approximation.
- **Two-sample stochastic Mirror-Prox** = same two-step shape but independent samples per
  evaluation: breaks the approximation, diverges on stochastic bilinear. SEG fixes it with the
  same-sample rule.

## Working code

Fills the `step` slot of the saddle-point harness: `oracle.grad(z) = F(z)` (deterministic),
`oracle.noise()` (additive Gaussian update perturbation), unconstrained feasible set so no
projection. Same deterministic operator for both evaluations; the benchmark stochasticity is
additive update noise, not independent sampled operators.

```python
from typing import Any
import numpy as np

from fixed_benchmark import (
    ProblemSpec, StepOutput, StochasticOracle,
    as_vector, make_step_output, run_cli,
)


def init_state(
    problem: ProblemSpec,
    initial_z: np.ndarray,
    seed: int,
    hyperparameters: dict[str, Any],
) -> dict[str, Any]:
    return {"z": as_vector(initial_z, expected_dim=2 * problem.dim), "step_index": 0}


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

    g = oracle.grad(z)                       # F(z_t)
    w = z - tau * g + oracle.noise()         # predictor: w = z_t - tau F(z_t) + noise
    gw = oracle.grad(w)                      # F(w), same deterministic operator
    z_next = z - tau * gw + oracle.noise()   # corrector: z_{t+1} = z_t - tau F(w) + noise

    metric_iterate = z_next if problem.name == "bilinear" else z
    return make_step_output(
        {"z": z_next, "step_index": step_index + 1},
        metric_iterate,
        2,                                   # two operator evaluations per iteration
    )


def get_hyperparameters(problem_name: str, sigma: float) -> dict[str, Any]:
    if problem_name == "bilinear":
        return {"tau": 0.1}      # tau < 1/L = 1 with margin; rotation field needs small tau
    if problem_name == "delta_nu":
        return {"tau": 1.0}      # ~1-Lipschitz monotone field; tau at the stability boundary
    raise KeyError(f"Unknown problem: {problem_name}")


if __name__ == "__main__":
    run_cli(init_state=init_state, step=step, get_hyperparameters=get_hyperparameters)
```
