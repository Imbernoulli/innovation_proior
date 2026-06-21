# Context: solving convex-concave saddle points / monotone variational inequalities

## Research question

We want to solve a convex-concave saddle-point problem

```
min_x max_y  f(x, y),
```

with `f` convex in `x`, concave in `y` — the model behind two-player zero-sum games,
game-theoretic equilibria, robust/minimax learning, primal-dual formulations, and adversarial
training. The object we actually care about is the joint *gradient field* (the "game vector
field")

```
F(z) = [ ∇_x f(x, y) ; -∇_y f(x, y) ],   z = (x, y),
```

and a solution `z* = (x*, y*)` is a point where this field vanishes, `F(z*) = 0` — equivalently
a saddle point. More generally `F` need not be a gradient of any single scalar; it is just a
**monotone operator**, and we are solving the **variational inequality** "find `z*` with
`⟨F(z*), z - z*⟩ ≥ 0` for all `z`," of which `F(z*) = 0` is the unconstrained special case.

The question is how to design a first-order iteration — using only explicit operator/gradient evaluations — that drives `‖F(z)‖ → 0` on convex-concave / monotone problems.

## Background

A minimax problem is *not* a minimization problem in disguise, and the failure mode is
geometric, not statistical. The cleanest diagnostic is the scalar bilinear game `f(x, y) = x·y`.
Its game vector field is

```
F(x, y) = [ ∂f/∂x ; -∂f/∂y ] = [ y ; -x ] = J·z,   J = [[0, 1], [-1, 0]],
```

a purely *rotational* field: `J` is skew-symmetric, with eigenvalues `±i`. The unique saddle is
the origin. A field like this has no "downhill" direction toward `z*` at all — at every point it
points *around* the equilibrium, tangent to circles, never inward. This is the structural fact
that breaks naive intuition carried over from minimization: there the gradient field is the
gradient of a convex function and points (in the convex case) toward the minimizer; here the
cross-derivatives make `F` rotational, and the right generalization of "convex gradient field"
is a **monotone operator**, `⟨F(z) - F(z'), z - z'⟩ ≥ 0` for all `z, z'`. Monotonicity is the
exact analogue of convexity for `F` (for `F = ∇φ` it *is* convexity of `φ`), and it is the
weakest standard structure under which one can still hope to converge to `z*`. A strengthening,
**`μ`-strong monotonicity** `⟨F(z) - F(z'), z - z'⟩ ≥ μ‖z - z'‖²`, plays the role of strong
convexity and is what yields linear rates.

Two pre-existing facts about the world frame everything that follows.

First, an *implicit* (backward) update is unconditionally stable where the explicit (forward)
one is not. This is classical for stiff ODE integration and for monotone operators: the
**backward Euler / resolvent** step `z_{t+1} = z_t - τ F(z_{t+1})`, i.e.
`z_{t+1} = (I + τ F)^{-1}(z_t)`, is firmly nonexpansive and Fejer-monotone toward the solution
set for *any* step size `τ > 0` on a monotone operator. On the bilinear field above,
`(I + τJ)^{-1}` has
eigenvalues `1/(1 ∓ iτ)` of modulus `1/√(1 + τ²) < 1` — it spirals *inward*. The catch is in
the name: it is implicit. Computing `(I + τ F)^{-1}` means solving a nonlinear fixed-point system
in `z_{t+1}` at every iteration, which for a general operator is as hard as the original problem.

Second, the cost of evaluating `F` is the budget. We measure work in operator evaluations (call
them gradient evaluations / SFO calls); a method that needs one `F`-evaluation per step is the
gold standard, two is acceptable, an inner solve per step is not.

A diagnostic empirical observation that motivates the whole line of work: on `f(x, y) = x·y`,
simultaneous gradient descent-ascent does not merely fail to converge — it *diverges
geometrically* from any nontrivial initialization, the iterates spiraling outward, while the
implicit/proximal update on the very same problem spirals inward and converges. Reproducing this
contrast on a 2-D vector-field plot is the standard way the instability is shown.

## Baselines

**Simultaneous gradient descent-ascent (Sim-GDA).** The direct method: step against the game
field,

```
z_{t+1} = z_t - τ F(z_t).
```

For minimization (`F = ∇φ`, `φ` convex) this is just gradient descent. The update operator is `I - τJ`; on the bilinear field where `J` has eigenvalues `±i`, `I - τJ` has eigenvalues `1 ∓ iτ` of modulus `√(1 + τ²) > 1`, so `‖z_t - z*‖` grows like `(1 + τ²)^{t/2}` — the iterates spiral outward geometrically.

**Alternating gradient descent-ascent (Alt-GDA).** Update `x` then immediately use the new `x`
when updating `y`. The sequential coupling shifts the Jacobian's spectrum and keeps the iterates bounded (they cycle rather than blow up).

**Proximal point method / backward step (Martinet 1970; Rockafellar 1976).** The implicit update
`z_{t+1} = z_t - τ F(z_{t+1}) = (I + τ F)^{-1}(z_t)`. On a monotone operator it is firmly
nonexpansive for any `τ > 0`; on the bilinear game Rockafellar's saddle-point analysis gives
`r_{k+1} ≤ r_k / (1 + τ² λ_min(B^T B))`, unconditional linear contraction. It is, in a precise sense, the *ideal* method — maximally stable, step size unrestricted. Each step requires solving the nonlinear system `(I + τ F)(z_{t+1}) = z_t`, making it implicit.

**Stochastic mirror-prox variants (Nemirovski 2004; Juditsky et al. 2011).** The stochastic VI
literature also studies projection/mirror steps when only noisy samples of the operator are
available. One common model draws fresh randomness inside the iteration, so consecutive operator calls can correspond to different sampled fields.

## Evaluation settings

The natural yardsticks for a saddle-point / VI solver, all pre-existing:

- **Scalar bilinear game** `f(x, y) = x·y`, the canonical instability instance, with field
  `F(z) = [y; -x]`. A fixed step size, a fixed nonzero starting point (e.g. `(10, 10)`), and a
  fixed iteration budget. This is where Sim-GDA visibly diverges, so it is the sharpest test of
  whether a method is stable at all.
- **A worst-case `(δ, ν)` convex-concave instance** built from a known lower-bound construction
  for first-order methods: a separable monotone (saturating / soft-clamped) component
  `(1 - δ)·clip_ν(·)` coupled by a small skew-symmetric bilinear term of strength `δ`,
  in `2d` dimensions. The clip map saturates at `±ν`, making the operator monotone and
  `Lipschitz`; this is a structured hard instance rather than a toy.
- **Update noise.** Each instance is also run with additive Gaussian perturbation injected into
  the updates (a fixed noise scale `σ`), to test robustness of the dynamics to per-step noise.
- **Metric.** The *gradient-norm* / operator-norm of the recorded iterate, `‖F(z_t)‖`, tracked
  over iterations and at a fixed final iteration count (lower is better). Driving the operator
  norm to zero is the concrete certificate used by the benchmark; monotone Lipschitz VI theory
  more commonly states its basic guarantee as an `O(1/t)` ergodic gap/merit bound, with stronger
  last-iterate statements requiring additional structure.
- **Protocol.** Identical fixed step size, initialization, iteration count, RNG seed, and noise
  model across the methods being compared; gradient norms read off the same recorded iterates.

## Code framework

The harness fixes the problem, the oracle, the update-noise model, the iteration counts, the
initialization, and the metric. The solver plugs into three slots: `init_state` (seed the state
from the fixed starting point), `step` (one iteration of the update rule), and
`get_hyperparameters` (the per-problem step size). The oracle exposes a
deterministic operator evaluation `oracle.grad(z) = F(z)` and a fresh additive-Gaussian update
perturbation `oracle.noise()`; the unconstrained feasible set is all of `R^{2d}` (no projection
needed). What the iteration does with the operator evaluations — how many it takes, at which
points, and how it combines them into the next iterate — is exactly what is to be designed.

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
    # Must preserve the fixed starting point in state["z"].
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

    # Primitives available:
    #   oracle.grad(point) -> F(point)        one deterministic operator evaluation
    #   oracle.noise()     -> Gaussian vector additive update perturbation
    # TODO: the iteration we will design — combine operator evaluations into z_next,
    #       and report how many evaluations it consumed.
    z_next = z      # placeholder
    sfo_calls = 0   # placeholder: number of oracle.grad(...) calls actually made
    metric_iterate = z_next if problem.name == "bilinear" else z
    return make_step_output(
        {"z": z_next, "step_index": step_index + 1},
        metric_iterate,
        sfo_calls,
    )


def get_hyperparameters(problem_name: str, sigma: float) -> dict[str, Any]:
    # Per-problem step size.
    if problem_name == "bilinear":
        return {"tau": 0.1}
    if problem_name == "delta_nu":
        return {"tau": 1.0}
    raise KeyError(f"Unknown problem: {problem_name}")


if __name__ == "__main__":
    run_cli(init_state=init_state, step=step, get_hyperparameters=get_hyperparameters)
```
