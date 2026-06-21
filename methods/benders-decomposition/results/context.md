## Research question

The setting is a large optimization model whose variables do not all play the same role. Some variables are "complicating": they carry discrete choices, long-horizon commitments, network design decisions, or first-stage plans. Once those choices are fixed, the remaining variables often form a much easier linear program: flows can be routed, production can be dispatched, recourse can be purchased, or capacities can be used.

The research question is how to optimize over the complicating variables without explicitly carrying every continuous recourse variable and every constraint that those variables imply. If the full model is written in variables `(x, y)`, the goal is to work in the projected space of `x` alone. The cost and feasibility effects of the eliminated `y` variables cannot simply be discarded; they are summarized by a value function `phi(x)` that records the best achievable recourse cost once `x` is chosen.

## Background

The structured models in view can be written as

```text
minimize    c^T x + q^T y
subject to  x in X
            T x + W y >= h
            y >= 0.
```

Here `x` is the strategic or first-stage decision and `y` is the operational or recourse decision. If `x` is fixed at `xbar`, the remaining problem is

```text
phi(xbar) = min q^T y
            s.t. W y >= h - T xbar, y >= 0.
```

The object `phi(x)` is the optimal value of the eliminated subproblem as a function of the strategic decision. For a linear subproblem, linear programming duality gives an equivalent dual form,

```text
phi(x) = max (h - T x)^T u
         s.t. W^T u <= q, u >= 0,
```

whose feasible region does not depend on `x`. For each fixed feasible dual vector `u`, the quantity `(h - T x)^T u` is an affine function of `x`. When the dual is unbounded, the primal subproblem is infeasible for that `x`.

## Baselines

The most direct baseline is to solve the full model in `(x, y)` at once with a general-purpose LP or mixed-integer solver. This keeps all variables and constraints in a single monolithic formulation. This is the standard approach when `x` is mixed-integer, when the recourse block is large, or when many scenarios create repeated copies of the same subproblem structure.

Another baseline is explicit variable elimination or projection. In principle one can eliminate `y` and write the exact feasible region and recourse function in `x`, obtaining a formulation purely in the strategic variables before any optimization begins.

A third baseline is generic cutting-plane optimization. Here an outer-approximation model in `x` is solved, a separation routine looks for an inequality violated by the current solution, that inequality is appended, and the model is re-solved. The separator treats the feasible region abstractly rather than exploiting the operational meaning of the subproblem.

A further baseline is heuristic fix-and-repair: fix `x`, attempt to construct a feasible `y`, and if the attempt fails or is costly, adjust `x` and try again.

## Evaluation settings

This class of problems is most compelling when the strategic variables are few or structurally difficult and the operational subproblem is large but tractable. Classic examples include facility location with transportation subproblems, network design with flow subproblems, unit commitment with dispatch subproblems, and two-stage stochastic linear or mixed-integer programs where each scenario creates a recourse subproblem.

Useful metrics go beyond wall-clock time. Evaluation asks whether a method reduces the effective model size, how quickly the optimality gap closes, and whether the strategic-level model remains manageable as the computation proceeds. In integer master settings, the interaction with branch-and-cut also matters: information can be generated at integer incumbent solutions, at fractional nodes, or within a combined branch-and-cut scheme.

Several regimes are worth tracking. Degenerate subproblems can return multiple or unstable dual solutions. Integer or nonconvex subproblems break the clean LP dual story, so settings differ in whether the subproblem stays a linear program once `x` is fixed. Numerical scaling of the data affects the conditioning of both the strategic model and the subproblem.

## Code framework

A solver in this setting is organized around two coupled models: a strategic model over `x` and a subproblem over `y` for a fixed `x`. The harness exposes hooks for building each model, solving the subproblem with access to its dual solution, and exchanging information between the two.

```python
def solve(build_strategic_model, build_subproblem, tol=1e-6, maxiter=1000):
    info = []                      # information exchanged between the two models
    lower_bound = float("-inf")
    upper_bound = float("inf")

    for _ in range(maxiter):
        x, master_value = build_strategic_model(info).solve()
        lower_bound = master_value

        result = build_subproblem(x).solve_with_dual()
        if result.status == "infeasible":
            info.append(result.infeasibility_certificate())
            continue

        recourse_value = result.objective_value
        upper_bound = min(upper_bound, first_stage_cost(x) + recourse_value)
        info.append(result.dual_solution())

        if upper_bound - lower_bound <= tol:
            return x, upper_bound, info

    return None, upper_bound, info
```

The framework provides the strategic model `(c^T x` over `x in X)`, the parametric subproblem `phi(x)` with its dual, the bookkeeping for lower and upper bounds, and a place to accumulate whatever information the subproblem returns at each fixed `x`.
