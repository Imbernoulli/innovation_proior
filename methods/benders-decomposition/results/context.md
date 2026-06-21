## Research question

The setting is a large optimization model whose variables do not all play the same role. Some variables are "complicating": they carry discrete choices, long-horizon commitments, network design decisions, or first-stage plans. Once those choices are fixed, the remaining variables often form a much easier linear program: flows can be routed, production can be dispatched, recourse can be purchased, or capacities can be used.

The research question is not merely how to split a big model into two smaller models. It is sharper: can we optimize over the complicating variables without explicitly carrying every continuous recourse variable and every constraint that those variables imply? Benders decomposition answers yes when the residual subproblem has a useful dual. The master problem chooses the complicating variables and a placeholder for the recourse value; the subproblem checks that choice and returns a dual certificate. That certificate becomes a cut added to the master.

The conceptual problem is therefore one of projection. If the full model is written in variables `(x, y)`, Benders tries to solve the projected problem in `x` alone. The cost and feasibility effects of the eliminated `y` variables are not discarded; they are represented by a value function that is gradually exposed by cuts.

## Background

Benders decomposition is most naturally seen in a model such as

```text
minimize    c^T x + q^T y
subject to  x in X
            T x + W y >= h
            y >= 0.
```

Here `x` is the master decision and `y` is the subproblem decision. If `x` is fixed at `xbar`, the remaining problem is

```text
phi(xbar) = min q^T y
            s.t. W y >= h - T xbar, y >= 0.
```

The important object is `phi(x)`, the optimal value of the eliminated subproblem as a function of the master decision. For a linear subproblem, duality gives

```text
phi(x) = max (h - T x)^T u
         s.t. W^T u <= q, u >= 0.
```

Each feasible dual vector `u` gives a valid affine lower bound on `phi(x)`. An optimal dual extreme point at the current `xbar` gives an optimality cut. If the subproblem is infeasible, a dual ray or Farkas certificate gives a feasibility cut excluding `xbar` and possibly many similar master decisions.

The master problem therefore carries a variable such as `eta` and only a partial description of the value function:

```text
minimize    c^T x + eta
subject to  x in X
            eta >= affine dual cuts seen so far
            feasibility cuts seen so far.
```

This master is an outer approximation of the projected problem. Solving it proposes an `x`; the subproblem either prices that proposal correctly or proves it impossible; the resulting cut refines the approximation.

## Baselines

The most direct baseline is to solve the full model in `(x, y)` at once. That keeps all constraints visible, but it can be impractical when `x` is mixed-integer, when the recourse block is huge, or when many scenarios create repeated copies of the same subproblem structure.

Another baseline is manual variable elimination or explicit projection. In principle, one can eliminate `y` and write the exact feasible region and recourse function in `x`. In practice, the projected formulation may have a very large number of inequalities. Benders can be understood as delayed projection: it does not write the full projection first; it asks the subproblem for only the facets or supporting planes that the current search has made relevant.

A third baseline is plain cutting-plane optimization without exploiting structure. Benders is more specific. Its cuts are not arbitrary violated inequalities found by a generic separator; they come from the dual of a meaningful operational subproblem. The cut has economic content: it says what shadow prices the recourse system assigns to the resources or requirements induced by the current master decision.

Compared with heuristically fixing `x` and repairing `y`, Benders keeps global logic. A bad master decision is not merely rejected; the dual certificate explains a whole affine region of decisions that are too cheap or infeasible. Each iteration leaves behind reusable evidence.

## Evaluation settings

Benders decomposition is most compelling when the master variables are few or structurally difficult and the subproblem is large but tractable. Classic examples include facility location with transportation subproblems, network design with flow subproblems, unit commitment with dispatch subproblems, and two-stage stochastic linear or mixed-integer programs where each scenario creates a recourse subproblem.

The key metrics are not just wall-clock time. Useful evaluation asks whether the method reduces the effective model size, whether the generated cuts are strong enough to close the gap quickly, and whether the master problem remains manageable as cuts accumulate. In integer master settings, the interaction with branch-and-cut also matters: cuts can be generated at integer incumbent solutions, at fractional nodes, or in a branch-and-Benders-cut scheme.

Failure modes are also part of the evaluation. Weak cuts can cause many iterations. Degenerate subproblems can return unstable dual solutions. Integer subproblems break the clean LP dual story unless generalized Benders, logic-based Benders, no-good cuts, or problem-specific inference replaces the simple dual certificate. Poorly scaled cuts can make the master numerically fragile.

## Code framework

A minimal Benders loop has three moving parts: a restricted master, a subproblem solver, and a cut manager.

```python
def solve_benders(master_model, build_subproblem, tol=1e-6, maxiter=1000):
    cuts = []
    lower_bound = float("-inf")
    upper_bound = float("inf")

    for _ in range(maxiter):
        x, eta, master_value = master_model.solve(cuts)
        lower_bound = master_value

        result = build_subproblem(x).solve_with_dual()
        if result.status == "infeasible":
            cuts.append(result.feasibility_cut())
            continue

        recourse_value = result.objective_value
        upper_bound = min(upper_bound, master_model.first_stage_cost(x) + recourse_value)
        cuts.append(result.optimality_cut(eta_name="eta"))

        if upper_bound - lower_bound <= tol:
            return x, upper_bound, cuts

    return None, upper_bound, cuts
```

The distinctive point is the direction of information flow. The master does not ask the subproblem for a primal repair and then forget it. It asks for a dual certificate that becomes a permanent statement about the projected problem. Over time, those certificates form a piecewise-linear approximation to the hidden recourse value function.
