# Benders Decomposition

## Core insight

Benders decomposition solves a structured optimization problem by separating variables into a master block and a subproblem block. The master chooses the complicating decisions, usually the discrete, strategic, or first-stage variables. Once those decisions are fixed, the subproblem checks whether the remaining operational variables can make the plan feasible and at what cost.

The unique insight is that the subproblem should return a dual certificate, not just a primal response. The dual solution produces a cut: a linear inequality that is valid for the projected problem over the master variables. Repeating this process gradually describes the value function of the variables that were removed from the master.

## Mechanism

For a fixed master decision `x`, the subproblem computes a recourse value `phi(x)`. In an LP subproblem, duality writes that value as the maximum of affine functions of `x`. Therefore:

- an optimal dual solution gives an optimality cut, tightening the lower approximation of `phi(x)`;
- a dual ray or Farkas certificate gives a feasibility cut, excluding master decisions that cannot be extended to feasible subproblem variables.

The master keeps an auxiliary variable, often `eta`, for the unknown subproblem value. It minimizes first-stage cost plus `eta` subject to all cuts generated so far. Each iteration proposes a new `x`; the subproblem either validates the estimate or adds evidence that the estimate is too optimistic.

## Why it matters

Benders is a shift from solving the full model in one monolithic formulation to learning the hidden projected model. The full `(x, y)` model is replaced by a master problem in `x` plus an evolving set of cuts. Those cuts are not heuristics: they are certificates derived from subproblem duality.

This makes the method powerful when the projection onto `x` would be too large to write explicitly. Instead of enumerating all projected inequalities or all possible recourse consequences, Benders asks for the next violated supporting plane only when the current master solution needs it.

The resulting mindset is:

```text
Do not carry every downstream variable in the master.
Project those variables away.
Use the subproblem dual to expose the hidden feasible region and value function one cut at a time.
```

That is why Benders is more than "divide and conquer." It is an evidence-driven cutting-plane method for the implicit feasible region and objective created by eliminating part of the model.

## Best use cases

Benders is most natural for facility location, network design, unit commitment, capacity expansion, and two-stage stochastic programs: problems where master decisions are hard but subproblems become LPs or otherwise separable once the master decision is fixed.

It is less direct when the subproblem is nonconvex or integer, because the clean LP dual cut is no longer available. Those cases require generalized, logic-based, or problem-specific Benders cuts.

## Compact implementation sketch

```python
def benders_solve(build_master, build_subproblem, max_iter=1000, tol=1e-6):
    cuts = []
    lower_bound = -float("inf")
    upper_bound = float("inf")

    for _ in range(max_iter):
        # Restricted master proposes complicating variables x and recourse bound eta.
        x, eta, master_val = build_master(cuts).solve()
        lower_bound = max(lower_bound, master_val)

        # Subproblem evaluates the recourse for fixed x and returns a dual certificate.
        sub = build_subproblem(x)
        result = sub.solve_with_dual()

        if result.status == "infeasible":
            cuts.append(result.feasibility_cut())
            continue

        # Optimality cut tightens the outer approximation of the recourse value.
        recourse = result.objective_value
        upper_bound = min(upper_bound, sub.first_stage_cost(x) + recourse)
        cuts.append(result.optimality_cut(eta))

        if upper_bound - lower_bound <= tol:
            return x, upper_bound, cuts

    return x, upper_bound, cuts
```
