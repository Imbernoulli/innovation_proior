Benders decomposition starts from a simple discomfort with monolithic optimization: the full model contains different kinds of variables, and treating them all as equally present can hide exploitable structure. The natural split is between decisions that define the situation and decisions that respond to that situation. Fix the first set, and the second set often becomes an ordinary linear program.

The decisive move is to ask what remains after the response variables are eliminated. If `x` is the master decision and `y` is the recourse or operational decision, then eliminating `y` does not make the problem simple; it creates an implicit value function `phi(x)`. This function says the best possible downstream cost, or whether downstream feasibility is even possible, for each proposed `x`.

Writing `phi(x)` explicitly is usually as hard as solving the original problem. But for a linear subproblem, its dual exposes the shape of `phi`. At a trial point `xbar`, solving the subproblem gives either an optimal dual solution or a feasibility certificate. The optimal dual solution is a supporting affine function of `phi`; the feasibility certificate is an inequality that the projected feasible region must satisfy.

That means the method is not just "solve a master and a subproblem." The subproblem is an oracle that returns evidence. The evidence is a cut: a linear statement that remains valid for all future master decisions, not only for the current one. The master accumulates these statements and becomes a progressively better model of the projected problem.

This is why Benders changes the mental model from one-shot full-model solution to learning. In the monolithic view, every variable and every constraint is present before optimization begins. In the Benders view, the master begins with an incomplete picture of the consequences of its own decisions. It proposes an `x`; the subproblem answers, "under these shadow prices, your estimate of the downstream value is too optimistic," or "this kind of `x` cannot be extended to a feasible `y`." The master is corrected by a cut and tries again.

The phrase "learning" should be read mathematically, not metaphorically. The projected feasible region and the recourse value function already exist. Benders does not invent them. It samples them through dual certificates and constructs an outer approximation. Each optimality cut is a supporting hyperplane of the value function. Each feasibility cut removes a region of master decisions that cannot be extended. The algorithm learns only the parts of this hidden object that are needed to prove optimality.

The dual is essential because it turns one solved subproblem into a global statement. A primal solution for `y` would say how to respond to the current `xbar`. A dual solution says something broader: given the current shadow prices, every `x` must pay at least a particular affine amount for recourse. That is the whole leverage of Benders. The subproblem's dual variables are not bookkeeping; they are the language in which the eliminated block speaks back to the master.

The same idea also explains the method's limits. If the subproblem is not convex, a clean dual certificate may not exist. If the value function is badly approximated by the generated cuts, the master can keep choosing optimistic points and require many iterations. If the master is integer, the cuts must interact with branch-and-bound. But the central insight remains intact: exploit structure by projecting away an easy block, then recover the consequences of that block through certificates rather than by carrying all of its variables at all times.

The most compact description is this: Benders decomposition replaces "optimize everything at once" with "optimize over the hard decisions while iteratively learning the feasible and cost consequences of the easy decisions." Its novelty is not decomposition alone, but decomposition plus dual-generated cuts that gradually reveal the value function of the variables that have been projected out.

## Minimal loop stub

```python
def benders_stub(master, build_subproblem, max_iter=1000, tol=1e-6):
    cuts = []
    lower_bound = -float("inf")
    upper_bound = float("inf")

    for _ in range(max_iter):
        x, eta, lb = master.solve(cuts)
        lower_bound = lb

        sub = build_subproblem(x)
        result = sub.solve_with_dual()

        if result.status == "infeasible":
            cuts.append(result.feasibility_cut())
            continue

        upper_bound = min(upper_bound, sub.first_stage_cost(x) + result.objective_value)
        cuts.append(result.optimality_cut(eta))

        if upper_bound - lower_bound <= tol:
            return x, upper_bound, cuts

    return x, upper_bound, cuts
```
