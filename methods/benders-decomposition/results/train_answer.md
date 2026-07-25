Large optimization models often mix variables that play very different roles. Some variables encode discrete strategic choices, network design decisions, capacity investments, or first-stage commitments; once those choices are fixed, the remaining variables describe flows, dispatch, recourse, or operational responses that frequently reduce to an ordinary linear program. Solving everything together hides this structure. A monolithic formulation forces the optimizer to explore the combinatorial complexity of the master variables while simultaneously carrying every continuous recourse variable and every constraint that governs them. That can be impractical when the master is mixed-integer or when many scenarios create repeated copies of the same subproblem structure. Projecting away the recourse variables is a natural idea, but it replaces an explicit model with an implicit value function and an exponentially large projected feasible region. Manual elimination or generic cutting-plane methods either produce an unmanageable number of inequalities or ignore the structural link between the master decision and the recourse subproblem, so neither is fully satisfactory.

Benders decomposition is the method that exploits this structure without writing the projection explicitly. It separates the problem into a master problem over the complicating variables and a subproblem over the recourse variables. The master chooses a candidate master decision and maintains an auxiliary variable, often called eta, that represents the best achievable recourse cost. For the fixed master decision, the subproblem evaluates the true recourse cost and, crucially, returns a dual certificate rather than only a primal response. If the subproblem is feasible, an optimal dual solution defines an affine lower bound on the recourse value function that holds for every master decision; this bound is added to the master as an optimality cut. If the subproblem is infeasible, a dual ray or Farkas certificate shows that the current master decision cannot be extended to a feasible recourse and yields a feasibility cut that removes a region of master decisions. The master is then resolved with the new cut, and the process repeats until the estimate eta agrees with the actual recourse value within tolerance.

The power of the method is that duality turns one solved subproblem into reusable global evidence. A primal recourse solution only says how to respond to the current master decision, but a dual solution states that under the current shadow prices every master decision must pay at least a certain affine amount for recourse. The master therefore learns only the parts of the hidden value function that the search encounters, building a piecewise-linear outer approximation from supporting planes and infeasibility regions. This makes Benders much more than divide and conquer: it is an evidence-driven cutting-plane algorithm for the implicit projected problem. It is most compelling when the master variables are few or structurally difficult and the subproblem is large but tractable, such as facility location with transportation subproblems, network design with flow subproblems, unit commitment with dispatch subproblems, and two-stage stochastic programs where each scenario creates a separable recourse subproblem.

The standard derivation assumes a linear subproblem so that strong duality supplies the certificates. If the subproblem is integer or nonconvex, the clean LP dual cut is no longer available and one must resort to generalized Benders, logic-based Benders, or problem-specific inference. Other practical concerns include weak cuts that cause many iterations, degenerate subproblems that return unstable dual solutions, and the interaction between generated cuts and branch-and-bound when the master is mixed-integer. Despite these limitations, the central idea remains the canonical way to avoid carrying downstream variables in the master while still respecting their cost and feasibility consequences.

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
