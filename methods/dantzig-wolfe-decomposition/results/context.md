## Research question

Large-scale linear programs are often not simply a matter of "many variables" — the variables split into several natural blocks, each block carrying its own complex but local feasibility constraints, with the blocks coupled together through only a small number of resource, demand, or balance constraints. The typical form is

`min sum_k c_k^T x_k`

subject to

`sum_k A_k x_k = b`, and `x_k in X_k`.

Here `X_k = {x_k: B_k x_k <= d_k, x_k >= 0}` is the feasible region belonging to subsystem `k` alone, and `sum_k A_k x_k = b` is the global coupling constraint. Putting all the `x_k` into one LP gives a correct model, but the solver then has to face the internal structure of every block, all the variables, and all the coupling relations simultaneously. The research question is: for an LP with this block-angular structure, can global coordination be handled algorithmically separately from each block's own internal feasibility?

## Background

The standard solving tools for linear programming are the simplex method and interior-point methods, both of which work directly on the original coordinates `x`, together with LP duality theory: each equality constraint corresponds to a dual price (shadow price), and at optimality the primal and dual solutions satisfy complementary slackness, with the dual solution constituting a certificate of optimality.

A related classical geometric fact is the extreme-point representation of a convex polyhedron: if `X_k` is a nonempty bounded polyhedron, then every `x_k in X_k` can be written as a convex combination of its extreme points `p_kr`:

`x_k = sum_r lambda_kr p_kr`, `sum_r lambda_kr = 1`, `lambda_kr >= 0`.

Within a single block, when optimizing a linear objective, the optimum must be attained at some extreme point, so under a fixed linear objective, optimizing over the sub-polyhedron `X_k` is itself a small LP that simplex can solve directly. The number of extreme points of a polyhedron `X_k` can in general grow exponentially with dimension.

## Baselines

Solving the original LP directly: hand all the original variables and constraints at once to a simplex or interior-point solver. The model is straightforward, but the solver has to handle the local block structure and the global coupling structure at the same time.

Explicit enumeration: first enumerate all the extreme points of each `X_k`, rewrite the problem in terms of extreme-point weights, and solve that. This formulation is equivalent to the original LP, but it depends on being able to list the full set of extreme points.

Block-wise heuristic: solve each block's local optimum independently, then use some patching step to satisfy the global coupling constraints. This is cheap computationally — each block is solved for its own local objective, and coordination happens afterward.

Benders decomposition is a neighboring decomposition idea: it typically fixes part of the variables, generates cuts in the subproblem that feed back to the master, and belongs to row generation — what gets added dynamically are constraints.

## Evaluation settings

Instances suited to this decomposition should have block-angular structure: several relatively independent sub-blocks `X_k` connected by a small number of linking constraints. Common examples include cutting stock, vehicle routing set-partitioning formulations, crew scheduling, multi-commodity flow path models, production planning, and large resource-allocation models.

Key evaluation metrics include: how many structural plans are handled during solving, whether the subproblem optimization is cheaper than scanning the original variables directly, how quickly the dual bound converges, and whether the resulting LP bound is strong. If the original problem contains integer variables, one usually also needs to evaluate the number of search-tree nodes, the integrality gap, and the difficulty of the subproblems.

A small demonstration can set up two subproblem blocks, each with its own local polyhedron `X_1, X_2`, coupled by a single shared resource constraint `A_1 x_1 + A_2 x_2 = b`. When the block-level objective is linear, the optimization over each `X_k` is

`min_{x in X_k} (c_k - A_k^T pi)^T x`,

where `pi` is the dual price of the coupling constraint.

## Code framework

There are three kinds of components available. First, an LP solver: it takes constraints and an objective as input and outputs the primal solution and dual prices `pi`. Second, a within-block linear optimization oracle: given each block's linear objective `c_k - A_k^T pi`, it optimizes over `X_k` and returns an extreme-point solution. Third, an outer-loop skeleton that ties the global LP together with the within-block optimization.

```python
def solve_lp(columns, b):
    """Return an LP solution plus dual prices pi and mu_k."""
    raise NotImplementedError

def optimize_block(block, pi, mu_k):
    """Solve min (c_k - A_k.T @ pi)^T x over X_k and return the extreme-point solution."""
    raise NotImplementedError

def coordinate(blocks, b, initial, eps=1e-8):
    state = list(initial)
    while True:
        sol = solve_lp(state, b)
        updates = []
        for block in blocks:
            r = optimize_block(block, sol.pi, sol.mu[block.id])
            # accept r into state based on its value relative to current duals
            updates.append(r)
        if converged(updates, eps):
            return sol, state
        state.extend(u for u in updates if accept(u))
```

This report focuses on the most common case, and the one that most clearly illustrates the core structure — the case where the subproblem `X_k` is bounded.
