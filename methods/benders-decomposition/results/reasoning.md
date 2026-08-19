I keep coming back to a discomfort with solving the monolithic model `min c^T x + q^T y` subject to `Tx + Wy >= h`, `x in X`, `y >= 0` all at once. The two variable blocks do not play the same role. The `x` are the decisions that define the situation — discrete choices, first-stage commitments, network design. The `y` are the response: once `x` is fixed, routing the flow or buying the recourse is an ordinary linear program. Carrying every `y` variable and every recourse constraint in the solver, on equal footing with `x`, throws away that asymmetry. I want to optimize over `x` while somehow not dragging the whole `y` block along at every step.

So let me try to eliminate `y`. Fix `x = xbar`. What is left is

```text
phi(xbar) = min q^T y   s.t.   W y >= h - T xbar,  y >= 0.
```

`phi(xbar)` is the best downstream cost given the plan `xbar`, and the original problem is exactly `min_{x in X} c^T x + phi(x)`. That looks like progress — I have reduced the problem to optimizing one function of `x`. But `phi` is defined by an embedded LP, so writing it down in closed form over all `x` is, in general, as hard as the original problem. Eliminating `y` did not make the problem small; it hid the difficulty inside `phi`. I cannot just tabulate `phi`.

The structure of that embedded LP is what I should lean on. Its right-hand side `h - T xbar` depends on `x` affinely, but its feasible region `{y >= 0 : W y >= rhs}` changes shape with `x`, which is awkward. The dual is more cooperative:

```text
phi(x) = max (h - T x)^T u   s.t.   W^T u <= q,  u >= 0.
```

The dual feasible region `{u >= 0 : W^T u <= q}` does not depend on `x` at all. That is the property I want. For any fixed dual-feasible `u`, the value `(h - T x)^T u` is an affine function of `x`, and by weak duality it is a lower bound on `phi(x)`. So `phi` is the pointwise maximum, over a fixed polyhedron of `u`, of affine functions of `x` — a convex, piecewise-linear function expressed as an upper envelope.

I want to check this is really true before building anything on it, because everything downstream depends on two things: strong duality holding (so the value I read off the dual is the right one) and a single dual solution giving a globally valid affine bound (so one solved subproblem produces a cut good for *all* `x`, not just the current one). Let me make it concrete with a tiny instance: `x` scalar, `y in R^2`, with `c = 2`, `q = (1, 3)`, constraints `x + y1 + y2 >= 4` and `y2 >= x - 1`, i.e.

```text
T = [[1], [-1]],  W = [[1, 1], [0, 1]],  h = (4, -1).
```

Solving both the primal and dual subproblem at several `x`:

```text
x = 0:  primal phi = 4,  dual phi = 4,  u* = (1, 0)
x = 1:  primal phi = 3,  dual phi = 3,  u* = (1, 0)
x = 2:  primal phi = 4,  dual phi = 4,  u* = (1, 2)
x = 3:  primal phi = 6,  dual phi = 6,  u* = (0, 3)
x = 4:  primal phi = 9,  dual phi = 9,  u* = (0, 3)
x = 5:  primal phi = 12, dual phi = 12, u* = (0, 3)
```

Primal equals dual at every point — strong duality holds, as it must for a feasible bounded LP. And notice the optimal `u*` changes between `x = 1`, `x = 2`, and `x = 3`: the dual vertex that achieves the maximum switches, which is exactly the piecewise-linear behaviour I predicted. `phi` is V-shaped with its minimum near `x = 1`.

Now the load-bearing claim. Take the dual solution at `xbar = 1`, `u* = (1, 0)`. It defines the affine function `(h - T x)^T u* = u*^T h - (u*^T T) x = 4 - 1*x`. I am claiming this single function, harvested at one point, is a valid lower bound on `phi(x)` everywhere. Evaluating `4 - x` against the true `phi` on a grid:

```text
x = -2:  cut = 6.0,  phi = 6.0   (tight)
x =  0:  cut = 4.0,  phi = 4.0   (tight)
x =  1:  cut = 3.0,  phi = 3.0   (tight)
x =  2:  cut = 2.0,  phi = 4.0
x =  3:  cut = 1.0,  phi = 6.0
x =  5:  cut = -1.0, phi = 12.0
```

The cut never exceeds `phi`, and it touches `phi` exactly along the stretch where `u*=(1,0)` is the optimal dual vertex (`x <= 1`), peeling away below it elsewhere. So one solved subproblem really does buy a globally valid supporting hyperplane of `phi`. This is the leverage I was looking for: the dual turns a single local solve into a statement about all future `x`. A *primal* `y` would only tell me how to respond to the current `xbar`; the dual `u*` says every `x` must pay at least `u*^T h - (u*^T T)x` for recourse.

That changes how I should structure the algorithm. Instead of writing `phi` down, I keep a master problem with an auxiliary scalar `eta` standing in for `phi(x)`, and I minimize `c^T x + eta` subject to the affine lower bounds I have collected so far:

```text
eta >= u_k^T h - (u_k^T T) x   for each dual solution u_k found.
```

These are the optimality cuts. The master is an outer approximation: with few cuts, `eta` underestimates `phi`, so the master is optimistic and `c^T x + eta` is a lower bound on the true optimum. Solving the subproblem at the proposed `x` gives the true `phi(x)`, hence a feasible point with value `c^T x + phi(x)` — an upper bound. The gap between the two is the optimality gap, and each new cut at the current `x` removes precisely the optimism that let the master pick it.

One case is still open: what if the subproblem is infeasible for some `x`? Then no `y` extends the plan, and `phi(x) = +infinity`. By LP duality, primal infeasibility (with a feasible dual) means the dual is unbounded — there is a recession direction `r >= 0` with `W^T r <= 0` but `(h - T x)^T r > 0`, a Farkas certificate. Let me confirm this happens and that the ray certifies it. Restrict to a one-variable subproblem `min y1` with `y1 >= 5` and `-y1 >= -1` (so `y1 <= 1`): clearly infeasible. The solver agrees — primal status "infeasible", dual status "unbounded". The candidate ray `r = (1, 1)` satisfies `W^T r = 1*1 + (-1)*1 = 0 <= 0` and `rhs^T r = 5 + (-1) = 4 > 0`. So `r` certifies infeasibility, and the inequality `(h - T x)^T r <= 0` is a *feasibility cut*: a linear statement that excludes every `x` whose right-hand side is too demanding for the recession direction `r`. The same dual solve therefore produces either an optimality cut (bounded) or a feasibility cut (unbounded); I do not need a separate mechanism.

It is worth checking the whole loop actually closes the gap and does not just generate cuts forever. On the example instance, with the master `min 2x + eta` over `x in [0, 5]`:

```text
it 0:  x = 0,  LB = -1e6 (no cuts yet),  phi = 4,  UB = 4
it 1:  x = 0,  LB = 4,   cut eta >= 4 - x active,  phi = 4,  UB = 4,  gap = 0
```

Two iterations: the first cut pins `eta` at the true value at `x = 0`, the master re-solves to the same `x` now with `LB = UB = 4`, and it stops. The optimum is `x* = 0` (total `2*0 + 4 = 4`, cheaper than `x = 1` at `2 + 3 = 5`, even though `x = 1` minimizes recourse alone — the master correctly trades first-stage cost against recourse). The bounds met; nothing was asserted that the run did not produce.

This also exposes where the method strains, and I should be honest about it rather than claim robustness. The clean dual story needs the subproblem to be a convex program once `x` is fixed; if `y` is integer or the subproblem is nonconvex, strong duality can fail and there may be no single affine certificate — the optimality/feasibility cut machinery above does not apply unchanged, and one needs generalized or logic-based cuts instead. Degenerate subproblems return non-unique `u*`, so the cut chosen can wobble between iterations and slow convergence. And if `x` is integer, these cuts have to live inside branch-and-bound rather than a pure LP master. I have not verified behaviour in any of those regimes; what I have verified is the convex-subproblem core.

Stepping back, the shape of the method is now forced by what I checked rather than assumed. Do not carry every downstream `y` variable in the master. Project them away into `phi(x)`. Recover the cost and feasibility consequences of that projection not by writing `phi` out, but by sampling it through subproblem duals — each bounded solve an optimality cut, each unbounded solve a feasibility cut — and let the master accumulate them until its outer approximation is tight enough to prove optimality. The decomposition alone is unremarkable; the working part is that the dual certificates are globally valid, which is the one thing I made sure to compute rather than take on faith.

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
