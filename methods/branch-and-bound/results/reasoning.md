Let me start from the thing that actually blocks me. I have a linear program, `max c'x` subject to `Ax <= b`, `x >= 0`, and the simplex method solves it cold — vertex to vertex, pivot by pivot, global optimum, done. But the problem in front of me has a catch: some of these variables have to come out *whole*. The number of tankers I dispatch, the number of machine setups I run, a yes/no switch on whether to build a depot. Three-and-seven-tenths tankers is not a thing. So I want `max c'x` subject to `Ax <= b`, `x >= 0`, and `x_j` integer for the marked `j`. And the moment I write "integer," simplex is the wrong tool, because the feasible set just stopped being convex.

Let me make sure I actually believe that, because everything downstream hangs on it. The continuous feasible set is the polytope `{Ax <= b, x >= 0}` — convex, simplex's home turf. The integer feasible set is the *lattice points inside that polytope*. Take two of them, say `(2,3)` and `(4,5)`; their midpoint `(3,4)` happens to be integer here, but take `(2,3)` and `(3,3)` — midpoint `(2.5, 3)`, not integer, not in the set. So the integer feasible set is generically non-convex: a cloud of isolated points. Simplex assumes one convex region and walks to its optimal vertex. There is no vertex to walk to here; there's a scatter of points. And any method that only inspects the neighbourhood of a candidate can certify at most a *local* optimum, because the discrete set has no global convex structure to lean on. So I can't just point simplex at this.

The brute-force escape is to enumerate. List every integer point in the box, check feasibility, keep the best. It's exact. It's also dead on arrival: `n` binary variables give `2^n` points, and general integers give far more. Forty binaries and I'm enumerating a trillion points. Worse, it's *stupid* enumeration — most of the lattice is obviously terrible, yet I'd look at all of it. I need to not look at the parts that can't possibly win. Which means I need, for chunks of the lattice at a time, a cheap way to prove "nothing in here beats what I already have." A bound. The question becomes: where do I get cheap bounds on a discrete maximization?

Here's the lever. The integer feasible set is a *subset* of the continuous polytope — it's exactly the polytope's lattice points, nothing outside. And there's a fact I trust completely: enlarging the feasible set can only raise a maximum. If `F ⊆ F'`, then `max_{F'} c'x >= max_F c'x`, because every competitor in the smaller problem is still available in the larger one, plus possibly better ones. So if I *drop* the integrality requirement — solve the LP over the whole polytope, integers be damned — I get a value that is `>=` the true integer optimum. The continuous relaxation *over-estimates* the integer answer. For a maximization, the LP value is an **upper bound** on the integer optimum. (If I were minimizing, the same enlargement *lowers* the minimum, so the LP value would be a *lower* bound. I'll keep maximization straight and flip carefully if I ever need min.)

That's the whole seed. The relaxation is cheap (simplex), convex, and gives me a *certified ceiling* on what any integer point in this region can achieve. A ceiling is exactly the proof-of-impossibility I wanted for pruning.

So step zero: solve the LP relaxation, ignore integrality. Call the optimal point `x*` and value `z_LP`. Two cases. If `x*` happens to already be integer in all the marked variables — wonderful. It's feasible for the integer problem (it satisfies every constraint *and* the integrality), and it achieves the upper bound `z_LP`, which no integer point can exceed. So it's *optimal*, immediately, no search. That actually happens for nicely structured problems — transportation-type LPs come out integral on their own. But in general some marked `x*_j` is fractional, say `x*_j = 2.6`. Now what?

The fractional value is the crack I pry on. The true integer optimum has *some* integer value for `x_j`. Whatever it is, it's either `<= 2` or `>= 3` — because there is no integer strictly between 2 and 3. The open strip `2 < x_j < 3` contains no integer point at all. So if I split the problem into two subproblems,

one with the extra constraint `x_j <= 2`, the other with `x_j >= 3`,

I have lost *nothing*: every integer-feasible point of the original lands in exactly one of the two children (or, if exactly integer on the boundary, in one of them), and the only thing I threw away is the open strip, which held no integer point to begin with. The two children's integer-feasible sets union back up to the original's. This is the *branch*. In general: pick a fractional marked variable `x*_j`, and form the two children `x_j <= floor(x*_j)` and `x_j >= ceil(x*_j)`.

And notice what the split bought me beyond completeness. The parent's LP optimum `x*` had `x_j = 2.6`. In the `x_j <= 2` child, `x* ` is *infeasible* — it's been cut out. In the `x_j >= 3` child, also infeasible. So when I re-solve each child's relaxation, neither can return `x*` again; each child's LP optimum is a *different*, and necessarily no-better, vertex. The bound can only tighten as I descend. The branching makes genuine progress — I'm not going to loop forever re-finding 2.6.

Each child is, structurally, the same kind of object I started with: a linear program with marked-integer variables, just with one variable's bounds tightened. (And in a bounded-variable LP, "`x_j <= 2`" isn't even a new row — it's just lowering `x_j`'s upper bound. The child differs from the parent by *one bound*, which is exactly the situation the dual simplex warm-starts from cheaply.) So I recurse. Solve the child's relaxation for its own upper bound; if still fractional, branch again. A tree of LP relaxations, each node a subproblem, each edge a tightened integer bound. The root is the bare relaxation; the leaves, if I let them grow, are subproblems pinned down to integer points.

Now I have to make the tree *finite and cheap*, which is where the bound earns its keep — otherwise this is enumeration wearing a tree costume. I need to keep track of the best *integer-feasible* solution I've actually found anywhere in the tree so far. Call it the **incumbent**, with value `z_inc`. Every time a node's relaxation comes back integral, that's a real feasible integer point — a candidate. If its value beats `z_inc`, it becomes the new incumbent. The incumbent is a *lower* bound on the true optimum `z*` (I have an integer point achieving it, so `z* >= z_inc`), while any open node's relaxation value is an *upper* bound on what that node's subtree can achieve. Those two bounds, squeezing from opposite sides, are the engine.

Because here is the prune. Take any node, solve its relaxation, get upper bound `u`. If `u <= z_inc` — if the very *best* that this whole subtree could conceivably achieve is no better than an integer solution I already hold — then there is no reason to explore the subtree at all. Every integer point under this node is bounded above by `u`, and `u` doesn't beat the incumbent, so none of them can be the optimum. *Fathom* the node: discard it, unexplored, with a proof that nothing was lost. The relaxation bound did the pruning — it killed an entire subtree by solving one LP. That is the difference between this and enumeration: I never open the parts of the lattice that the bound certifies can't win.

So at each node I have three ways to stop without branching — three *fathoming* conditions. First, **infeasible**: the relaxation has no solution (the tightened bounds contradict the constraints), so the subtree is empty; drop it. Second, **bound**: the relaxation value `u <= z_inc`; the subtree can't beat the incumbent; drop it. Third, **integral**: the relaxation optimum is already integer in all marked variables; it's a leaf — update the incumbent if it's better, then stop, because a relaxation that's already integral *is* the best the subtree offers, there's nothing finer to find below. Only if none of these fire — feasible, bound strictly above incumbent, and fractional — do I branch.

Let me make sure the logic is airtight, because it's easy to fool myself on the bound direction. Maximization. Relaxation value = upper bound (enlarged feasible set raises the max). Incumbent = lower bound (an actual achieved integer value). Prune when `upper_bound_of_node <= incumbent`. Yes — for a *max*, you discard a node when its ceiling is at or below your floor. (Flipped for min: relaxation gives a *floor*, incumbent gives a *ceiling*, prune when node's floor `>=` incumbent.) I'll commit to max and feed `-c` to a minimizing LP solver, negating the value on the way back, so the bookkeeping stays in one convention.

There's a subtlety I should pin down: completeness. Does this tree provably reach the optimum, or could it skip it? At every branch the two children's integer-feasible sets union to the parent's, and the only thing excluded is an integer-free strip — so no integer point ever falls out of the tree. Every integer-feasible point lives in some leaf. I only ever fathom a node when (a) it's empty, (b) its *upper bound* — which dominates every integer point beneath it — is `<= z_inc`, meaning those points can't beat what I hold, or (c) it's already an integral leaf I've evaluated. So I never discard a node that could contain a *strictly better* integer point than my incumbent. When the search ends — every node fathomed — the incumbent has been compared, directly or by bound, against all of them. It's optimal. (Assuming the variables are bounded so the tree is finite; integer variables in a box give a finite lattice, so depth is bounded.)

While the search is *running*, I also get a live certificate of how close I am — the **optimality gap**. The incumbent `z_inc` is the best integer value found; the largest upper bound over all *still-open* nodes, call it `z_bar`, is the best anything unexplored could possibly do. So the true optimum is trapped: `z_inc <= z* <= z_bar`. The gap is `z_bar - z_inc` (or relative, `(z_bar - z_inc)/|z_bar|`). When it hits zero, I've *proved* optimality — every open node's ceiling has dropped to the incumbent's floor. And the gap gives me a principled early stop: if I only need a solution provably within 1% of optimal, I halt when the relative gap drops below 1%, long before the tree is exhausted. That's huge in practice — the last fraction of the gap is often the most expensive to close.

Now, *which* open node do I expand next, and *which* fractional variable do I branch on? These are choices, and they trade off differently. Take node selection first. One option, **best-first**: always expand the open node with the largest upper bound — the most promising ceiling. This tends to minimize the number of nodes explored, because you never waste effort on a node once a better incumbent has dropped every competitor's ceiling below it; you go straight at whatever could still hold the optimum. The cost: you may keep a huge *frontier* of open nodes alive in memory, each with its stored bounds and basis. The other option, **depth-first**: dive — always expand a child of the node you just made, plunging to a leaf. The memory is tiny, just the bounds along one root-to-leaf path. And diving has a second virtue: it reaches integer leaves *fast*, so you get an incumbent *early*, and an early incumbent means the bound-prune starts firing sooner on everything else. The cost: without a good incumbent guiding you, you might dive into and explore subtrees a best-first search would have pruned outright. In practice you mix them — dive to get an incumbent, then let the bounds steer — but the two pure strategies are the poles. (Doing this by hand, with paper for storage, you'd naturally pursue a branch until its bound was no longer the best and then set it aside — a best-first flavour; on a computer with finite memory, depth-first's small footprint usually wins, and you sprinkle in diving.)

Branching variable: there are several fractional marked variables to choose from; which one splits the problem most usefully? A clean default is **most-fractional** — pick the `x*_j` whose fractional part is closest to `0.5`, i.e. maximize `|x*_j - round(x*_j)|`. The intuition: a variable sitting at, say, `2.5` is maximally *undecided* — the relaxation has no opinion which way it should go — so forcing it either way (`<= 2` or `>= 3`) is a big, genuine commitment that should move the child relaxations substantially and tighten both bounds hard. A variable at `2.95` is almost decided already; branching on it barely changes anything in one child and is nearly redundant. (There are sharper rules — estimate, per candidate, how much the objective would *drop* in each child and pick the variable promising the biggest drop, i.e. pseudocost/strong branching — but those need either history or extra LP probes; most-fractional is the simple, no-bookkeeping choice and it's a fine default.)

So the method has assembled itself out of three moves I can name: *bound* — solve the LP relaxation at each node for a certified ceiling; *branch* — when the relaxation is fractional in a marked variable, split on `x_j <= floor` vs `x_j >= ceil`, losing no integer points; and *prune/fathom* — kill any node that's infeasible, or whose ceiling can't beat the incumbent, or that's already an integral leaf. The incumbent rises, the open bounds fall, the gap closes, and when it shuts the incumbent is provably optimal. Divide the discrete space, conquer each piece with a cheap convex bound, and let the bound throw away everything that can't matter.

One loose thread before I code it. The relaxation bound is only as good as the relaxation is *tight* — if the continuous polytope bulges far past the integer hull near a fractional vertex, the ceiling `u` is loose, it rarely drops below the incumbent, and I branch a lot. Can I tighten a node's bound *before* branching? At a node with a fractional optimum `x*`, I can derive a valid linear inequality — a *cut* — satisfied by every integer-feasible point but *violated by `x*`* (a Gomory cut, read off the simplex tableau, is exactly such a thing). Add it to the node's LP and re-solve: the polytope shrinks toward the integer hull, `x*` is sliced off, and the new relaxation optimum is lower — a *tighter* ceiling, with no integer point lost. Cuts alone converge painfully slowly and stall on real instances, but *inside* the branch-and-bound tree, used to sharpen a loose node bound before I resort to branching, they're potent — fewer nodes, harder pruning. That hybrid — bound, then *cut* to tighten, then branch only if still needed — is **branch-and-cut**, and it's what the bound-tightening loose thread leads to. I'll leave the cut generation as the obvious extension and build the pure bound-and-branch engine first.

Let me write it. The pieces map one-to-one: a function that solves a node's relaxation (the LP solver, fed `-c` to maximize), a stack of open nodes where each node is just the per-variable bound vector (so branching = copy the bounds and tighten one), the incumbent and its value carried across the search, the most-fractional pick, and the three fathoming tests. Then I'll demonstrate on a 0/1 knapsack and a small general-integer LP and check both against brute-force enumeration, since on tiny instances I can afford the ground truth.

```python
import numpy as np
from scipy.optimize import linprog
import math, itertools


def branch_and_bound(c, A_ub, b_ub, n, int_vars=None, bounds=None, tol=1e-6):
    """Maximize c'x s.t. A_ub x <= b_ub, bounds l_j<=x_j<=u_j, x_j integer for j in int_vars.
    LP-relaxation tree search: bound by the relaxation, branch on a fractional variable,
    fathom by infeasibility / bound / integrality, return a provably optimal integer point."""
    if int_vars is None:
        int_vars = list(range(n))
    if bounds is None:
        bounds = [(0, None)] * n

    best_val = -np.inf            # incumbent value = LOWER bound on the optimum (z_inc)
    best_x = None                # incumbent integer-feasible point

    def relax(bnds):
        # node relaxation: drop integrality, solve the LP. linprog MINimizes,
        # so pass -c and negate -> a maximization value = UPPER bound for this node.
        res = linprog(-np.asarray(c, float), A_ub=A_ub, b_ub=b_ub,
                      bounds=bnds, method="highs")
        if not res.success:
            return None, None    # infeasible node -> empty subtree
        return res.x, -res.fun   # (vertex, node upper bound)

    stack = [list(bounds)]       # open nodes = per-variable bound vectors; depth-first
    nodes = 0
    while stack:
        bnds = stack.pop()
        nodes += 1
        x, ub = relax(bnds)
        if x is None:            # FATHOM: infeasible
            continue
        if ub <= best_val + tol: # FATHOM by bound: ceiling can't beat incumbent floor
            continue
        # most-fractional marked variable: the one nearest a half-integer
        frac, j = max((abs(x[k] - round(x[k])), k) for k in int_vars)
        if frac <= tol:          # relaxation already integral -> leaf candidate
            if ub > best_val + tol:
                best_val, best_x = ub, x.copy()   # update incumbent
            continue
        lo, hi = bnds[j]
        down = list(bnds); down[j] = (lo, math.floor(x[j]))  # x_j <= floor(x*_j)
        up   = list(bnds); up[j]   = (math.ceil(x[j]), hi)   # x_j >= ceil(x*_j)
        stack.append(down)       # the open strip floor<x_j<ceil holds no integer:
        stack.append(up)         # the two children lose no integer-feasible point

    return best_x, best_val, nodes


if __name__ == "__main__":
    # ---- 0/1 knapsack: maximize value, total weight <= capacity ----
    vals = np.array([8, 11, 6, 4, 7, 3])
    wts  = np.array([5,  7, 4, 3, 5, 2])
    cap  = 14
    n = len(vals)
    x, val, nodes = branch_and_bound(
        c=vals, A_ub=wts.reshape(1, -1), b_ub=[cap],
        n=n, int_vars=list(range(n)), bounds=[(0, 1)] * n)
    print("B&B  :", x.round().astype(int), "value", val, "nodes", nodes)

    best = (-1, None)            # brute-force ground truth (tiny instance)
    for combo in itertools.product([0, 1], repeat=n):
        a = np.array(combo)
        if wts @ a <= cap and vals @ a > best[0]:
            best = (int(vals @ a), a)
    print("brute:", best[1], "value", best[0])
    assert abs(val - best[0]) < 1e-6
    print("knapsack matches brute force")

    # ---- general-integer LP (relaxation is fractional, so branching fires) ----
    c2  = [4, -1]
    A2  = [[7, -2], [0, 1], [2, -2]]
    b2  = [14, 3, 3]
    x2, v2, nodes2 = branch_and_bound(c2, A2, b2, n=2,
                                      int_vars=[0, 1], bounds=[(0, None), (0, None)])
    print("B&B  :", x2, "value", v2, "nodes", nodes2)
    best2 = (-1e9, None)
    for a in range(0, 11):
        for bb in range(0, 4):
            p = np.array([a, bb])
            if all(np.array(A2) @ p <= b2):
                if c2[0]*a + c2[1]*bb > best2[0]:
                    best2 = (c2[0]*a + c2[1]*bb, p)
    print("brute:", best2[1], "value", best2[0])
    assert abs(v2 - best2[0]) < 1e-6
    print("integer-LP matches brute force")
```

The causal chain, end to end: the integer feasible set is the lattice inside a polytope — non-convex, so simplex can't touch it directly, and exponentially large, so enumeration is hopeless; but dropping integrality gives the LP relaxation, whose optimum *over-estimates* the integer maximum and so hands me a *certified upper bound* cheaply at any subproblem. If that relaxation is already integer I'm done; otherwise some `x*_j` is fractional, and since no integer lies in `(floor, ceil)` I split into `x_j <= floor(x*_j)` and `x_j >= ceil(x*_j)`, losing no integer point while cutting the fractional vertex out of both children so the bound tightens as I descend. I carry the best integer point found — the incumbent, a *lower* bound — and fathom any node that's infeasible, or whose relaxation ceiling can't beat the incumbent, or that's already integral; the bound, not enumeration, throws away the subtrees that can't contain the optimum. The incumbent rises, the open ceilings fall, and the gap `z_bar - z_inc` between them certifies how close I am — zero gap is a proof of optimality, a small gap is a principled early stop. Best-first expansion minimizes nodes at the cost of memory; depth-first dives for a quick incumbent on a tiny footprint; most-fractional branching commits the most-undecided variable; and when the relaxation bound is loose, a valid cut that slices off the fractional vertex without removing any integer point tightens it before I branch — bound, cut, then branch. Divide the discrete space, conquer each piece with a convex bound, and let the bound prune.
