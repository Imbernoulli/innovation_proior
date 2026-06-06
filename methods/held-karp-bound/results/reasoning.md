OK, let me think about this from the pain. I'm trying to solve the symmetric TSP to *proven*
optimality, and the only honest way I know is branch-and-bound: split the set of tours by
forcing edges in or out, and at every node of that search prune whenever a lower bound on the
best completable tour already exceeds some tour I'm holding. Everything rides on that lower
bound. If it's loose, the tree explodes and I get nowhere past ten cities. If it's tight, the
tree stays small and the whole thing is feasible. So my real problem isn't "solve TSP"; it's
"manufacture a lower bound on the optimal tour cost that's both cheap to recompute thousands of
times and close to the true optimum." Let me hunt for that.

What do I have on the shelf? The cheapest lower bound is: each city has two incident edges in
any tour, so the tour costs at least half the sum, over cities, of that city's two cheapest
edges. Trivial. And hopelessly loose — it never looks at whether those edges even connect into
one structure; it's a pile of local minima with no global coherence. Forget it as a driver.

Next up the ladder is the minimum spanning tree. Delete any single edge from a tour and what's
left is a connected spanning subgraph with `n-1` edges — a spanning tree. In the ordinary
nonnegative distance setting, that tree is no more expensive than the tour I cut it from, and
the minimum spanning tree is no more expensive than that. So MST <= any tour, hence MST <= OPT.
The MST is a genuine lower bound and I can compute it fast — Prim grows a tree one cheapest
crossing edge at a time, Kruskal adds globally cheapest non-cycle-forming edges. Good. But stare
at what an MST actually looks like: `n-1` edges, no cycle at all, and degrees all over the place
— leaves of degree 1, hub nodes of degree 5. A tour has `n` edges, exactly one cycle, and
*every* node degree exactly 2. The MST has thrown away the cycle and thrown away the degree
structure. It's structurally nothing like a tour, and that's exactly why its cost sits far below
the optimum. Loose.

So here's the tension I have to resolve. The tour itself — `n` edges, connected, every degree
2 — is intractable. The MST — `n-1` edges, connected, any degrees — is easy but too far from a
tour. I want something *in between*: a relaxation that's almost a tour, so its cost is a sharp
bound, but still easy to compute. What's the smallest weakening of "tour" that's still
polynomial?

Let me count degrees of freedom. A tour has `n` edges; the MST has `n-1`. The MST is one edge
short of having a cycle. What if I take an MST and add exactly one more edge to force a single
cycle back in? That gives me `n` edges and one cycle — closer to a tour. But where do I add it,
and why would the result be a clean lower bound I can reason about?

Let me try to be surgical. Single out one node — call it the special node, say node `1`. Pull
it out of the graph entirely, build the MST on the *other* `n-1` nodes, and then bring node `1`
back by connecting it with its two cheapest edges to that tree. Count: the MST on `n-1` nodes
has `n-2` edges, plus 2 edges for node 1, equals `n` edges total. Degrees sum to `2n`, so the
average degree is exactly 2 — like a tour. And node 1 has degree exactly 2 by construction. The
two edges I added to node 1 reconnect it to a tree that already spans everything, so they close
exactly one cycle. Connected, `n` edges, one cycle. Let me call this object a **1-tree**: a
spanning tree on all-but-one node, plus the one special node hung back on by its two cheapest
edges.

Now the crucial check — is a 1-tree's minimum cost actually a lower bound on the tour, and how
loose is it? A tour visits node 1 with exactly two edges and is a connected spanning subgraph
with `n` edges and a single cycle — so a tour *is itself a 1-tree* (the degree-2-everywhere
kind). Every tour is a 1-tree, but not every 1-tree is a tour. So tours are a *subset* of
1-trees, which means the minimum over the bigger set is at most the minimum over the smaller:
min-cost 1-tree <= min-cost tour = OPT. There it is — the cheapest 1-tree is a valid lower
bound, and I can compute it with one MST plus a sort for the two cheapest edges at node 1.
Cheap. And it's a *better* relaxation than the MST because it's forced to carry a cycle and
forced to keep node 1 at degree 2 — it's strictly closer to a tour.

But it's still not tight. The minimum 1-tree will happily make some node a high-degree hub and
others degree-1 leaves, as long as that's cheap — the only node pinned to degree 2 is node 1.
A real tour needs *every* node at degree 2. So the gap between my 1-tree and a tour is exactly
the degree violations: nodes whose 1-tree degree is 1 or 3 or 4 instead of 2. I need to push
the minimum 1-tree toward all-degrees-2 without losing the lower-bound guarantee. How?

Here's the wall. I can't just *forbid* degrees other than 2 — that would turn the relaxation
back into the tour problem, which is intractable; I'd be back where I started. I need a softer
lever. I want to *discourage* degree violations through cost, not outlaw them.

Let me think about what knob I'm allowed to turn. I'm free to change the cost matrix, *as long
as I don't change which tour is optimal*, because then my bound is bounding the same problem.
When can I perturb edge costs and leave the optimal tour unchanged? Suppose I add a number to
every edge. No good — a tour has `n` edges and a 1-tree has `n` edges too, so a uniform additive
shift just adds a constant to everything; harmless but useless, it doesn't distinguish degrees.
I need a perturbation that hits a node *in proportion to how many edges it uses*. That's the
clue: attach a price `pi_i` to each *node*, and add `pi_i + pi_j` to the cost of edge `(i,j)`.
Define the perturbed cost `c'(i,j) = c(i,j) + pi_i + pi_j`.

Now look at what this does to the cost of any subgraph. Sum `c'` over the edges of a structure
`t`: each edge `(i,j)` contributes `pi_i + pi_j`, so node `i` collects `pi_i` once for every
edge incident to it — that's `pi_i` times its degree. So

  perturbed-cost(t) = cost(t) + sum_i pi_i * deg_t(i).

Stare at this for a tour. In a tour every degree is exactly 2, so

  perturbed-cost(tour) = cost(tour) + 2 * sum_i pi_i.

The extra term `2 sum_i pi_i` doesn't depend on *which* tour I picked — it's the same constant
for all of them. So under the perturbation, every tour's cost shifts by the identical constant,
and therefore the *optimal* tour is exactly the same tour as before. The perturbation leaves
the TSP invariant. That's the freedom I was looking for: I can dial the node prices `pi`
however I like and I'm still bounding the same problem.

And here's why the prices bite on the 1-tree but not on tours: a general 1-tree does *not* have
all degrees 2, so its perturbed cost picks up `sum_i pi_i * deg_t(i)`, which is *not* the same
constant — it depends on the degree pattern. So when I compute the minimum 1-tree under
perturbed costs, raising `pi_i` on a node makes every edge touching that node more expensive,
which pushes the minimizing 1-tree to use *fewer* edges there — to lower that node's degree.
Lowering `pi_i` does the opposite. I finally have a lever on degrees that doesn't change OPT.

Let me turn this into an actual bound. Fix prices `pi`. Compute the minimum 1-tree under
perturbed cost `c'`; call its (raw, un-perturbed) cost `L` and its degree vector `deg`. Its
perturbed cost is `L + sum_i pi_i * deg_i`. Now I'll mirror the subset argument under perturbed
costs. Tours are a subset of 1-trees, so the minimum perturbed cost over all 1-trees is at most
the minimum perturbed cost over tours, which is `cost(t*) + 2 sum_i pi_i` where `t*` is the
optimal tour. So

  min over 1-trees of perturbed-cost  <=  cost(t*) + 2 sum_i pi_i.

The left side, evaluated at the actual minimizing 1-tree, is `L + sum_i pi_i deg_i`. Subtract
`2 sum_i pi_i` from both sides:

  L + sum_i pi_i deg_i - 2 sum_i pi_i  <=  cost(t*) = OPT,

that is

  w(pi) := L + sum_i pi_i (deg_i - 2)  <=  OPT.

So for *every* choice of prices `pi`, the quantity `w(pi)` — the raw 1-tree cost corrected by a
term that charges each node by its price times its degree-excess `(deg_i - 2)` — is a valid
lower bound on the optimal tour. When `pi = 0` this is just the plain minimum 1-tree, my earlier
loose bound. Nonzero prices can only help me find a *larger* bound. And notice the correction
term is beautiful: a node sitting at the right degree 2 contributes nothing; only the violators
`deg_i != 2` move the bound.

So now the obvious move: I don't want just *some* `pi`, I want the *best* one. Every `pi` gives
a lower bound `w(pi) <= OPT`; the tightest bound is

  HK := max over pi of w(pi).

This is exactly a Lagrangian dual. Look back at what I did: the constraint I couldn't enforce
directly was "every node has degree 2." I relaxed it — dropped it from the feasible set (so the
feasible set became all 1-trees, which I can optimize over) and priced its violation in the
objective with a multiplier `pi_i` per node. `w(pi)` is the relaxed minimum, a lower bound for
any prices, and maximizing over the multipliers is the dual. Driving `pi` to maximize `w` is
driving the minimum 1-tree's degrees toward 2 — toward an actual tour. If at the optimum the
minimizing 1-tree happens to come out all-degrees-2, it's a tour, and then the bound equals OPT
exactly and I've solved the instance at this node. In general it won't, and `w` sits below OPT,
but as tight as this relaxation can make it.

How do I actually maximize `w(pi)`? Let me understand the shape of `w` as a function of `pi`.
For a *fixed* 1-tree `t`, its perturbed cost `cost(t) + sum_i pi_i deg_t(i)` is *linear* (affine)
in `pi`, and so is the corrected quantity `cost(t) + sum_i pi_i(deg_t(i) - 2)`. The bound `w(pi)`
takes, at each `pi`, the *minimum* of these affine functions over all the finitely many 1-trees.
A pointwise minimum of affine functions is concave and piecewise-linear. So `w(pi)` is concave —
good, a single hill to climb, no spurious local maxima — but it has kinks: at prices where the
identity of the cheapest 1-tree switches, `w` is not differentiable.

Concave and nonsmooth. I can't just take a gradient; at the kinks there isn't one. But I don't
need the gradient — I need any *ascent-ish* direction, and a concave nonsmooth function has
subgradients (supergradients) everywhere. Take a `pi`, compute its minimizing 1-tree `t` with
degrees `deg`. Locally, for `pi` in the region where `t` stays the minimizer, `w(pi) =
cost(t) + sum_i pi_i (deg_i - 2)`, whose gradient is the vector `g` with `g_i = deg_i - 2`.
That vector `g = deg - 2` is a supergradient of `w` at `pi`: it's the gradient of the affine
piece that's currently active, and because `w` is the lower envelope, moving along `g` does not
overstate the increase. So my ascent direction is simply the degree-excess vector. I love that
it's the *same* object — `deg_i - 2` — that measures how far each node is from being tour-like
*and* tells me which way to move its price. The geometry and the constraint violation are the
same arrow.

So the update writes itself: take a step along the subgradient,

  pi_i  <-  pi_i + step * (deg_i - 2).

Read it as control. A node with degree 3 (too many edges) has `deg_i - 2 = +1`, so its price
goes *up*, its incident edges get more expensive, and the next minimum 1-tree is pushed to drop
an edge there. A node with degree 1 (too few) has `deg_i - 2 = -1`, price goes *down*, its edges
cheapen, the next 1-tree is encouraged to add an edge. A node already at degree 2 isn't touched.
Every step nudges the relaxed solution toward all-degrees-2 — toward a tour — exactly as the
dual wants.

Now the step size, and this is where I have to be careful, because naive subgradient ascent
doesn't behave like gradient ascent. With a *constant* step the iterate doesn't settle at the
maximum — `w` oscillates around it, because near the top the subgradients keep pointing across
the kink and a fixed step overshoots back and forth forever. Subgradient theory says I need a
step that goes to zero — but not too fast (if it vanishes too quickly I stall before reaching
the top). The classic prescription is a diminishing, non-summable schedule. Let me find a
concrete one that works in practice here.

The Polyak idea: if I knew the optimal value `w*`, the right step would be `step =
(w* - w(pi)) / ||g||^2`, which is the exact step to reach `w*` if `w` were a single linear
piece — it scales the move by how far below the target I am, normalized by the subgradient's
squared length. I don't know `w*`. But I *do* have, or can cheaply get, an *upper* bound `UB`
on the optimal tour — any heuristic tour's cost is one, and a Christofides tour is a good cheap
one. Held, Wolfe and Crowder's point in their validation of subgradient optimization is that
substituting an *over*-estimate `UB` for the unknown `w*` works essentially as well as the true
target and is far easier to obtain. So:

  step = lambda * (UB - w(pi)) / sum_i (deg_i - 2)^2,

with `lambda` a scalar in `(0, 2]`. The normalizer `sum_i (deg_i - 2)^2` is exactly `||g||^2`,
the squared length of my subgradient — when only a few nodes are off-degree, the steps are
larger; as the 1-tree approaches a tour and the violations shrink, the denominator shrinks too,
keeping the step alive. To get the vanishing behaviour, I start `lambda = 2`, run a batch of
iterations (say `2n`), then *halve* `lambda` and halve the batch length, and repeat, until the
batch is tiny. The halving drives the effective step to zero gradually so the iterate settles,
while the early large steps cover ground fast.

There's a second, even cheaper schedule that doesn't need an upper bound at all. Just *prescribe*
a positive step that decreases smoothly to exactly zero at a final iteration `M`. Volgenant and
Jonker fix the schedule by three conditions: the second difference of the step is constant (so
the schedule is a smooth quadratic in the iteration count), `step(M) = 0` (it dies out exactly
at the end), and `step(1) - step(2) = 3 (step(M-1) - step(M))` (the early steps are large, the
late ones small, in a fixed ratio). Solving that recurrence gives the closed form

  step(m) = [ (m-1)(2M-5) / (2(M-1)) - (m-2) + (1/2)(m-1)(m-2) / ((M-1)(M-2)) ] * step1,

where `step1` is the initial scale. A natural scale for `step1` is `L / (2n)`, the un-perturbed
1-tree cost divided by `2n` — it ties the step magnitude to the typical edge cost in the
instance so I don't have to hand-tune units — and I refresh `step1` from the current 1-tree
cost each time I find a new best `w`. Pick `M` to grow sublinearly with `n` (empirically around
`28 n^0.62` works well); more cities need somewhat more iterations but not proportionally more.
This schedule never needs `UB`, never needs to know how far below the optimum I am — it just
spends a fixed, decreasing budget — and in practice it converges nicely. I'll make it the
default and keep the upper-bound version as an alternative.

One more practical wall. Computing a minimum 1-tree means an MST on `n-1` nodes, and on a dense
cost matrix that's order `n^2` work and memory per iteration, repeated for hundreds of
iterations *and* re-run at every branch-and-bound node. That's brutal. But I don't actually need
the MST over the *complete* graph at every subgradient step if I only want good prices; cheap
edges are the ones most likely to matter. So I can restrict each MST to a sparse graph of each
node's nearest neighbours, then add the ordinary MST edges to keep that sparse graph connected.
There's a subtlety though: a 1-tree on a *restricted* graph can only cost *more* than on the
complete graph, because it minimizes over fewer choices. That restricted value is useful as a
search signal for the prices, but I must not report it as a lower bound on the original complete
problem. The certificate has to come from one final recomputation of the minimum 1-tree on the
complete graph using the best prices found. The complete-graph value is back under the subset
argument, so that final pass restores validity.

Let me assemble the whole thing. Initialize prices to zero and remember the best bound seen.
Each iteration: compute the minimum 1-tree under current perturbed costs (MST on the `n-1`
ordinary nodes plus the two cheapest edges hanging the special node back on); read off the raw
1-tree cost `L` and the degree vector; form `w = L + sum_i pi_i (deg_i - 2)`; if it beats the
best so far, record it and the prices; then step the prices along the subgradient `pi_i +=
step * (deg_i - 2)`. When the schedule runs out, take the best prices, recompute on the complete
graph, and return that certified value, never the sparse search value. If the complete-graph
1-tree at any price has all degrees equal to 2, it is a tour and the bound equals OPT exactly;
otherwise the final complete-graph value remains at most OPT and is usually far tighter than the
plain MST or unweighted 1-tree because the prices have squeezed down the degree violations.

```python
import math
import numpy as np


def _prim_mst_edges(weight, allowed=None):
    k = weight.shape[0]
    if k <= 1:
        return []
    in_tree = np.zeros(k, dtype=bool)
    best = np.full(k, np.inf)
    parent = np.full(k, -1, dtype=int)
    in_tree[0] = True
    if allowed is None:
        best[:] = weight[0]
        parent[:] = 0
    else:
        best[allowed[0]] = weight[0, allowed[0]]
        parent[allowed[0]] = 0
    best[0] = np.inf
    parent[0] = -1
    edges = []
    for _ in range(k - 1):
        masked = np.where(in_tree, np.inf, best)
        v = int(np.argmin(masked))
        if not np.isfinite(masked[v]):
            raise ValueError("candidate graph must be connected")
        u = int(parent[v])
        edges.append((u, v))
        in_tree[v] = True
        candidates = ~in_tree if allowed is None else ((~in_tree) & allowed[v])
        upd = candidates & (weight[v] < best)
        best[upd] = weight[v, upd]
        parent[upd] = v
    return edges


def _candidate_edges(cost, width):
    k = cost.shape[0]
    if k <= 1 or width is None or width <= 0 or width >= k - 1:
        return None
    allowed = np.zeros((k, k), dtype=bool)
    for i in range(k):
        added = 0
        for j in np.argsort(cost[i]):
            if i == j:
                continue
            allowed[i, j] = True
            allowed[j, i] = True
            added += 1
            if added == width:
                break
    for u, v in _prim_mst_edges(cost):
        allowed[u, v] = True
        allowed[v, u] = True
    np.fill_diagonal(allowed, False)
    return allowed


def _tour_upper_bound(cost):
    n = cost.shape[0]
    unseen = set(range(1, n))
    cur = 0
    total = 0.0
    while unseen:
        nxt = min(unseen, key=lambda j: cost[cur, j])
        total += cost[cur, nxt]
        unseen.remove(nxt)
        cur = nxt
    return total + cost[cur, 0]


def compute_one_tree(cost, pi, allowed=None):
    # Perturb by node potentials, compute the MST on n-1 ordinary nodes, then
    # attach the left-out node by its two cheapest perturbed edges.
    n = cost.shape[0]
    extra = n - 1
    weighed = cost + pi[:, None] + pi[None, :]
    sub_edges = _prim_mst_edges(weighed[:extra, :extra], allowed)
    degrees = np.zeros(n, dtype=int)
    one_tree_cost = 0.0
    for u, v in sub_edges:
        degrees[u] += 1
        degrees[v] += 1
        one_tree_cost += cost[u, v]
    to_extra = weighed[extra, :extra]
    a, b = np.argsort(to_extra)[:2]
    for v in (int(a), int(b)):
        degrees[extra] += 1
        degrees[v] += 1
        one_tree_cost += cost[extra, v]
    return one_tree_cost, degrees


def _bound_value(cost, pi, allowed=None):
    L, deg = compute_one_tree(cost, pi, allowed)
    return L + float(np.dot(pi, deg - 2)), L, deg


def held_karp_lower_bound(cost, algorithm="VJ", upper_bound=None,
                          max_iterations=0, nearest_neighbors=40):
    cost = np.asarray(cost, dtype=float)
    n = cost.shape[0]
    if n < 2:
        return 0.0
    if n == 2:
        return cost[0, 1] + cost[1, 0]

    algorithm = algorithm.upper()
    search_edges = _candidate_edges(cost[:n - 1, :n - 1], nearest_neighbors)
    pi = np.zeros(n)
    best_pi = pi.copy()
    best_search_w = -math.inf
    step1 = 0.0
    m = 0
    if algorithm == "HWC":
        if upper_bound is None:
            upper_bound = _tour_upper_bound(cost)
        num_iter = 2 * n
        it = 0
        lam = 2.0
    elif algorithm == "VJ":
        M = max(3, max_iterations if max_iterations > 0 else int(28 * n ** 0.62))
    else:
        raise ValueError("algorithm must be 'VJ' or 'HWC'")

    while True:
        if algorithm == "HWC":
            if it >= num_iter:
                num_iter //= 2
                if num_iter < 2:
                    break
                it = 0
                lam /= 2.0
            else:
                it += 1
        else:
            m += 1
            if m > M:
                break

        w, L, deg = _bound_value(cost, pi, search_edges)
        if w > best_search_w:
            best_search_w = w
            best_pi = pi.copy()
            if algorithm == "VJ":
                step1 = L / (2 * n)

        if algorithm == "HWC":
            norm = float(np.sum((deg - 2) ** 2))
            step = lam * (upper_bound - w) / norm if norm > 0 else 0.0
        else:
            step = (((m - 1) * (2 * M - 5) / (2 * (M - 1))) - (m - 2)
                    + 0.5 * (m - 1) * (m - 2) / ((M - 1) * (M - 2))) * step1

        pi = pi + step * (deg - 2)

    final_w, _, _ = _bound_value(cost, best_pi, None)     # complete-graph certificate
    plain_w, _, _ = _bound_value(cost, np.zeros(n), None)
    return max(plain_w, final_w)
```

So the chain is: tours are intractable but they're exactly the all-degrees-2 1-trees, and
1-trees are cheap (one MST plus two edges), so the minimum 1-tree is an easy but loose lower
bound; the only freedom that leaves OPT invariant is a per-node price `pi_i + pi_j` on each
edge, because for a tour that adds the same constant `2 sum pi` to every tour; that freedom
yields `w(pi) = L + sum_i pi_i(deg_i - 2) <= OPT` for all prices, a Lagrangian dual whose
concave piecewise-linear shape has the degree-excess `deg - 2` as its subgradient; subgradient
ascent on the prices with a vanishing step drives the 1-tree's degrees toward 2, tightening the
bound toward OPT; and the complete-graph recomputation is the certificate that makes the value
safe to hand to branch-and-bound.
