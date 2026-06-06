# The Held-Karp 1-tree lower bound for the symmetric TSP

## Problem

Branch-and-bound for the symmetric travelling-salesman problem needs, at every node of the
search tree, a lower bound on the optimal tour cost that is both cheap to recompute and tight.
The plain minimum spanning tree and the sum-of-two-cheapest-edges bounds are far too loose; the
tour itself is intractable. The Held-Karp bound fills the gap.

## Key idea

A **1-tree** is a spanning tree on all but one distinguished node, with that node hung back on
by its two cheapest edges. It has `n` edges, one cycle, and average degree 2. A tour is exactly
a 1-tree in which *every* node has degree 2, so tours are a subset of 1-trees and

  min-cost 1-tree  <=  min-cost tour  =  OPT.

To tighten this, attach a potential `pi_i` to each node and perturb edge costs to
`c'(i,j) = c(i,j) + pi_i + pi_j`. For any structure `t`, the perturbed cost equals
`cost(t) + sum_i pi_i * deg_t(i)`. Every tour has all degrees 2, so its perturbed cost is
`cost(tour) + 2 * sum_i pi_i` — a constant shift independent of which tour — hence the optimal
tour is invariant to the perturbation. Repeating the subset argument under perturbed costs and
subtracting the constant gives, for the minimum 1-tree with raw cost `L` and degree vector
`deg`,

  w(pi) = L + sum_i pi_i * (deg_i - 2)  <=  OPT,  for every pi.

The **Held-Karp bound** is `max_pi w(pi)`: the Lagrangian dual obtained by relaxing the
degree-2 equality at every node and pricing its violation with `pi`. Maximizing over `pi`
drives the minimum 1-tree's degrees toward 2 — toward a tour — and tightens the bound. If the
prices ever make the minimum 1-tree all-degrees-2, it is a tour and the bound equals OPT.

## The algorithm

`w(pi)` is a pointwise minimum of affine functions of `pi`, hence concave and piecewise-linear.
At a price vector `pi` with minimizing 1-tree of degrees `deg`, the vector `g_i = deg_i - 2` is
a supergradient. Maximize by subgradient ascent:

  pi_i  <-  pi_i + step * (deg_i - 2).

A node of degree 3 has its price raised (its edges get costlier, the next 1-tree drops one); a
degree-1 node has its price lowered. Two step-size schedules:

- **Volgenant-Jonker** (default): a prescribed positive step decreasing to exactly 0 at a final
  iteration `M`,
  `step(m) = [ (m-1)(2M-5)/(2(M-1)) - (m-2) + (1/2)(m-1)(m-2)/((M-1)(M-2)) ] * step1`,
  with `step1 = L / (2n)` refreshed on each new best `w`, and `M ~ 28 n^0.62`. Needs no upper
  bound.
- **Held-Wolfe-Crowder**: a Polyak step `step = lambda * (UB - w) / sum_i (deg_i - 2)^2`, where
  `UB` is any upper bound on OPT (a complete heuristic tour cost is enough) and `lambda` starts
  at 2 and is halved together with the iteration budget in successive passes until the budget is
  tiny.

For speed, the per-iteration MST is computed on a sparse nearest-neighbour graph; because a
1-tree on a restricted graph can only cost more than the complete-graph minimum for the same
prices, the sparse values are only used to choose prices. The final bound is recomputed once on
the complete graph with the best prices to guarantee validity.

## Working code

```python
import math
import numpy as np


def _prim_mst_edges(weight, allowed=None):
    """Prim MST on a dense matrix, optionally restricted to a connected edge mask."""
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
    """Nearest-neighbour edge mask plus MST edges for connectivity."""
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
    """Any complete tour cost is a valid UB for the Polyak target."""
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
    """Minimum 1-tree under node potentials pi. Returns (raw 1-tree cost, degrees)."""
    n = cost.shape[0]
    extra = n - 1                                   # left-out special node
    weighed = cost + pi[:, None] + pi[None, :]      # c'(i,j) = c + pi_i + pi_j
    sub_edges = _prim_mst_edges(weighed[:extra, :extra], allowed)
    degrees = np.zeros(n, dtype=int)
    one_tree_cost = 0.0                             # accumulate RAW cost
    for u, v in sub_edges:
        degrees[u] += 1
        degrees[v] += 1
        one_tree_cost += cost[u, v]
    to_extra = weighed[extra, :extra]               # two cheapest edges for the special node
    a, b = np.argsort(to_extra)[:2]
    for v in (int(a), int(b)):
        degrees[extra] += 1
        degrees[v] += 1
        one_tree_cost += cost[extra, v]
    return one_tree_cost, degrees


def _bound_value(cost, pi, allowed=None):
    one_tree_cost, degrees = compute_one_tree(cost, pi, allowed)
    return one_tree_cost + float(np.dot(pi, degrees - 2)), one_tree_cost, degrees


def held_karp_lower_bound(cost, algorithm="VJ", upper_bound=None,
                          max_iterations=0, nearest_neighbors=40):
    """Held-Karp lower bound on the optimal symmetric-TSP tour for matrix `cost`.
    algorithm: "VJ" (Volgenant-Jonker) or "HWC" (Held-Wolfe-Crowder)."""
    cost = np.asarray(cost, dtype=float)
    n = cost.shape[0]
    if n < 2:
        return 0.0
    if n == 2:
        return cost[0, 1] + cost[1, 0]

    algorithm = algorithm.upper()
    search_edges = _candidate_edges(cost[:n - 1, :n - 1], nearest_neighbors)

    if algorithm == "HWC":
        if upper_bound is None:
            upper_bound = _tour_upper_bound(cost)
        num_iter = 2 * n
        lam = 2.0
        it = 0
    elif algorithm == "VJ":
        M = max(3, max_iterations if max_iterations > 0 else int(28 * n ** 0.62))
        step1 = 0.0
    else:
        raise ValueError("algorithm must be 'VJ' or 'HWC'")

    pi = np.zeros(n)
    best_pi = pi.copy()
    best_search_w = -math.inf
    m = 0
    while True:
        if algorithm == "HWC":
            if it >= num_iter:
                num_iter //= 2
                if num_iter < 2:
                    break
                it = 0
                lam /= 2
            else:
                it += 1
        else:
            m += 1
            if m > M:
                break

        w, one_tree_cost, degrees = _bound_value(cost, pi, search_edges)
        if w > best_search_w:
            best_search_w = w
            best_pi = pi.copy()
            if algorithm == "VJ":
                step1 = one_tree_cost / (2 * n)     # refresh step scale on improvement

        if algorithm == "HWC":
            norm = float(np.sum((degrees - 2) ** 2))
            step = lam * (upper_bound - w) / norm if norm > 0 else 0.0
        else:
            if m == 1:
                step1 = one_tree_cost / (2 * n)
            step = (((m - 1) * (2 * M - 5) / (2 * (M - 1))) - (m - 2)
                    + 0.5 * (m - 1) * (m - 2) / ((M - 1) * (M - 2))) * step1

        pi = pi + step * (degrees - 2)              # subgradient ascent: g_i = deg_i - 2

    final_w, _, _ = _bound_value(cost, best_pi, None)      # complete-graph certificate
    plain_w, _, _ = _bound_value(cost, np.zeros(n), None)  # never worse than plain 1-tree
    return max(plain_w, final_w)
```

Every complete-graph value returned by the code satisfies `<= OPT`. If the complete-graph
minimum 1-tree has all degrees equal to 2 at the chosen prices, it is a tour and the bound
equals OPT; otherwise the returned value is the best certified value among the unweighted
1-tree and the final complete-graph recomputation with the optimized prices.
