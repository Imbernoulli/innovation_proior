# The Held-Karp 1-tree lower bound

## Problem

For the symmetric TSP, exact branch-and-bound needs a cheap lower bound at every node. This is the
Held-Karp 1-tree Lagrangian lower bound computed by subgradient, or relaxation, ascent. It is not
the `O(2^n n^2)` Held-Karp exact dynamic program; the dynamic program solves the whole TSP
exponentially, while this routine supplies a fast bound for pruning.

## Bound

A 1-tree is a spanning tree on vertices `{2,...,n}` plus two distinct edges incident to vertex 1.
A tour is exactly a 1-tree whose every vertex has degree 2, so the minimum 1-tree cost is a lower
bound on the optimum tour cost `C*`.

Introduce node potentials `π` and compute 1-trees under perturbed edge weights
`c_ij + π_i + π_j`. Every tour gains the same constant `2Σ_i π_i`, while a 1-tree with degrees
`d_ik` gains `Σ_i π_i d_ik`. Therefore

  `C* + 2Σ_i π_i ≥ min_k [c_k + Σ_i π_i d_ik]`,

so

  `C* ≥ min_k [c_k + Σ_i π_i(d_ik − 2)] = w(π)`.

With `v_k = (d_ik − 2)_i`, this is `w(π) = min_k [c_k + π·v_k]`. The best bound is
`max_π w(π)`. The function is concave and piecewise linear because it is the minimum of affine
functions.

## Ascent

If `k(π)` is an active minimum 1-tree at `π`, then for every `τ`,

  `w(τ) − w(π) ≤ (τ − π)·v_{k(π)}`.

For a maximizer `π*`, this gives
`(π* − π)·v_{k(π)} ≥ w(π*) − w(π) ≥ 0`, so the degree residual points toward the optimal set in
the relaxation-method sense. The update is

  `π_{m+1} = π_m + t_m(d_m − 2)`.

The safe distance-decrease condition is

  `0 < t < 2(w(π*) − w(π)) / ‖d−2‖²`.

For a constant step `t`,

  `sup_m w(π_m) ≥ max_π w(π) − (t/2) limsup_m ‖d_m−2‖²`.

For a target level `ℓ < max w`, the relaxation step is
`t = λ(ℓ − w(π))/‖d−2‖²` with `ε ≤ λ ≤ 2`. Practical HWC-style code often substitutes a tour
upper bound `UB ≥ C* ≥ max w` and uses `t = λ(UB − w(π))/‖d−2‖²`, while halving `λ` so the steps
eventually vanish. The OR-Tools-style default schedule is Volgenant-Jonker, a positive decreasing
step sequence whose first step is based on the unweighted 1-tree cost.

## Code

```python
import math
import numpy as np

def _prim_mst(weighed):
    """Minimum spanning tree on a dense perturbed-cost matrix."""
    k = weighed.shape[0]
    in_tree = np.zeros(k, dtype=bool)
    best = weighed[0].copy()
    parent = np.zeros(k, dtype=int)
    in_tree[0] = True
    best[0] = np.inf
    edges = []
    for _ in range(k - 1):
        v = int(np.argmin(np.where(in_tree, np.inf, best)))
        edges.append((parent[v], v))
        in_tree[v] = True
        upd = (~in_tree) & (weighed[v] < best)
        best[upd] = weighed[v][upd]
        parent[upd] = v
    return edges

def compute_one_tree(cost, pi):
    """Minimum 1-tree under node potentials pi. Returns raw cost and degrees."""
    n = cost.shape[0]
    extra = n - 1
    weighed = cost + pi[:, None] + pi[None, :]
    sub_edges = _prim_mst(weighed[:extra, :extra])

    degrees = np.zeros(n, dtype=int)
    one_tree_cost = 0.0
    for u, v in sub_edges:
        degrees[u] += 1
        degrees[v] += 1
        one_tree_cost += cost[u, v]

    order = np.argsort(weighed[extra, :extra])
    for v in (int(order[0]), int(order[1])):
        degrees[extra] += 1
        degrees[v] += 1
        one_tree_cost += cost[extra, v]
    return one_tree_cost, degrees

class VolgenantJonker:
    """Vanishing step schedule reaching zero at iteration M."""
    def __init__(self, n, max_iterations=0):
        self.n = n
        self.M = max_iterations if max_iterations > 0 else int(28 * n ** 0.62)
        self.step1 = 0.0
        self.m = 0
        self._init = False

    def cont(self):
        self.m += 1
        return self.m <= self.M

    def step(self):
        m, M = self.m, self.M
        return ((m - 1) * (2 * M - 5) / (2 * (M - 1)) * self.step1
                - (m - 2) * self.step1
                + 0.5 * (m - 1) * (m - 2) / ((M - 1) * (M - 2)) * self.step1)

    def on_one_tree(self, one_tree_cost):
        if not self._init:
            self._init = True
            self.step1 = one_tree_cost / (2 * self.n)

    def on_new_wmax(self, one_tree_cost):
        self.step1 = one_tree_cost / (2 * self.n)

class HeldWolfeCrowder:
    """Upper-bound Polyak-style evaluator with lambda halving."""
    def __init__(self, n, upper_bound):
        self.n = n
        self.UB = upper_bound
        self.num_iter = 2 * n
        self.lam = 2.0
        self.it = 0
        self._step = 0.0

    def cont(self):
        if self.it >= self.num_iter:
            self.num_iter //= 2
            if self.num_iter < 2:
                return False
            self.it = 0
            self.lam /= 2
        else:
            self.it += 1
        return True

    def step(self):
        return self._step

    def on_one_tree(self, one_tree_cost, w, degrees):
        norm = float(np.sum((degrees - 2) ** 2))
        self._step = self.lam * (self.UB - w) / norm if norm > 0 else 0.0

    def on_new_wmax(self, one_tree_cost):
        pass

def held_karp_lower_bound(cost, algorithm="VJ", upper_bound=None, max_iterations=0):
    """Return the Held-Karp 1-tree lower bound."""
    cost = np.asarray(cost, dtype=float)
    n = cost.shape[0]
    if n < 2:
        return 0.0
    if n == 2:
        return cost[0, 1] + cost[1, 0]

    if algorithm == "HWC":
        if upper_bound is None:
            raise ValueError("HWC needs an upper_bound on OPT")
        alg = HeldWolfeCrowder(n, upper_bound)
    else:
        alg = VolgenantJonker(n, max_iterations)

    pi = np.zeros(n)
    best_pi = pi.copy()
    max_w = -math.inf
    w = 0.0
    while alg.cont():
        one_tree_cost, degrees = compute_one_tree(cost, pi)
        if isinstance(alg, HeldWolfeCrowder):
            alg.on_one_tree(one_tree_cost, w, degrees)
        else:
            alg.on_one_tree(one_tree_cost)
        g = degrees - 2
        w = one_tree_cost + float(np.dot(pi, g))
        if w > max_w:
            max_w = w
            best_pi = pi.copy()
            alg.on_new_wmax(one_tree_cost)
        pi = pi + alg.step() * g

    one_tree_cost, degrees = compute_one_tree(cost, best_pi)
    return one_tree_cost + float(np.dot(best_pi, degrees - 2))

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    pts = rng.random((5, 2))
    d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
    print("plain min-1-tree   :", compute_one_tree(d, np.zeros(5))[0])
    print("Held-Karp bound (VJ):", held_karp_lower_bound(d, "VJ"))
```

The returned value is always evaluated as raw 1-tree cost plus `π·(d−2)` at the best potentials
found, so it remains a lower bound on `C*`; the ascent only chooses better potentials.
