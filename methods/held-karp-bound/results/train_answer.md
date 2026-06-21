Exact branch-and-bound for the symmetric traveling-salesman problem lives or dies on the lower bound it can compute at each search node. The cheapest available certificate is a minimum spanning tree: a tour contains one, so the MST weight is a valid lower bound on the optimum tour cost C*. But an MST is free to concentrate many edges on a few hub vertices, so it usually looks nothing like a Hamiltonian cycle and the gap to C* is large. The assignment relaxation goes in a different direction by finding a minimum-cost perfect 2-matching; it is tighter in some instances but typically returns subtours and short cycles, leaving its own gap. A more aggressive idea is to introduce per-vertex prices and maximize the resulting spanning-tree bound, but straightforward steepest ascent or column-generation formulations over an exponential collection of candidate structures turn out to be too slow for repeated evaluation inside a search tree. What is needed is a bound whose cost is essentially the cost of one spanning-tree computation, but that can be driven upward until the relaxed object looks much more like a tour.

The method is the Held-Karp 1-tree lower bound, computed by subgradient, or relaxation, ascent. A 1-tree is a spanning tree on vertices {2,...,n} together with the two cheapest edges incident to a chosen special vertex 1; it has n edges like a tour and is exactly as cheap to compute as an MST on n-1 vertices plus two minimum lookups. Since every tour is a 1-tree in which every vertex has degree 2, the minimum-weight 1-tree is always a lower bound on C*. The key insight is to add a potential π_i to every edge touching vertex i, replacing c_ij by c_ij + π_i + π_j. A tour's cost shifts by exactly 2Σ_i π_i, which is the same constant for every tour, so the TSP optimum is unchanged up to that offset. A 1-tree, however, shifts by Σ_i π_i d_i where d_i is its degree at i, so the potentials reshape which 1-tree is minimal. This yields the family of bounds C* ≥ w(π), where w(π) = min_k [ c_k + π·(d_k − 2) ] with c_k the raw cost of 1-tree k and d_k its degree vector. The best bound is max_π w(π), which is the Lagrangian dual obtained by dualizing the degree-2 constraints.

Maximizing w(π) is the remaining task. Because w is the minimum of finitely many affine functions, it is concave and piecewise linear, and at a point π the degree-residual vector v = d_{k(π)} − 2 of the current minimum 1-tree is a subgradient. Demanding that w strictly increase at every step is what makes steepest ascent crawl near the kinks, so instead the iteration is designed to decrease distance to the set of maximizers. The update π ← π + t v is a relaxed projection onto the half-space defined by the active 1-tree, and for 0 < t < 2(w* − w(π))/||v||² the next iterate is strictly closer to any maximizer. A constant step t guarantees sup_m w(π_m) ≥ max w − (t/2) limsup_m ||v_m||²; since the iterates eventually force most degrees to 2, the residual norm becomes small and the bound approaches the dual optimum. In practice one uses a vanishing schedule, such as the Volgenant-Jonker rule, or a Held-Wolfe-Crowder-style upper-bound rule t = λ(UB − w)/||v||² with λ repeatedly halved so that steps vanish. The entire routine slots into branch-and-bound: at each node edges are forced in or out and the ascent is run on the restricted 1-tree problem; if the bound reaches the incumbent upper bound the node is discarded, otherwise the search branches when the ascent stalls.

```python
import math
import numpy as np

def _prim_mst(weighed):
    """Greedy minimum spanning tree on a dense symmetric weight matrix."""
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
    """Minimum 1-tree under node potentials pi; returns raw cost and degrees."""
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
        return (
            (m - 1) * (2 * M - 5) / (2 * (M - 1)) * self.step1
            - (m - 2) * self.step1
            + 0.5 * (m - 1) * (m - 2) / ((M - 1) * (M - 2)) * self.step1
        )

    def on_one_tree(self, one_tree_cost):
        if not self._init:
            self._init = True
            self.step1 = one_tree_cost / (2 * self.n)

    def on_new_wmax(self, one_tree_cost):
        self.step1 = one_tree_cost / (2 * self.n)

class HeldWolfeCrowder:
    """Polyak-style step using an upper bound, with halving relaxation parameter."""
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
    """Return the Held-Karp 1-tree Lagrangian lower bound on the symmetric TSP."""
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
    print("plain min-1-tree    :", compute_one_tree(d, np.zeros(5))[0])
    print("Held-Karp bound (VJ):", held_karp_lower_bound(d, "VJ"))
```
