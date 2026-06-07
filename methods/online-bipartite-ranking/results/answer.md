# RANKING for online bipartite matching

## Problem

Bipartite G = (L, R, E) with a perfect matching of size n. L is known; the vertices of R arrive online, revealing their edges on arrival, and each must be matched to a free neighbor (or dropped) irrevocably before the next arrives. Maximize the matching size; measure by the worst-case competitive ratio E[ALG]/OPT against an oblivious adversary.

Deterministic algorithms are pinned at exactly 1/2: maximal greedy matching gives ≥ OPT/2, and a two-row phase adversary forces every deterministic algorithm, even one that refuses edges, to get at most one online match where the offline optimum gets two. Per-arrival random choice does no better (≈ 1/2 + o(1)) because memoryless randomness can be steered into spending the offline vertices a later sparse arrival needs.

## Key idea

Use a single global source of randomness fixed once, not a fresh coin per step. Draw a uniformly random priority order on L at initialization; every arrival defers to that one order. The correlation across decisions self-corrects: high-priority offline vertices are spent deliberately, low-priority ones are held back, so the algorithm stops over-investing in dense regions of the graph.

## Algorithm (RANKING)

Initialization: pick a uniformly random total order σ on L (equivalently, each i ∈ L draws Y_i ~ U[0,1] independently; sorted thresholds give a uniform permutation).

On arrival of j ∈ R: among j's currently-unmatched neighbors, match j to the one of highest priority (smallest σ, i.e. smallest Y_i); if none, drop j.

Guarantee: RANKING is (1 − 1/e)-competitive, and no randomized online algorithm beats 1 − 1/e (Yao's lemma applied to RANDOM on the complete upper-triangular instance), so it is optimal.

## Why 1 − 1/e — two analyses

**Combinatorial.** Reduce to the upper-triangular worst case (refusal/monotonicity arguments). For diagonal pair i, at least one of {row i, column i} is matched, so |M| = (n + |D|)/2 with D the both-matched set, giving E|M| = n/2 + (1/2)E|D|. With x_t = Pr[rank-t vertex matched], a permutation-perturbation argument (move a uniformly chosen vertex to rank t to manufacture the independence a naive count lacks) yields 1 − x_t ≤ (1/n) Σ_{s≤t} x_s. Setting S_t = Σ_{s≤t}x_s gives S_t(1 + 1/n) ≥ 1 + S_{t−1}; minimizing S_n under these constraints sets the inequalities tight, yielding x_t = (1 − 1/(n+1))^t and the lower bound (1/n)Σ_{t=1}^n x_t ≥ 1 − (1 − 1/(n+1))^n → 1 − 1/e.

**Randomized primal-dual.** Matching LP: max Σ x_ij with vertex sums ≤ 1. Dual: min Σ α_i + Σ β_j with α_i + β_j ≥ 1 on every edge. Threshold form: on matching i to j set α_i = g(Y_i)/F, β_j = (1 − g(Y_i))/F (others 0), with g monotone non-decreasing, g(1) = 1. Then α_i + β_j = 1/F on every match, so dual value = (1/F)|ALG| deterministically. For a fixed edge (i,j), fix all thresholds except Y_i; with critical threshold y^c (the threshold matched to j in the run with i deleted),

- Dominance Lemma: i is matched whenever Y_i < y^c, so E[α_i] ≥ ∫_0^{y^c} g(y) dy / F.
- Monotonicity Lemma: β_j ≥ (1 − g(y^c))/F for all Y_i (with i present, the free set stays a superset of the free set in the run with i deleted).

Hence E_{Y_i}[α_i + β_j] ≥ (1/F)[∫_0^θ g(y) dy + 1 − g(θ)] with θ = y^c. Requiring ∫_0^θ g(y) dy + 1 − g(θ) ≥ F for all θ makes the expected dual vector feasible edge-by-edge. Weak duality applied to that expected dual gives E[ALG] = F · E[dual value] ≥ F · OPT. The tight equality differentiates to g′ = g; with g(1) = 1 this gives g(θ) = e^{θ−1}, and F = 1 − 1/e. Feasible-in-expectation (rather than deterministic) duality is the relaxation that lets randomness exceed the deterministic 1/2; the same integral equation is the dual-charging shape behind fractional water-filling.

## Code

```python
"""
RANKING for online bipartite matching.

Offline side L is known up front; online side R arrives one vertex at a time,
revealing its edges only on arrival. Each arriving j must be matched (or dropped)
irrevocably before the next arrival.

Two equivalent forms are implemented:
  - permutation form: fix a uniformly random total order on L, match each arriving
    j to its unmatched neighbor of smallest rank.
  - threshold form: each i in L draws Y_i ~ U[0,1] independently, match each
    arriving j to the unmatched neighbor of smallest Y_i. (This is the form that
    makes the primal-dual analysis natural; g(Y_i)=e^{Y_i-1} sets the duals.)

Both are the SAME algorithm: the sorted Y-values induce a uniformly random
permutation of L.
"""

import random
import math
import heapq


def initialize_state(L, rng):
    """Draw the persistent random priorities used for the whole run."""
    return {i: rng.random() for i in L}      # Y_i ~ U[0,1]; ties have prob 0


def choose_neighbor(avail, state):
    """Choose the available neighbor with smallest persistent priority."""
    return min(avail, key=lambda v: state[v])


def stateful_match(L, arrivals, neighbors, state):
    """Generic one-pass harness filled with RANKING's priority rule."""
    matched_to = {}          # i in L  ->  j in R
    for j in arrivals:
        # On arrival we only get to see edges of j; among its UNMATCHED neighbors,
        # take the one earliest in the fixed order. No look-ahead, no revocation.
        avail = [i for i in neighbors[j] if i not in matched_to]
        if avail:
            matched_to[choose_neighbor(avail, state)] = j
    return matched_to


def ranking_match(L, arrivals, neighbors, rank):
    """Run RANKING.

    L:         iterable of offline vertices (known in advance).
    arrivals:  list giving the online arrival order (vertices of R).
    neighbors: dict j -> iterable of its neighbors in L (revealed on arrival of j).
    rank:      dict i -> comparable key; j is matched to the unmatched neighbor
               with the SMALLEST key. A uniformly random key gives RANKING.

    Returns the matching as a dict i -> j.
    """
    return stateful_match(L, arrivals, neighbors, rank)


def random_rank(L, rng):
    """Uniformly random total order on L (the random priority permutation)."""
    return initialize_state(L, rng)


def greedy_match(L, arrivals, neighbors):
    """Deterministic first-neighbor greedy.

    Always yields a maximal matching, hence >= OPT/2; an adversary forces = OPT/2.
    """
    matched_to = {}
    for j in arrivals:
        for i in neighbors[j]:
            if i not in matched_to:
                matched_to[i] = j
                break
    return matched_to


# --- complete upper-triangular graph ------------------------------------------
# Rows (offline L) and columns (online R) are 1..n. The matrix is upper-triangular
# with ones on and above the diagonal: column j is adjacent to rows 1..j. The
# unique perfect matching sits on the diagonal (row j matched to column j).
# Columns arrive in the order n, n-1, ..., 1: the first column
# to arrive sees every row, so an unlucky ranking wastes a high-priority row that a
# later, sparser column would have needed. RANKING attains 1 - 1/e here in the
# limit, matching the general upper bound.
def upper_triangular(n):
    L = list(range(1, n + 1))
    neighbors = {j: list(range(1, j + 1)) for j in range(1, n + 1)}  # rows 1..j
    arrivals = list(range(n, 0, -1))                                 # n, n-1, ..., 1
    return L, arrivals, neighbors


def competitive_ratio_on_upper_triangular(n, trials, match_fn, init_state, seed=0):
    rng = random.Random(seed)
    L, arrivals, neighbors = upper_triangular(n)
    total = 0
    for _ in range(trials):
        state = init_state(L, rng)
        total += len(match_fn(L, arrivals, neighbors, state))
    return (total / trials) / n


def ranking_size_upper_triangular(n, rng):
    """Fast simulation of RANKING on the upper-triangular graph."""
    rank = random_rank(range(1, n + 1), rng)
    heap = [(rank[i], i) for i in range(1, n + 1)]
    heapq.heapify(heap)

    matched = 0
    for j in range(n, 0, -1):
        while heap and heap[0][1] > j:
            heapq.heappop(heap)
        if heap:
            heapq.heappop(heap)
            matched += 1
    return matched


def ranking_ratio_on_upper_triangular(n, trials, seed=0):
    """Monte-Carlo estimate of E[|RANKING|]/OPT on the n x n upper-triangular graph.
    OPT = n (the diagonal perfect matching). Should approach 1 - 1/e for large n."""
    rng = random.Random(seed)
    total = 0
    for _ in range(trials):
        total += ranking_size_upper_triangular(n, rng)
    return (total / trials) / n


def greedy_half_trap(n):
    """Instance where first-neighbor greedy gets exactly half of OPT."""
    if n % 2:
        raise ValueError("n must be even")
    half = n // 2
    L = list(range(1, n + 1))
    arrivals = list(range(1, n + 1))
    neighbors = {}
    for j in range(1, half + 1):
        neighbors[j] = L[:]
    for k in range(1, half + 1):
        neighbors[half + k] = [k]
    return L, arrivals, neighbors


def greedy_ratio_on_half_trap(n):
    """Deterministic greedy / OPT on its standard half lower-bound instance."""
    L, arrivals, neighbors = greedy_half_trap(n)
    return len(greedy_match(L, arrivals, neighbors)) / n


if __name__ == "__main__":
    target = 1 - 1 / math.e
    print("RANKING on the complete upper-triangular family")
    for n, trials in ((100, 2000), (1000, 2000), (10000, 500)):
        r = ranking_ratio_on_upper_triangular(n, trials=trials)
        print(f"n={n:5d}  E[RANKING]/OPT ~ {r:.4f}   (1-1/e = {target:.4f})")

    print("\nFirst-neighbor greedy on the deterministic half-trap")
    for n in (100, 1000, 10000):
        r = greedy_ratio_on_half_trap(n)
        print(f"n={n:5d}  GREEDY/OPT      = {r:.4f}   (1/2 = {0.5:.4f})")
```

The Monte-Carlo harness on the upper-triangular family sanity-checks RANKING near 1 − 1/e ≈ 0.6321, while the deterministic half-trap keeps first-neighbor greedy at exactly 1/2.
