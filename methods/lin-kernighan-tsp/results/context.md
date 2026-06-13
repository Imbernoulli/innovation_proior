# Context: local search for the symmetric traveling-salesman problem

## Research question

Given an `n x n` symmetric matrix of distances between `n` cities, find a closed tour that visits
every city exactly once and has minimum total length. The number of distinct tours is
`(n-1)!/2`, so exhaustive search is hopeless even for moderate `n`. Exact methods exist but do not
scale predictably: dynamic programming and branch-and-bound with strong lower bounds can certify
small or favorable instances, yet the running time remains exponential in the hard cases. The
practical need is a method that, for instances of a hundred or more cities, returns optimal or
near-optimal tours quickly, with statistical confidence rather than a proof certificate.

The central tension is between the strength of a local move and the cost of searching for it. A
stronger move can escape more local optima, but if searching for that move costs too much, the
procedure loses the very speed that makes a heuristic useful. The prevailing way of setting that
tradeoff is to fix the exchange size in advance. That is the awkward point: one value of `k` must
serve the whole run, even though early moves may need large rearrangements and late moves may need
small repairs.

## Background

The dominant practical attack is iterative local search. Start from a feasible tour, repeatedly
replace it by a shorter neighboring tour, stop when no neighbor improves it, and repeat from many
random initial tours. The neighborhood is defined by an edge exchange. A **k-opt move** removes
`k` tour edges and adds `k` different edges so that the result is again one Hamiltonian cycle; the
move is accepted if the new tour is shorter. A tour is **k-optimal** (Lin's term:
`lambda`-optimal) if no improving k-opt move exists.

The k-optimality classes nest. A 3-optimal tour is also 2-optimal, a 4-optimal tour is also
3-optimal, and so on; at the limit, replacing all `n` edges can express any tour, so n-optimality
is global optimality. Larger `k` therefore imposes a stronger local-optimality condition. In the
Euclidean plane, 2-opt already has a clear geometric meaning: a 2-optimal tour has no crossed
edges. But 2-opt is still weak, and 3-opt removes many defects that cannot be repaired by a single
segment reversal.

The cost grows quickly. A full k-opt checkout must examine on the order of `n^k` edge subsets, so
the per-checkout work is roughly `Theta(n^k)`. Lin's 1965 3-opt procedure is the practical fixed-k
reference point: it improves greatly over Croes-style inversion search, but the checkout that
certifies no improving 3-opt move remains cubic. Moving to 4-opt increases the work by another
factor of `n`, and the benefit is not predictable enough to justify a fixed 4-opt sweep as the
default.

A second ingredient is an elementary arithmetic fact about cyclic sequences of real numbers. If a
cyclic list has positive total sum, then some rotation of that list has every partial sum positive:
rotate to begin just after a place where the prefix sum is minimal, and every prefix after that
rotation lies above the minimum.

## Baselines

**Croes 1958 - inversion search.** Croes uses the inversion transformation: choose two positions in
the tour, reverse the segment between them, and keep the reversal if it shortens the tour. This is
the standard 2-opt move, because it breaks two tour edges and reconnects the two open paths in the
only other way that remains a tour. The automatic part of the method stops at an inversion-free
tour. Its gap is the weakness of 2-opt: a tour may have no improving reversal while still being far
from optimal. Croes also describes manual adjustment, but that is not a scalable automatic
procedure.

**Lin 1965 - lambda-optimality and 3-opt.** Lin formalizes `lambda`-optimality and implements
3-opt as the next practical step beyond inversion search. A 3-opt move breaks three tour edges and
reconnects the resulting pieces in a shorter Hamiltonian cycle; one common view is to remove a
section of consecutive cities and reinsert it elsewhere, either in the same order or reversed. The
method generates many random starts, iterates to a 3-opt local optimum, and keeps the best tours;
a reduction idea fixes edges that appear repeatedly in good local optima. The gap is that `k = 3`
is fixed for the whole run. It is much stronger than 2-opt, but it still cannot make a move whose
useful exchange depth is four, ten, or thirty, while checking all such depths directly is too
expensive.

**Held-Karp exact methods.** Dynamic programming over subsets and 1-tree lower bounds inside
branch-and-bound provide exact solutions or strong lower bounds, but they remain exponential in
general. They are the correctness yardstick on small instances, not a practical engine for large
unstructured cases.

**Human-guided tour repair.** Plotting Euclidean tours and letting a human repair visible defects
can work on some two-dimensional instances. Its gap is not only human time; the idea depends on
geometry that may not exist for arbitrary symmetric distance matrices.

## Evaluation settings

The natural yardsticks are classical TSP instances with known or conjectured optima, random
symmetric matrices, and random Euclidean point sets. The relevant sizes range from small instances
where exact answers are known up through roughly 100-110 city cases and larger geometric examples.
The main quality metric is tour length relative to the optimum or best known tour. The operational
metrics are time per local optimum, time spent checking out a local optimum, the number of distinct
local optima seen across random restarts, and the probability that one random restart reaches the
best known tour.

Random restarts are central to the protocol. A strong improvement procedure should turn an
arbitrary feasible tour into a good local optimum, and repeated random starts give an empirical
distribution over local optima rather than one deterministic answer.

## Code framework

The available programming pieces are generic: a distance matrix, candidate lists that order nearby
cities first, an array representation of a tour with a position index, and two legal tour-editing
primitives: reverse a contiguous block, or remove one city and insert it after another. The empty
slot is the improvement rule that decides which legal edit to try and when to accept it.

```python
import math, random

def build_dist(coords):
    n = len(coords)
    D = [[0.0] * n for _ in range(n)]
    for i in range(n):
        xi, yi = coords[i]
        for j in range(i + 1, n):
            xj, yj = coords[j]
            D[i][j] = D[j][i] = math.hypot(xi - xj, yi - yj)
    return D

def neighbor_lists(D, k):
    n = len(D)
    return [
        sorted((j for j in range(n) if j != i), key=lambda j: D[i][j])[:k]
        for i in range(n)
    ]

def tour_length(order, D):
    n = len(order)
    return sum(D[order[i]][order[(i + 1) % n]] for i in range(n))

def reverse_block(order, pos, lo, hi):
    while lo < hi:
        order[lo], order[hi] = order[hi], order[lo]
        pos[order[lo]] = lo
        pos[order[hi]] = hi
        lo += 1
        hi -= 1

def relocate_after(order, pos, p, after):
    city = order.pop(p)
    if after > p:
        after -= 1
    order.insert(after + 1, city)
    for idx, node in enumerate(order):
        pos[node] = idx

def improve_step(order, pos, D, nbr):
    # TODO: choose a legal edge exchange or insertion move, verify that it shortens
    # the tour, apply it, and return True; return False when no move is found.
    pass

def solve(coords, k_neighbors=12, restarts=4, seed=0):
    D = build_dist(coords)
    nbr = neighbor_lists(D, k_neighbors)
    n = len(coords)
    rng = random.Random(seed)
    best_order, best_len = None, float("inf")
    for _ in range(restarts):
        order = list(range(n))
        rng.shuffle(order)
        pos = [0] * n
        for idx, node in enumerate(order):
            pos[node] = idx
        while improve_step(order, pos, D, nbr):
            pass
        length = tour_length(order, D)
        if length < best_len:
            best_order, best_len = list(order), length
    return best_order, best_len

if __name__ == "__main__":
    rng = random.Random(7)
    pts = [(100 * rng.random(), 100 * rng.random()) for _ in range(60)]
    D = build_dist(pts)
    start = list(range(len(pts)))
    rng.shuffle(start)
    start_len = tour_length(start, D)
    order = start[:]
    pos = [0] * len(order)
    for idx, node in enumerate(order):
        pos[node] = idx
    nbr = neighbor_lists(D, 12)
    while improve_step(order, pos, D, nbr):
        pass
    end_len = tour_length(order, D)
    assert sorted(order) == list(range(len(pts)))
    assert end_len <= start_len + 1e-9
    print(round(start_len, 3), round(end_len, 3))
```
