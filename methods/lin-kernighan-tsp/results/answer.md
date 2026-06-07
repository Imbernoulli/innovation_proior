# The Lin-Kernighan heuristic for the TSP

## Problem

For the symmetric TSP, a tour is improved by replacing tour edges with non-tour edges while keeping
one Hamiltonian cycle. Fixed k-opt local search gets stronger as `k` grows, but a full checkout of
all k-edge exchanges costs roughly `Theta(n^k)` and requires choosing `k` before the run begins.

## Method

Lin-Kernighan makes the exchange depth variable. Instead of enumerating all k-opt moves for a fixed
`k`, it grows one move as a sequential chain:

- break `x_i = (t_{2i-1}, t_{2i})`,
- add `y_i = (t_{2i}, t_{2i+1})`,
- then break `x_{i+1} = (t_{2i+1}, t_{2i+2})`,
- and at a final depth `k`, use the closing edge `y_k = (t_{2k}, t1)`.

The gain of each pair is

`g_i = |x_i| - |y_i|`,

and the gain of a closed depth-`k` sequential exchange, where the final `y_k` is the closing edge,
is

`G = sum_{i=1}^k g_i = f(T) - f(T')`.

The crucial pruning rule is the cumulative positive-gain criterion: keep extending only while every
running sum `G_i = g_1 + ... + g_i` is positive. This is justified by the positive-rotation lemma:
if a cyclic sequence of gains has positive total sum, some rotation has all partial sums positive.
So an improving sequential exchange can be searched from a start and direction where the gain
ledger never goes non-positive.

Other mechanics make the search cheap enough:

- closability usually determines the next broken edge uniquely after an added edge,
- each depth tests the close-up candidate `y_i* = (t_{2i}, t1)` with gain
  `C_i = H_{i-1} + |x_i| - |y_i*|`, while continuation uses
  `H_i = H_{i-1} + |x_i| - |y_i|`; the best positive `C_i` becomes `G*`,
- added edges are drawn from short candidate lists and ranked with lookahead `|x_{i+1}| - |y_i|`,
- broken and added edge sets are kept disjoint inside a move,
- backtracking is concentrated at levels 1 and 2, including the special alternate `x2` case that
  can temporarily form two subtours before repair.

The ideal basic search can be made 3-optimal at local checkout, while still reaching deeper
sequential exchanges without paying for exhaustive 4-opt, 5-opt, and higher sweeps. Candidate-list
and simplified-code versions should be treated as heuristics, not as certificates of full
3-optimality.

## Code

This is a runnable compact implementation of the same local-search scaffold using a 2.5-opt
improvement rule: candidate 2-opt reversals plus single-city relocation. It is intentionally weaker
than full Lin-Kernighan, but it keeps the important implementation discipline: legal tour edits,
near-neighbor candidates, exact gain checks, and repeated improvement until no accepted move
remains.

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
    n = len(order)
    eps = 1e-12

    # Candidate 2-opt: add a short edge, close the tour, and require exact positive gain.
    for i in range(n - 1):
        a = order[i]
        b = order[i + 1]
        old_ab = D[a][b]
        for c in nbr[a]:
            first_gain = old_ab - D[a][c]
            if first_gain <= eps:
                break
            k = pos[c]
            if k <= i + 1 or (i == 0 and k == n - 1):
                continue
            d = order[(k + 1) % n]
            gain = old_ab + D[c][d] - D[a][c] - D[b][d]
            if gain > eps:
                reverse_block(order, pos, i + 1, k)
                return True

    # Single-city relocation: a 3-edge edit accepted only when the exact net gain is positive.
    for p in range(n):
        v = order[p]
        a = order[(p - 1) % n]
        b = order[(p + 1) % n]
        remove_gain = D[a][v] + D[v][b] - D[a][b]
        for c in nbr[v]:
            after = pos[c]
            e = order[(after + 1) % n]
            if c in (a, b, v) or e == v:
                continue
            insert_cost = D[c][v] + D[v][e] - D[c][e]
            if remove_gain - insert_cost > eps:
                relocate_after(order, pos, p, after)
                return True

    return False

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
