OK, let me think this through from the local-search end, because that is where the pressure is. I have a symmetric distance matrix and a feasible tour. I can improve it by exchanging edges: remove `k` tour edges, add `k` non-tour edges, and keep the result only if it is still one tour and is shorter. If no improving exchange of size `k` exists, the tour is k-optimal.

The hierarchy is attractive and annoying at the same time. A 3-optimal tour is already 2-optimal; a 4-optimal tour is already 3-optimal; if I let `k` reach `n`, I can replace the whole tour, so n-optimality is global optimality. Larger `k` is exactly the direction I want. In the Euclidean case, 2-opt only removes crossings; 3-opt fixes more structural mistakes; still larger exchanges can move pieces that no fixed 3-edge reconnection can move.

Then the cost hits. To certify k-optimality by direct search I have to inspect about `n^k` choices. Three-opt is already cubic just to check out one local optimum. Four-opt is another factor of `n`, five-opt another, and I still have to choose the value of `k` before the run begins. That is the wrong shape. A random tour at the beginning may need a large rearrangement, while a nearly clean tour near the end may only need a small repair. A fixed `k` spends the same kind of move everywhere.

So I do not want to choose `k` first. I want to grow one exchange and let its useful depth reveal itself. Start with a tour edge `x1 = (t1, t2)` that I break. From the exposed endpoint `t2`, add a new edge `y1 = (t2, t3)`. Now `t3` has too many incident edges unless I break one of its old tour edges, so I break `x2 = (t3, t4)`. The loose endpoint has moved to `t4`. If I add `y2 = (t4, t5)`, I must break `x3 = (t5, t6)`. The move becomes an alternating chain

`x1, y1, x2, y2, x3, ...`

with `x_i = (t_{2i-1}, t_{2i})` broken and, while I am continuing the chain, `y_i = (t_{2i}, t_{2i+1})` added. If I stop after `k` broken edges, the last added edge is not a continuation edge; it is the closing edge `y_k = (t_{2k}, t1)`. Then the same alternating list has exactly `k` broken edges and `k` added edges, so it is a k-opt move. The depth is no longer an input to the whole algorithm; it is the stopping point of this particular chain.

But an open-ended chain is dangerous. If every added edge can branch to many cities, this becomes an exponential tree of partial exchanges. I need a rule that kills bad partial chains early without killing the profitable exchanges I am actually searching for.

The natural ledger is gain. For each pair, define

`g_i = |x_i| - |y_i|`.

A closed sequential exchange has total gain `G = sum_i g_i = f(T) - f(T')`, with the final `y_k` understood as the closing edge back to `t1`, so it improves the tour exactly when `G > 0`. Individual `g_i` values need not all be positive. A chain may spend some already earned gain to get into position for a later saving. If I forbid every negative step, I fall back toward greedy shallow search.

The running sum is the useful object. Let `G_i = g_1 + ... + g_i`. Suppose I keep only chains with `G_i > 0` at every prefix. Am I throwing away a good exchange whose first few choices look bad but whose final sum is positive?

For a sequential closed exchange, the gains sit on a cycle: I can choose the starting broken edge and direction. So I only need the elementary fact about positive-sum cycles. Take numbers `a_1, ..., a_m` with total sum `S_m > 0`, and prefix sums `S_0 = 0, S_i = a_1 + ... + a_i`. Choose the largest index `r` where `S_r` is minimal. Rotate the sequence to start at `a_{r+1}`. If a rotated prefix ends before wraparound at original index `p > r`, its sum is `S_p - S_r`, which is positive because no later prefix ties the minimum. If it wraps and ends at `p <= r`, its sum is `(S_m - S_r) + S_p`, and since `S_p >= S_r`, this is at least `S_m`, hence positive. Every prefix of the rotated sequence is positive.

That gives the pruning rule its teeth. Any improving sequential exchange has some cyclic numbering whose cumulative gains stay positive. If I scan all starts and orientations, requiring `G_i > 0` at every construction step does not exclude that exchange merely because of prefix order. It does not cover nonsequential exchanges, and it does not rescue a practical implementation that truncates candidate lists too aggressively, but it is the right mathematical reason the chain search can be pruned so hard.

Now I need feasibility at every depth. After adding the continuation edge `y_{i-1} = (t_{2i-2}, t_{2i-1})`, the next broken edge `x_i` is not a free choice in the normal case. I choose `x_i = (t_{2i-1}, t_{2i})` so that joining `t_{2i}` back to `t1` would make one tour. The other adjacent tour edge at `t_{2i-1}` would close into two subtours. That makes `x_i` uniquely determined by the previous added edge in the usual feasible branch, and it means I can test a legal close-up at each depth instead of carrying a vague partial graph.

The close-up test is where the adaptive `k` is chosen. Let `H_{i-1}` be the open gain after the already chosen continuation edge `y_{i-1}`. Once `x_i` is fixed, closing now with `y_i* = (t_{2i}, t1)` would give `C_i = H_{i-1} + |x_i| - |y_i*|`. If `C_i` is better than the best recorded close-up gain, I set `G* = C_i` and remember `k = i`. If I do not close, I choose a continuation edge `y_i = (t_{2i}, t_{2i+1})` and update the open gain to `H_i = H_{i-1} + |x_i| - |y_i|`, keeping the branch only while `H_i > 0`. When no admissible continuation remains, or when the open gain has fallen below the recorded close-up gain, the best recorded prefix is the exchange if `G* > 0`; otherwise I backtrack.

Which added edge should I try? Since `y_i` is an edge I am paying for, short edges are the plausible ones. Nearest-neighbor lists are therefore not just a speed hack; they are aligned with the gain formula. But pure nearest-first is too local. If a candidate `y_i` forces me to break a very short `x_{i+1}`, it may be a bad choice even though `y_i` itself is short. The more informed score is lookahead: among a small set of short candidate additions, prefer the one with large `|x_{i+1}| - |y_i|`. That lets the current tour shape influence the choice instead of sorting only by raw distance.

Backtracking has to be limited. Full backtracking at every level would eventually recover exhaustive k-opt search and lose the point. The useful compromise is to branch near the root: try both choices for the first broken edge adjacent to `t1`, try a few alternatives for `y1`, and at the second level try the alternate `x2`, including the special temporary two-subtour case that can be repaired one step later. Past that, the gain rule and lookahead usually make the chain narrow enough to follow greedily. The depth can still become large, but the branching does not.

This is why sequential variable-depth exchange beats fixed 2-opt and fixed 3-opt. A 2-opt or 3-opt method decides in advance how many edges a move may touch, so it either cannot represent deeper rearrangements or must pay to enumerate a much larger fixed neighborhood. The sequential chain represents a depth-2 or depth-3 exchange when it closes early, and a deeper k-opt exchange when the gain ledger keeps supporting extension. In the ideal basic search, checking all starts, orientations, and the required shallow backtracking implies the local optima are 3-optimal. In practical truncated versions, that formal guarantee becomes only a heuristic expectation; a small candidate-list implementation does not certify 3-optimality.

Turning this into a small program forces a choice of scope. A faithful full chain needs choose-added / choose-broken recursion, closability tests, disjoint broken and joined edge sets, restoration, and shallow backtracking. A few repeated 2-opt reversals are not Lin-Kernighan, because they do not track the alternating `x_i, y_i` ledger, the separate close-up gain, or the uniquely forced next break in the feasible branch. For a short self-contained artifact, the local-search slot can still be made correct by using a 2.5-opt core: positive-gain candidate 2-opt plus single-city relocation. It is weaker than full variable-depth LK, but every move is a legal tour edit, every accepted move is checked by exact gain, and it improves tours in the same local-search framework.

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

Fixed k-opt gives a monotone strength hierarchy but charges roughly `Theta(n^k)` and makes me choose `k` before I know the move. A sequential exchange grows the move as alternating broken and added edges, and a close-up at any depth turns the prefix into a legal k-opt move. Additive gains make the running ledger meaningful; the positive-rotation lemma justifies requiring positive cumulative open gain without losing improving sequential exchanges. Closability keeps each recorded prefix cashable as a tour, shallow backtracking recovers the fragile early choices, lookahead chooses short additions that also break expensive old edges, and the move depth becomes local to the situation instead of fixed for the whole run.
