The problem is the symmetric traveling-salesman problem: given an $n \times n$ matrix of distances between $n$ cities, find a closed tour visiting each city exactly once with minimum total length. There are $(n-1)!/2$ distinct tours, so exhaustive search is hopeless, and exact methods — dynamic programming over subsets, branch-and-bound with $1$-tree lower bounds — stay exponential in the hard cases. What is wanted for a hundred cities or more is a procedure that returns optimal or near-optimal tours quickly, with statistical confidence rather than a proof certificate. The practical engine for this is iterative local search: start from a feasible tour, repeatedly replace it with a shorter neighboring tour until none improves it, and repeat from many random starts. The neighborhood is an edge exchange — a $k$-opt move removes $k$ tour edges and adds $k$ non-tour edges so the result is again a single Hamiltonian cycle, accepted only when it is shorter. The $k$-optimality classes nest: a $3$-optimal tour is $2$-optimal, a $4$-optimal tour is $3$-optimal, and at the limit $n$-optimality is global optimality. Larger $k$ is exactly the direction one wants, since $2$-opt only removes crossings and even $3$-opt cannot express every useful rearrangement. The trouble is cost. Certifying $k$-optimality by direct search inspects on the order of $n^k$ subsets; Croes' inversion search ($2$-opt) is weak, Lin's $3$-opt is already cubic per checkout, and each further increment of $k$ costs another factor of $n$. Worse, fixed $k$-opt forces one value of $k$ on the whole run, even though a fresh random tour may need a large rearrangement while a nearly clean tour near the end needs only a small repair. Fixing the exchange size in advance is the wrong shape for the problem.

I propose the Lin-Kernighan heuristic, whose central idea is to make the exchange depth variable: do not choose $k$ first, but grow a single move as a sequential chain and let its useful depth reveal itself through a running gain ledger. Break a tour edge $x_1 = (t_1, t_2)$; from the exposed endpoint $t_2$ add a new edge $y_1 = (t_2, t_3)$; now $t_3$ has too many incident edges, so break one of its tour edges $x_2 = (t_3, t_4)$, moving the loose endpoint to $t_4$; add $y_2 = (t_4, t_5)$, which forces breaking $x_3 = (t_5, t_6)$; and so on. The move is an alternating chain $x_1, y_1, x_2, y_2, x_3, \ldots$ with $x_i = (t_{2i-1}, t_{2i})$ broken and, while the chain continues, $y_i = (t_{2i}, t_{2i+1})$ added. If the chain stops after $k$ broken edges, the last added edge is not a continuation but the closing edge $y_k = (t_{2k}, t_1)$ back to the start, so the alternating list contains exactly $k$ broken and $k$ added edges — a legitimate $k$-opt move whose depth is the stopping point of this particular chain rather than an input to the algorithm.

An open-ended chain that branches freely would explode into an exponential tree of partial exchanges, so the load-bearing piece is a pruning rule that kills unprofitable partial chains without discarding the improving ones. The natural ledger is gain. For each pair define
$$g_i = |x_i| - |y_i|,$$
and a closed depth-$k$ sequential exchange, with $y_k$ understood as the closing edge, has total gain
$$G = \sum_{i=1}^{k} g_i = f(T) - f(T'),$$
so it improves the tour exactly when $G > 0$. The individual $g_i$ need not all be positive — a good chain may spend earned gain to get into position for a later saving — so forbidding every negative step would collapse the method back into greedy shallow search. Instead I track the cumulative sum $G_i = g_1 + \cdots + g_i$ and keep extending only while every prefix stays positive, $G_i > 0$. The worry is whether this discards an improving exchange whose first choices look bad but whose final sum is positive, and the answer is no, by an elementary fact about positive-sum cycles. The gains of a sequential closed exchange sit on a cycle, so the start and direction are free to choose. Given numbers $a_1, \ldots, a_m$ with total $S_m > 0$ and prefix sums $S_0 = 0$, $S_i = a_1 + \cdots + a_i$, pick the largest index $r$ where $S_r$ is minimal and rotate to start at $a_{r+1}$. A rotated prefix ending before wraparound at original index $p > r$ has sum $S_p - S_r > 0$, since no later prefix ties the minimum; one that wraps and ends at $p \le r$ has sum $(S_m - S_r) + S_p \ge S_m > 0$, since $S_p \ge S_r$. Every prefix of the rotated sequence is positive. So scanning all starts and orientations and demanding $G_i > 0$ at every step never excludes an improving sequential exchange merely because of prefix order — that is the precise mathematical reason the chain search can be pruned so aggressively.

Three further mechanics make each step cheap and keep every recorded prefix cashable as a real tour. First, feasibility usually pins down the next broken edge: after adding the continuation edge $y_{i-1} = (t_{2i-2}, t_{2i-1})$, I take $x_i = (t_{2i-1}, t_{2i})$ to be the one tour edge at $t_{2i-1}$ such that joining $t_{2i}$ back to $t_1$ would close into a single tour; the other adjacent tour edge would split into two subtours. So $x_i$ is uniquely determined in the normal branch, and I can test a legal close-up at each depth rather than carry a vague partial graph. Second, the close-up test is where the adaptive $k$ is actually selected. With $H_{i-1}$ the open gain after the chosen continuation $y_{i-1}$, closing now with $y_i^* = (t_{2i}, t_1)$ would yield
$$C_i = H_{i-1} + |x_i| - |y_i^*|,$$
and whenever $C_i$ beats the best recorded close-up I set $G^* = C_i$ and remember $k = i$; if instead I continue, I choose $y_i = (t_{2i}, t_{2i+1})$ and update
$$H_i = H_{i-1} + |x_i| - |y_i|,$$
keeping the branch only while $H_i > 0$. When no admissible continuation remains, or the open gain falls below the recorded close-up, the best recorded prefix is taken as the move if $G^* > 0$, otherwise I backtrack. Third, the choice of which added edge to try is guided by the gain formula itself: since $y_i$ is an edge I pay for, short candidates are the plausible ones, which is why nearest-neighbor candidate lists are not merely a speed hack but are aligned with the objective. Pure nearest-first is still too local, though — a short $y_i$ that forces breaking a very short $x_{i+1}$ is a poor choice — so candidates are ranked by lookahead $|x_{i+1}| - |y_i|$, letting the current tour shape, not raw distance alone, decide. Backtracking is deliberately limited to the root: both choices of the first broken edge at $t_1$, a few alternatives for $y_1$, and at the second level the alternate $x_2$, including the special case that temporarily forms two subtours before a one-step repair. Past that, the gain rule and lookahead keep the chain narrow enough to follow greedily, so the depth can grow large while the branching stays small. This is exactly why sequential variable-depth exchange beats fixed $2$-opt and $3$-opt: it represents a shallow exchange when it closes early and a deep $k$-opt exchange when the ledger keeps supporting extension, without ever paying to enumerate a large fixed neighborhood. The ideal basic search, with the full set of starts, orientations, and shallow backtracking, makes the local optima $3$-optimal while still reaching deeper exchanges; in truncated candidate-list implementations that formal guarantee softens into a heuristic expectation rather than a certificate.

Turning the full chain into a small self-contained program would demand choose-added/choose-broken recursion, closability tests, disjoint broken and joined edge sets, restoration, and shallow backtracking, and a few repeated $2$-opt reversals are emphatically not Lin-Kernighan because they track none of the alternating $x_i, y_i$ ledger, the separate close-up gain, or the uniquely forced next break. For a compact runnable artifact the local-search slot is instead filled correctly with a $2.5$-opt core — positive-gain candidate $2$-opt reversals plus single-city relocation — which is weaker than full variable-depth Lin-Kernighan but preserves the discipline that matters: every move is a legal tour edit, every accepted move is verified by an exact gain check, candidates come from near-neighbor lists, and improvement repeats until no accepted move remains, all driven from random restarts.

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
