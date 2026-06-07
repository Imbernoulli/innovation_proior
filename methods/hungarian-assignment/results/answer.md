# The Hungarian Method (Kuhn–Munkres) for the Assignment Problem

## Problem

Given an `n × n` cost matrix `c_ij`, choose a permutation `j_1, …, j_n` (one entry per row and per
column) minimizing `Σ_i c_{i j_i}`. (For a *maximization* with ratings `r_ij`, set `c_ij = −r_ij`,
solve, and negate the total.) Naively this is `n!` permutations. The problem is a linear program —
minimize `Σ_ij c_ij x_ij` subject to unit row/column sums and `x_ij ≥ 0` — but a degenerate one, and
its constraint matrix is **totally unimodular**, so the LP relaxation is integral (its vertices are
permutation matrices, Birkhoff). The Hungarian Method exploits this structure to solve it in `O(n^3)`
*and* return a dual certificate of optimality.

## Key idea

A **primal–dual** method. Maintain dual **potentials** `u_i` (rows) and `v_j` (columns) that are
**feasible** — every **reduced cost** `c_ij − u_i − v_j ≥ 0`. By weak duality, for any feasible
`(u, v)` and any assignment, `Σ_i c_{i j_i} ≥ Σ_i u_i + Σ_j v_j`, so the dual lower-bounds every
assignment. **Complementary slackness:** an assignment is optimal iff it uses only **tight** edges
(reduced cost `0`). So:

1. Build a **maximum matching on the tight (zero-reduced-cost) edges** via **augmenting paths**
   (a matching is maximum iff it has no augmenting path — Berge).
2. If it is **perfect**, complementary slackness certifies optimality — done.
3. If not, **König's theorem** (max matching = min vertex cover in a bipartite graph) gives a
   minimum vertex cover. Every uncovered edge has reduced cost `> 0`; let `δ` be the **minimum
   uncovered reduced cost**. **Update the duals** (Egerváry's step) to expose a new tight edge while
   preserving feasibility and strictly raising the dual objective `Σ u_i + Σ v_j`. Repeat.

In **reduced-matrix** form the duals are folded into the matrix `a_ij = c_ij − u_i − v_j ≥ 0`:
row-reduce, column-reduce (creating zeros = tight edges), match the zeros, cover them with the
minimum number of lines (König); if fewer than `n` lines, take `d =` minimum uncovered entry, subtract
`d` from every uncovered row and add `d` to every covered column (this keeps all entries `≥ 0`,
creates a new zero, and lowers the dual objective), and repeat until `n` independent zeros exist.

## Algorithm

Reduced cost `w_ij = c_ij − u_i − v_j`. Dual feasibility: `w_ij ≥ 0`. Optimality: a perfect matching
on `{w_ij = 0}`.

- **Init.** `u_i = 0`, `v_j = min_i c_ij` (any feasible duals).
- **Repeat:** maximum matching `M` on tight edges via augmenting paths.
  - If `|M| = n`: return `M` and cost `Σ u_i + Σ v_j`.
  - Else let `L` = vertices reachable by alternating paths from exposed rows;
    `δ = min{ w_ij : i ∈ A∩L, j ∈ B∖L }`. Update `u_i += δ` for `i ∈ A∩L`, `v_j −= δ` for
    `j ∈ B∩L`. (Equivalently, in the tableau: subtract `δ` from uncovered rows, add `δ` to covered
    columns.) Feasibility is preserved and the dual objective rises by `δ·(n − |cover|) > 0`.

Amortized into one **shortest augmenting path per row**, the dual updates become the running minimum
slack along a single growing path, giving `O(n^3)` (strongly polynomial — independent of the cost
magnitudes).

## Code

Two faithful, self-contained forms; both verified against brute force.

```python
INF = float("inf")

# ---- primal-dual O(n^3) potential / shortest-augmenting-path form ----
def hungarian_potential(cost):
    """Min-cost assignment. cost is n x m with m >= n. Returns (assignment, total)."""
    n, m = len(cost), len(cost[0])
    u = [0]*(n+1); v = [0]*(m+1)          # dual potentials, feasible throughout
    p = [0]*(m+1)                         # p[j] = row matched to column j (0 = free)
    way = [0]*(m+1)                       # predecessor column, to trace the path
    for i in range(1, n+1):
        p[0] = i; j0 = 0
        minv = [INF]*(m+1); used = [False]*(m+1)
        while True:
            used[j0] = True
            i0 = p[j0]; delta = INF; j1 = -1
            for j in range(1, m+1):                       # relax / extend alternating path
                if not used[j]:
                    cur = cost[i0-1][j-1] - u[i0] - v[j]  # reduced cost (slack)
                    if cur < minv[j]:
                        minv[j] = cur; way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]; j1 = j           # smallest slack = Koenig minimum
            for j in range(0, m+1):                       # dual update by delta
                if used[j]:
                    u[p[j]] += delta; v[j] -= delta       # tight edges stay tight
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:                                # reached a free column
                break
        while j0:                                         # flip the augmenting path
            j1 = way[j0]; p[j0] = p[j1]; j0 = j1
    assign = [0]*n
    for j in range(1, m+1):
        if p[j] != 0:
            assign[p[j]-1] = j-1
    return assign, sum(cost[i][assign[i]] for i in range(n))


# ---- classic O(n^3) matrix form: reduce, cover, subtract/add minimum uncovered ----
def hungarian_matrix(cost):
    """Min-cost assignment, square cost. Returns (assignment, total)."""
    n = len(cost)
    a = [row[:] for row in cost]
    for i in range(n):                                    # row reduction
        r = min(a[i])
        for j in range(n): a[i][j] -= r
    for j in range(n):                                    # column reduction
        c = min(a[i][j] for i in range(n))
        for i in range(n): a[i][j] -= c
    while True:
        match_col = [-1]*n; match_row = [-1]*n
        def aug(i, seen):                                 # augmenting-path match on zeros
            for j in range(n):
                if a[i][j] == 0 and not seen[j]:
                    seen[j] = True
                    if match_row[j] == -1 or aug(match_row[j], seen):
                        match_col[i] = j; match_row[j] = i; return True
            return False
        for i in range(n): aug(i, [False]*n)
        if all(c != -1 for c in match_col):               # n independent zeros -> optimal
            assign = match_col[:]
            return assign, sum(cost[i][assign[i]] for i in range(n))
        # Koenig minimum vertex cover from the maximum matching:
        row_marked = [match_col[i] == -1 for i in range(n)]
        col_marked = [False]*n; changed = True
        while changed:
            changed = False
            for i in range(n):
                if row_marked[i]:
                    for j in range(n):
                        if a[i][j] == 0 and not col_marked[j]:
                            col_marked[j] = True; changed = True
            for j in range(n):
                if col_marked[j] and match_row[j] != -1 and not row_marked[match_row[j]]:
                    row_marked[match_row[j]] = True; changed = True
        covered_rows = [not m for m in row_marked]        # cover = unmarked rows + marked cols
        covered_cols = col_marked
        d = min(a[i][j] for i in range(n) for j in range(n)
                if not covered_rows[i] and not covered_cols[j])
        for i in range(n):                                # subtract from uncovered rows,
            for j in range(n):                            # add to covered columns
                if not covered_rows[i]: a[i][j] -= d
                if covered_cols[j]:     a[i][j] += d


def brute_force(cost):
    import itertools
    n = len(cost); best, bestp = INF, None
    for perm in itertools.permutations(range(n)):
        s = sum(cost[i][perm[i]] for i in range(n))
        if s < best: best, bestp = s, perm
    return list(bestp), best


if __name__ == "__main__":
    R = [[8,7,9,9],[5,2,7,8],[5,1,4,8],[2,2,2,6]]         # maximization -> negate
    neg = [[-x for x in row] for row in R]
    for fn in (hungarian_potential, hungarian_matrix):
        a, t = fn(neg)
        print(fn.__name__, "assignment", a, "rating sum", -t)   # 25, (0,2,3,1)
```

`hungarian_potential` runs in `O(n^3)` and accepts rectangular `n × m` (`m ≥ n`) cost matrices;
`hungarian_matrix` is the literal König/Egerváry tableau for square inputs. Both agree with each
other and with full permutation enumeration on small random matrices.
