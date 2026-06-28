# The Hungarian Method (Kuhn–Munkres) for the Assignment Problem

## Problem

Given an `n × n` cost matrix `c_ij`, choose a permutation `j_1, …, j_n` (one entry per row and per
column) minimizing `Σ_i c_{i j_i}`. (For a *maximization* with ratings `r_ij`, set `c_ij = −r_ij`,
solve, and negate the total.) Naively this is `n!` permutations. The problem is a linear program —
minimize `Σ_ij c_ij x_ij` subject to unit row/column sums and `x_ij ≥ 0` — but a degenerate one, and
its constraint matrix is **totally unimodular**, so the LP relaxation is integral (its vertices are
permutation matrices, Birkhoff). The Hungarian Method exploits this structure to solve it in `O(n^3)`
*and* produce a dual certificate of optimality (the potentials `u, v`).

## Key idea

A **primal–dual** method. Maintain dual **potentials** `u_i` (rows) and `v_j` (columns) that are
**feasible** — every **reduced cost** `c_ij − u_i − v_j ≥ 0`. By weak duality, for any feasible
`(u, v)` and any assignment, `Σ_i c_{i j_i} ≥ Σ_i u_i + Σ_j v_j`, so the dual lower-bounds every
assignment. **Complementary slackness:** an assignment is optimal iff it uses only **tight** edges
(reduced cost `0`). So:

1. Build a **maximum matching on the tight (zero-reduced-cost) edges** via **augmenting paths**
   (a matching is maximum iff it admits no augmenting path).
2. If it is **perfect**, complementary slackness certifies optimality — done.
3. If not, **König's theorem** (max matching = min vertex cover in a bipartite graph) gives a
   minimum vertex cover. Every uncovered edge has reduced cost `> 0`; let `δ` be the **minimum
   uncovered reduced cost**. **Update the duals** (Egerváry's step) to expose a new tight edge while
   preserving feasibility and strictly raising the dual objective `Σ u_i + Σ v_j`. Repeat.

In **reduced-matrix** form the duals are folded into the matrix `a_ij = c_ij − u_i − v_j ≥ 0`:
row-reduce, column-reduce (creating zeros = tight edges), match the zeros, cover them with the
minimum number of lines (König); if fewer than `n` lines, take `d =` minimum uncovered entry, subtract
`d` from every uncovered row and add `d` to every covered column (this keeps all entries `≥ 0`,
creates a new zero, and raises the dual objective — now a lower bound on the cost), and repeat until
`n` independent zeros exist.

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

A single self-contained C++17 program. It reads the instance from **stdin** — an integer `n`, then
the `n × n` integer cost matrix in row-major order — and writes the minimum total cost to **stdout**,
followed by `n` lines `i j` (row `i` is matched to column `j`, both 0-based). For a *maximization*
with ratings, negate the matrix on input and negate the printed total. It uses `long long` throughout
to avoid overflow.

```cpp
// Hungarian (Kuhn-Munkres) assignment, primal-dual O(n^3) shortest-augmenting-path form.
// Reads from stdin: an integer n, then an n x n integer cost matrix (row-major).
// Writes to stdout: the minimum total assignment cost, then n lines "i j"
// meaning row i (0-based) is matched to column j. Uses long long to avoid overflow.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    if (!(cin >> n)) return 0;
    vector<vector<long long>> cost(n, vector<long long>(n));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            cin >> cost[i][j];

    const long long INF = LLONG_MAX / 4;
    // 1-based with a dummy column 0 anchoring each augmenting path.
    vector<long long> u(n + 1, 0), v(n + 1, 0); // dual potentials (reduced costs >= 0)
    vector<int> p(n + 1, 0);                    // p[j] = row matched to column j (0 = free)
    vector<int> way(n + 1, 0);                  // predecessor column, to trace the path
    for (int i = 1; i <= n; i++) {
        p[0] = i;
        int j0 = 0;
        vector<long long> minv(n + 1, INF);     // min reduced cost reaching each column
        vector<char> used(n + 1, false);        // columns already on the path
        do {
            used[j0] = true;
            int i0 = p[j0], j1 = -1;
            long long delta = INF;
            for (int j = 1; j <= n; j++) {       // relax: extend the alternating path
                if (!used[j]) {
                    long long cur = cost[i0 - 1][j - 1] - u[i0] - v[j]; // reduced cost (slack)
                    if (cur < minv[j]) { minv[j] = cur; way[j] = j0; }
                    if (minv[j] < delta) { delta = minv[j]; j1 = j; }   // smallest slack = Koenig min
                }
            }
            for (int j = 0; j <= n; j++) {        // dual update by delta: tight edges stay tight
                if (used[j]) { u[p[j]] += delta; v[j] -= delta; }
                else          minv[j] -= delta;
            }
            j0 = j1;
        } while (p[j0] != 0);                     // until a free column -> augmenting path found
        do {                                      // flip the path via the predecessor chain
            int j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
        } while (j0);
    }

    vector<int> assign(n, 0);
    for (int j = 1; j <= n; j++)
        if (p[j] != 0) assign[p[j] - 1] = j - 1;

    long long total = 0;
    for (int i = 0; i < n; i++) total += cost[i][assign[i]];

    cout << total << "\n";
    for (int i = 0; i < n; i++) cout << i << " " << assign[i] << "\n";
    return 0;
}
```

The program runs in `O(n^3)` (strongly polynomial, independent of the cost magnitudes) and the
potentials it builds internally are the dual certificate of optimality. It agrees with full
permutation enumeration on small random matrices.
