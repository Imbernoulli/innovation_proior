The assignment problem asks us to pair n rows with n columns in an n × n cost or rating matrix so that each row and each column is used exactly once and the total selected value is optimal. A legal solution is a permutation of the columns, and the naive way to find the best one is to enumerate all n! permutations. That is already hopeless at n = 12, where 12! is nearly half a billion, so we need structural insight rather than brute force.

The problem is a linear program, but throwing a general simplex solver at it is the wrong move. The assignment polytope is the most degenerate special case of the transportation problem: many bases describe the same vertex, and simplex grinds through pivots that do not improve the objective. The 0/1 matching result of König solves only the qualification special case, not matrices with general integer weights. What we want is an exact method that respects the bipartite combinatorics, runs in polynomial time, and hands back a small certificate of optimality.

The right tool is the Hungarian Method, also known as the Kuhn–Munkres algorithm. It is a primal-dual method built on the linear-programming dual of the assignment problem. We keep a row potential u_i for each person and a column potential v_j for each job so that every reduced cost c_ij − u_i − v_j is nonnegative. Because any feasible (u, v) gives a lower bound Σ_i u_i + Σ_j v_j on every assignment, an assignment that uses only tight edges — those with reduced cost exactly zero — must be optimal. This is complementary slackness: when the lower bound meets the assignment value, both are certified best possible.

The algorithm therefore alternates between two combinatorial steps. First it builds a maximum matching on the tight edges using augmenting paths; if the matching is perfect, we are done. If the matching is short, König's theorem gives a minimum vertex cover of the tight graph, which means the uncovered entries all have strictly positive reduced cost. Let δ be the smallest such uncovered slack. We update the potentials by subtracting δ from the uncovered rows and adding δ to the covered columns. This keeps every reduced cost nonnegative, creates at least one new tight edge, and strictly improves the dual objective. Repeating the loop must terminate, because each dual update lowers the gap by a positive amount and the optimum is bounded.

A clean way to implement the same logic in O(n^3) is to add one row at a time and grow a single shortest augmenting path while raising potentials by the minimum slack encountered along the way. The single-file C++17 program below reads the instance from stdin — an integer n, then the n × n integer cost matrix in row-major order — and writes the minimum total cost followed by the chosen column for each row. For a maximization problem, negate the matrix on input and negate the reported total. It works in long long throughout to avoid overflow on large entries.

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

The Hungarian Method is the canonical polynomial-time algorithm for the assignment problem. It is strongly polynomial, independent of the size of the matrix entries, and the potentials it produces serve as a compact, hand-checkable proof that the returned permutation is optimal.