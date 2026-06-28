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

A single-file C++17 program: it reads `n` and an `n × n` symmetric cost matrix from stdin and
prints the plain minimum 1-tree cost and the Held-Karp bound (Volgenant-Jonker schedule) to stdout.

```cpp
// Held-Karp 1-tree Lagrangian lower bound for the symmetric TSP via subgradient
// ("relaxation") ascent. NOT the O(2^n n^2) exact DP; this is a cheap bound for
// branch-and-bound pruning.
//
// Input (stdin):  n, then an n x n symmetric real cost matrix (row-major).
// Output (stdout): the plain minimum 1-tree cost and the Held-Karp bound (VJ schedule).
#include <bits/stdc++.h>
using namespace std;

// Minimum 1-tree under node potentials pi: MST on nodes {0..n-2} plus the two
// cheapest edges from the left-out node n-1, all under weighed cost
// c(i,j)+pi[i]+pi[j]. Returns the sum of RAW edge costs in one_tree_cost and the
// degree of each node in deg.
static double compute_one_tree(const vector<vector<double>>& cost,
                               const vector<double>& pi,
                               vector<int>& deg) {
    int n = (int)cost.size();
    int extra = n - 1;  // the left-out / special node
    fill(deg.begin(), deg.end(), 0);
    double one_tree_cost = 0.0;

    // Prim MST on the extra ordinary nodes {0..extra-1} under perturbed cost.
    vector<double> best(extra);
    vector<int> parent(extra, 0);
    vector<char> in_tree(extra, 0);
    for (int j = 0; j < extra; ++j) best[j] = cost[0][j] + pi[0] + pi[j];
    in_tree[0] = 1;
    best[0] = numeric_limits<double>::infinity();
    for (int t = 0; t < extra - 1; ++t) {
        int v = -1;
        double bv = numeric_limits<double>::infinity();
        for (int j = 0; j < extra; ++j)
            if (!in_tree[j] && best[j] < bv) { bv = best[j]; v = j; }
        int u = parent[v];
        deg[u]++; deg[v]++;
        one_tree_cost += cost[u][v];          // accumulate RAW cost
        in_tree[v] = 1;
        for (int j = 0; j < extra; ++j) {
            double w = cost[v][j] + pi[v] + pi[j];
            if (!in_tree[j] && w < best[j]) { best[j] = w; parent[j] = v; }
        }
    }

    // Attach the extra node by its two cheapest (perturbed) edges.
    int e1 = -1, e2 = -1;
    double w1 = numeric_limits<double>::infinity(), w2 = w1;
    for (int j = 0; j < extra; ++j) {
        double w = cost[extra][j] + pi[extra] + pi[j];
        if (w < w1) { w2 = w1; e2 = e1; w1 = w; e1 = j; }
        else if (w < w2) { w2 = w; e2 = j; }
    }
    for (int v : {e1, e2}) {
        deg[extra]++; deg[v]++;
        one_tree_cost += cost[extra][v];
    }
    return one_tree_cost;
}

// Volgenant-Jonker vanishing step schedule reaching 0 at iteration M
// (EJOR 9:83-89, 1982). step1 is seeded from the first / best 1-tree cost.
struct VolgenantJonker {
    int n, M, m = 0;
    double step1 = 0.0;
    bool inited = false;
    VolgenantJonker(int n_, int max_iterations)
        : n(n_), M(max_iterations > 0 ? max_iterations
                                      : (int)(28.0 * pow((double)n_, 0.62))) {}
    bool cont() { ++m; return m <= M; }
    double step() const {
        double mm = m, MM = M;
        return (mm - 1) * (2 * MM - 5) / (2 * (MM - 1)) * step1
               - (mm - 2) * step1
               + 0.5 * (mm - 1) * (mm - 2) / ((MM - 1) * (MM - 2)) * step1;
    }
    void on_one_tree(double one_tree_cost) {
        if (!inited) { inited = true; step1 = one_tree_cost / (2.0 * n); }
    }
    void on_new_wmax(double one_tree_cost) { step1 = one_tree_cost / (2.0 * n); }
};

// Held-Karp 1-tree Lagrangian lower bound on OPT for cost matrix `cost`,
// using the Volgenant-Jonker schedule. w(pi) = cost(1-tree) + sum_i pi_i*(deg_i-2).
static double held_karp_lower_bound(const vector<vector<double>>& cost,
                                    int max_iterations = 0) {
    int n = (int)cost.size();
    if (n < 2) return 0.0;
    if (n == 2) return cost[0][1] + cost[1][0];

    VolgenantJonker alg(n, max_iterations);
    vector<double> pi(n, 0.0), best_pi(n, 0.0);
    vector<int> deg(n, 0);
    double max_w = -numeric_limits<double>::infinity();

    while (alg.cont()) {
        double one_tree_cost = compute_one_tree(cost, pi, deg);
        alg.on_one_tree(one_tree_cost);
        double w = one_tree_cost;
        for (int i = 0; i < n; ++i) w += pi[i] * (deg[i] - 2);  // w(pi) <= OPT
        if (w > max_w) {
            max_w = w;
            best_pi = pi;
            alg.on_new_wmax(one_tree_cost);
        }
        double s = alg.step();
        for (int i = 0; i < n; ++i) pi[i] += s * (deg[i] - 2);  // ascent g_i=deg_i-2
    }

    double one_tree_cost = compute_one_tree(cost, best_pi, deg);
    double w = one_tree_cost;
    for (int i = 0; i < n; ++i) w += best_pi[i] * (deg[i] - 2);
    return w;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n;
    if (!(cin >> n) || n <= 0) return 0;
    vector<vector<double>> cost(n, vector<double>(n, 0.0));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) cin >> cost[i][j];

    vector<int> deg(n, 0);
    vector<double> zero(n, 0.0);
    // A 1-tree needs n >= 3 (a tree on n-1 >= 2 nodes plus two distinct edges).
    double plain = (n >= 3) ? compute_one_tree(cost, zero, deg)
                            : (n == 2 ? cost[0][1] + cost[1][0] : 0.0);
    double bound = held_karp_lower_bound(cost);

    cout << fixed << setprecision(4);
    cout << "plain min-1-tree    : " << plain << "\n";
    cout << "Held-Karp bound (VJ): " << bound << "\n";
    return 0;
}
```

The returned value is always evaluated as raw 1-tree cost plus `π·(d−2)` at the best potentials
found, so it remains a lower bound on `C*`; the ascent only chooses better potentials.
