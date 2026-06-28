# Context

## Research question

Given a complete undirected graph on `n` cities with a symmetric cost matrix `(c_ij)`, the
symmetric traveling-salesman problem asks for a minimum-weight tour — a cycle visiting every
vertex exactly once. Exact solution proceeds by branch-and-bound: recursively split the set of
tours (by forcing some edges in and others out) and discard any branch whose cheapest possible
tour already exceeds the best tour found so far. Each node of the search tree needs a **lower
bound** on the optimum cost of its subproblem. The question is how to compute, at each node, a
function `LB ≤ C*` (the optimum tour cost) that is as large as possible while remaining far
cheaper to evaluate than the tour problem itself — cheap enough to be recomputed at the
thousands of nodes that arise in instances of 40, 50, or 60 cities.

The deliverable is a single self-contained C++17 program reading from stdin and writing to
stdout.

## Background

The cost of an optimum tour `C*` is what branch-and-bound must under-estimate. Several relaxations
of "tour" were known to give lower bounds:

A tour is a connected spanning subgraph in which every vertex has degree exactly 2. Relaxing the
degree-2 requirement to *connectivity* alone gives a spanning structure; relaxing connectivity
to a *perfect 2-matching* gives the assignment relaxation. Each removes a constraint and so each
optimal relaxed object costs no more than the optimal tour.

Three pieces of mathematics are relevant.

**Minimum spanning trees are easy.** Kruskal (1956) and Prim/Dijkstra (1957–1959) compute a
minimum spanning tree of a weighted graph in low-order polynomial time by a greedy edge or
vertex scan. A spanning tree on `k` vertices has `k−1` edges; degrees are unconstrained. Gale
(1968) and Kruskal (1956) connect the greedy/matroid view of MST to exactly such selection
problems, and the greedy MST routine produces, as a by-product, the sensitivity of the tree
cost to forcing or forbidding individual edges.

**Lagrangian relaxation of integer programs.** When a hard combinatorial problem
`min cᵀx s.t. Ax = b, x ∈ X` has a set of "complicating" equality constraints `Ax = b` whose
removal leaves an *easy* problem over `X`, one can dualize them: for a multiplier vector `π`,
`L(π) = min_{x∈X} [cᵀx + πᵀ(b − Ax)]` is a lower bound on the optimum for every `π`, because at
the true optimum the bracketed term added is zero and the minimization over the larger set `X`
can only lower the value. The best such bound is `max_π L(π)`. As a minimum of finitely many
affine functions of `π`, `L(π)` is concave and piecewise linear — hence non-differentiable at
the breakpoints where the inner minimizer switches. The systematic exploitation of this
structure for integer programming was being worked out at the time (Geoffrion).

**The relaxation method for linear inequalities.** Agmon (1954) and Motzkin & Schoenberg (1954),
following an idea of Motzkin, studied solving a consistent system of linear inequalities
`a_iᵀx ≥ b_i` by iterated projection: pick a violated inequality, and step from the current point
toward (and across) its bounding hyperplane along the normal. Agmon's basic lemma (2.1): if `x`
is on the wrong side of an oriented hyperplane `π` and the solution `y` on the right side, and
`x_r` is the orthogonal projection of `x` onto `π`, then for `0 < λ < 2`,
`|x + λ(x_r − x) − y| < |x − y|` — the relaxed projection strictly decreases the Euclidean
distance to every solution point, and Agmon proves a linear convergence rate. The relaxation
parameter ranges over `λ ∈ (0, 2)`, and the lemma controls distance to the solution set rather
than the residual at the chosen inequality.

**Empirical state of TSP solving.** Exact branch-and-bound with the bounds in use exhausted
search trees for instances up to roughly 20–30 cities. The assignment relaxation and the bare
minimum-spanning-structure bound are cheap precisely because they drop the tour's degree-2
condition, so the optimal relaxed object need not resemble a Hamiltonian cycle.

## Baselines

**Bare spanning relaxation.** Drop both degree-2 and exact-connectivity-into-a-cycle; keep a
spanning connected structure. Its minimum weight is computed by a greedy MST routine and lower-
bounds `C*` because a tour contains a spanning tree.

**Assignment relaxation.** Relax a tour to a minimum-cost perfect 2-matching / assignment.
Solvable by the Hungarian / assignment algorithm; its dual gives values `u_i, v_j` with
`u_i + v_j ≤ c_ij`. It bounds `C*` from below and typically returns subtours (2-cycles and short
cycles).

**Column-generation linear program / steepest ascent over node prices.** Treat the best
achievable per-vertex-price bound as an optimization: maximize, over price vectors, the
spanning-relaxation cost computed under price-perturbed edge weights. One can attack this as a
large linear program with one constraint per candidate structure, generating columns as needed,
or by a steepest-ascent procedure that increases the objective at each step.

## Evaluation settings

The natural yardsticks are symmetric TSP instances from the literature and constructed test
families: published instances (e.g. the Dantzig 42-city problem, Croes 20-city, Karg–Thompson
57-city), `random(M)` instances with `c_ij` drawn i.i.d. from a discrete uniform distribution on
`{0,…,M}`, `random Euclidean(M)` instances with `n` points dropped uniformly in an `M × M`
square and `c_ij` the Euclidean distance, and `p × q` knight's-tour problems (cities are board
squares, cost 0 between knight-move-adjacent squares and ∞ otherwise) used to test for the
existence of a Hamiltonian circuit. Instance sizes of interest range from 20 up to 64 cities.
The metrics are: the value of the lower bound relative to the optimum tour cost `C*` (how tight),
the number of branch-and-bound nodes generated (how much pruning), and wall-clock time on the
machines of the day (an IBM 360/91). Each run is parameterized by controls on the bound
computation at each node, on how much effort to spend before branching, and by an upper bound on
`C*` used to discard subproblems.

## Input-output contract

The deliverable is a single self-contained C++17 program. It reads `n` followed by an `n × n`
symmetric real cost matrix from stdin, in row-major order. It writes the plain minimum 1-tree
cost and the Held-Karp bound computed with the Volgenant-Jonker schedule to stdout. The program
entry point is `int main()`.

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
