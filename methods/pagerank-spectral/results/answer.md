# PageRank: importance as a stationary distribution / principal eigenvector

## Problem

Rank every page on the Web by a single, query-independent **importance** score, computed once from the link graph, that is objective, scalable to ≈10⁸ nodes, resistant to manipulation, and faithful to a common-sense notion of importance (a link from an important page should count more than many links from obscure ones).

## Key idea

Importance is **recursive**: a page is important if important pages link to it. Each page splits one unit of endorsement evenly across its out-links, so the score vector r satisfies the eigenvector equation r = Mr with M column-stochastic — equivalently, r is the **stationary distribution of a random surfer** who follows links at random. The bare recursion is ill-posed on the real Web (non-unique on disconnected pieces, divergent at cyclic *rank sinks*, degenerate at *dangling* pages). The fix is a **teleport / damping** term: with probability 1−α the surfer jumps to a uniform random page. This produces the **Google matrix**

  **G = α M + (1 − α) (1/n) 11ᵀ**,  α = 0.85,

where dangling columns of M have first been patched to the uniform column. G is strictly positive and column-stochastic, so by **Perron–Frobenius** it has a unique, strictly positive principal eigenvector (eigenvalue 1) — the PageRank vector. The teleport makes the Markov chain irreducible and aperiodic (guaranteeing existence + uniqueness) and bounds the second eigenvalue by α, so **power iteration converges geometrically at rate at most α**, in a number of iterations independent of graph size.

## The result, stated cleanly

**Link matrix.** For pages 1..n, let N_v be the out-degree of page v. For N_v > 0, define M_{uv} = 1/N_v if v links to u and 0 otherwise; for N_v = 0, set the whole column M_{:v} = (1/n)1. This patched M is column-stochastic.

**PageRank equation (per page).** Let d = Σ_{v:N_v=0} r(v) be the current dangling mass.
  r(u) = α · Σ_{v→u, N_v>0} r(v)/N_v + αd/n + (1−α)/n,  with Σ_u r(u) = 1.
Equivalently, with the Google matrix G = α M + (1−α)(1/n)11ᵀ:
  **r = G r**,  r > 0,  ‖r‖₁ = 1.

**Existence / uniqueness / positivity (Perron–Frobenius).** Every entry of G is ≥ (1−α)/n > 0, so G is strictly positive. Since G is column-stochastic, eᵀG = eᵀ and ‖G‖₁ = 1, so ρ(G) ≤ 1 while eigenvalue 1 exists; the Perron root is exactly 1. For a strictly positive matrix the dominant eigenvalue is simple and strictly larger in modulus than all others, with a unique strictly positive eigenvector. Hence r exists, is unique up to scale, is strictly positive, and (normalized) is the unique stationary distribution of the irreducible aperiodic random-surfer chain.

**Convergence theorem (rate α).** Let x₂ be any eigenvector of G with eigenvalue λ₂ ≠ 1. The left eigenvector of G for eigenvalue 1 is e = 1, so eᵀx₂ = 0 (right/left eigenvectors for distinct eigenvalues are orthogonal). Then the rank-one teleport annihilates x₂: (11ᵀ)x₂ = 1(eᵀx₂) = 0. Therefore
  λ₂ x₂ = G x₂ = α M x₂ ⟹ M x₂ = (λ₂/α) x₂,
so x₂ is an eigenvector of the column-stochastic M with eigenvalue λ₂/α, and since ‖M‖₁ = 1 implies ρ(M) ≤ 1,
  **|λ₂| ≤ α**.
The power method error contracts like |λ₂|ᵏ ≤ αᵏ; reaching an L₁ tolerance ε takes ≈ log ε / log α iterations — independent of n. For α = 0.85 this worst-case bound is ≈85 iterations for ε = 10⁻⁶ and ≈99 iterations for ε = 10⁻⁷.

**Why α = 0.85.** α < 1 is required for strict positivity (hence uniqueness) and for |λ₂| ≤ α < 1 (hence convergence). α → 1 makes the ranking more link-faithful but |λ₂| can approach 1 (convergence slows sharply, disconnection/sinks reassert); α → 0 washes the score toward uniform (1/n). α = 0.85 keeps the ranking strongly link-driven while mixing fast enough to converge in a practical number of sparse sweeps.

## Algorithm (power iteration, G never materialized)

G is dense but never formed: α M is a sparse mat-vec, the teleport adds the single scalar (1−α)/n to every component, and dangling mass is redistributed as one more scalar broadcast — O(#links + n) per iteration.

The deliverable is a single self-contained C++17 program. It reads from stdin a header `n m` (pages, directed links) then `m` lines `src dst` (0-indexed, page src links to dst), and writes `n` lines `i score` — the importance of each page, scores summing to 1.

```cpp
// PageRank via power iteration on the Google matrix G = alpha*M + (1-alpha)*(1/n)*11^T.
// Reads from stdin: "n m", then m lines "src dst" (0-indexed directed edges, page src links to dst).
// Writes to stdout: n lines "i score", the importance score of each page (scores sum to 1).
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    long long m;
    if (!(cin >> n >> m)) return 0;
    if (n <= 0) return 0;

    const double alpha = 0.85;
    const double tol = 1.0e-6;       // L1 tolerance per node: stop when |r-r_last|_1 < n*tol
    const int max_iter = 100;

    // Column-stochastic link operator stored sparsely: for each edge v->u we will
    // push r(v)/N_v onto u. Keep out-degree N_v and the out-neighbour list per page.
    vector<vector<int>> out(n);
    vector<long long> outdeg(n, 0);
    for (long long e = 0; e < m; ++e) {
        long long s, d;
        cin >> s >> d;
        if (s < 0 || s >= n || d < 0 || d >= n) continue; // ignore out-of-range edges
        out[(int)s].push_back((int)d);
        outdeg[(int)s]++;
    }

    vector<double> r(n, 1.0 / n);      // positive start, sums to 1
    vector<double> rn(n, 0.0);
    const double teleport = (1.0 - alpha) / n;

    for (int it = 0; it < max_iter; ++it) {
        // dangling mass (pages with no out-links) is redistributed uniformly.
        double dangle = 0.0;
        for (int v = 0; v < n; ++v)
            if (outdeg[v] == 0) dangle += r[v];
        double broadcast = alpha * dangle / n + teleport; // per-page constant trickle

        for (int u = 0; u < n; ++u) rn[u] = broadcast;
        // sparse link flow: every non-dangling page v splits alpha*r(v) over its out-links.
        for (int v = 0; v < n; ++v) {
            if (outdeg[v] == 0) continue;
            double share = alpha * r[v] / (double)outdeg[v];
            for (int u : out[v]) rn[u] += share;
        }

        double diff = 0.0;
        for (int u = 0; u < n; ++u) diff += fabs(rn[u] - r[u]);
        swap(r, rn);
        if (diff < (double)n * tol) break;
    }

    cout.setf(std::ios::fixed);
    cout << setprecision(10);
    for (int u = 0; u < n; ++u) cout << u << ' ' << r[u] << '\n';
    return 0;
}
```

This is the canonical sparse power iteration (cf. NetworkX `pagerank`): build the column-stochastic link operator implicitly, redistribute dangling mass to the uniform teleport target, add the (1−α) uniform teleport each step, and stop on the L₁ change between iterates.
