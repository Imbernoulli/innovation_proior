#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

// Aperture Atlas: maximum-weight independent set on a general (dense) conflict graph.
// testId is a difficulty/structure ladder:
//   - testId 1 is small (example scale), growing to ~600 targets by testId 10.
//   - graph density is tuned so the independence number stays moderate (~T), so the
//     ceiling F = 10*B is genuinely hard to reach and heuristics land in a rich range.
//   - the weight model and cross-structure vary per test so distinct strategies diverge.

int main(int argc, char** argv) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);

    // ---- number of targets grows along the ladder ----
    int n = 24 + 58 * (testId - 1);          // 24, 82, 140, ..., 546
    if (n < 5) n = 5;

    // ---- target a moderate independence number T so scores don't saturate the cap ----
    // For G(n,p), the independence number ~ 2 ln n / ln(1/(1-p)); solving for a target T:
    //   1 - p = n^(-2/T).  T ~ 9 keeps the best dockets around a few times the top target.
    double T = 8.5;
    double keep = pow((double)n, -2.0 / T);  // = 1 - p  (prob an edge is ABSENT)
    if (keep < 0.05) keep = 0.05;
    if (keep > 0.9)  keep = 0.9;
    double p = 1.0 - keep;                    // probability a pair conflicts

    // ---- weight model varies per test (adds strategic structure / divergence) ----
    // mode 0: uniform values
    // mode 1: a few "flagship" high-value targets among a low-value background
    // mode 2: cluster-correlated values (targets grouped, each group has a value scale)
    int mode = testId % 3;
    int wmax = 1000;

    // group assignment (used by mode 2, and to seed some cross structure)
    int groups = 5 + (testId % 4);            // 5..8 sky regions
    vector<int> grp(n + 1);
    vector<int> gscale(groups);
    for (int g = 0; g < groups; g++) gscale[g] = rnd.next(150, wmax);
    for (int i = 1; i <= n; i++) grp[i] = rnd.next(0, groups - 1);

    vector<int> w(n + 1);
    for (int i = 1; i <= n; i++) {
        if (mode == 0) {
            w[i] = rnd.next(1, wmax);
        } else if (mode == 1) {
            if (rnd.next(0, 99) < 8) w[i] = rnd.next(wmax - 200, wmax);  // flagship
            else                     w[i] = rnd.next(1, 250);            // background
        } else {
            int base = gscale[grp[i]];
            int lo = max(1, base - 120), hi = min(wmax, base + 120);
            w[i] = rnd.next(lo, hi);
        }
    }

    // ---- build the conflict graph ----
    // dense G(n,p) backbone; for cluster-mode also thicken intra-group conflicts so that
    // high-value same-region targets compete (forces real selection, not per-region picks).
    // Use a hash set to dedup (u,v).
    vector<pair<int,int>> edges;
    edges.reserve((size_t)((double)n * n * p * 0.6) + 16);
    set<long long> seen;
    auto key = [&](int a, int b) -> long long {
        if (a > b) swap(a, b);
        return (long long)a * (n + 1) + b;
    };
    auto addEdge = [&](int a, int b) {
        if (a == b) return;
        long long k = key(a, b);
        if (seen.insert(k).second) edges.push_back({a, b});
    };

    for (int i = 1; i <= n; i++)
        for (int j = i + 1; j <= n; j++) {
            double pr = p;
            if (mode == 2 && grp[i] == grp[j]) pr = min(0.95, p + 0.35); // same region conflicts more
            if (rnd.next(0.0, 1.0) < pr) addEdge(i, j);
        }

    int m = (int)edges.size();

    // guarantee at least one edge so the problem is non-trivial
    if (m == 0) { addEdge(1, 2); m = 1; }

    // shuffle so edge order carries no positional information
    shuffle(edges.begin(), edges.end());

    printf("%d %d\n", n, m);
    for (int i = 1; i <= n; i++) printf("%d%c", w[i], i == n ? '\n' : ' ');
    for (auto& e : edges) printf("%d %d\n", e.first, e.second);
    return 0;
}
