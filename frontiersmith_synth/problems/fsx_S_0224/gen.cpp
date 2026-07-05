#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main(int argc, char** argv) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);

    // ---- structure ladder ----
    // testId 1 is small (sanity scale); grows to a large contact net by testId 10.
    int n = 20 + 200 * (testId - 1);        // 20, 220, 420, ..., 1820
    if (n > 2000) n = 2000;
    long long mm = 10LL * n;                 // target ~degree 20
    if (mm > 20000) mm = 20000;
    int m = (int)mm;

    const int PMAX = 50;

    // planted latent group for each venue -> a "good" cut tends to align with it,
    // but the population band may fight that split, so heuristics diverge.
    vector<int> g(n + 1), p(n + 1);
    // near-balanced latent groups (slight per-test skew keeps instances distinct).
    // A cut aligned with the groups severs almost all the heavy transmission, while
    // the checker's cut-oblivious baseline severs only about half -> big headroom.
    double grpBias = 0.46 + 0.03 * (testId % 3); // ~0.46 .. 0.52
    vector<int> ing0, ing1;
    for (int i = 1; i <= n; i++) {
        g[i] = (rnd.next(0.0, 1.0) < grpBias) ? 1 : 0;
        if (g[i]) ing1.push_back(i); else ing0.push_back(i);
        // populations: mostly small, occasional large venue (heavy-tailed)
        if (rnd.next(0.0, 1.0) < 0.15)
            p[i] = rnd.next(PMAX / 2 + 1, PMAX);
        else
            p[i] = rnd.next(1, PMAX / 2);
    }
    if (ing0.empty()) { ing0.push_back(1); g[1] = 0; }
    if (ing1.empty()) { ing1.push_back(n); g[n] = 1; }

    // most channels bridge the two latent groups and carry heavy transmission;
    // the rest are light intra-group noise.
    double pCross = 0.72 + 0.02 * (testId % 4);

    vector<array<int,3>> edges;
    edges.reserve(m);
    for (int e = 0; e < m; e++) {
        int u, v, w;
        if (rnd.next(0.0, 1.0) < pCross) {
            u = ing0[rnd.next(0, (int)ing0.size() - 1)];
            v = ing1[rnd.next(0, (int)ing1.size() - 1)];
            w = rnd.next(50, 100);          // heavy cross-group channel
        } else {
            auto& grp = (rnd.next(0, 1) ? ing1 : ing0);
            if (grp.size() < 2) { e--; continue; }
            u = grp[rnd.next(0, (int)grp.size() - 1)];
            v = grp[rnd.next(0, (int)grp.size() - 1)];
            if (u == v) { e--; continue; }
            w = rnd.next(1, 6);             // light intra-group channel
        }
        edges.push_back({u, v, w});
    }

    ll T = 0;
    for (int i = 1; i <= n; i++) T += p[i];

    // population band around half of T, wide enough that a balanced split exists.
    ll half = T / 2;
    ll width = max((ll)(2 * PMAX + 5), T / 6);
    ll L = half - width;
    ll U = half + width;
    if (L < PMAX) L = PMAX;
    if (U > T - PMAX) U = T - PMAX;
    if (L > U) { L = PMAX; U = T - PMAX; }

    printf("%d %d %lld %lld\n", n, m, L, U);
    for (int i = 1; i <= n; i++) {
        printf("%d%c", p[i], i == n ? '\n' : ' ');
    }
    // shuffle edge order so index order carries no structure
    shuffle(edges.begin(), edges.end());
    for (auto& e : edges) printf("%d %d %d\n", e[0], e[1], e[2]);
    return 0;
}
