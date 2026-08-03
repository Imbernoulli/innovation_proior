#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

int main(int argc, char** argv) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);

    // ---- structure ladder: asteroid mining beacon swarm ----
    // testId 1 tiny (example scale); grows larger / denser toward testId 10.
    int n = 6 + (testId - 1) * 150;         // 6 .. 1356
    int K = 4 + (testId - 1);               // 4 .. 13 licensed channels
    if (K > 14) K = 14;
    int Wmax = 9;
    int gmax = 3;

    // ---- per-beacon factory tuning: home channel p and rigidity s ----
    // Home channels are SKEWED toward channel 1 on some tests and uniform on
    // others, so the retuning term interacts differently with the do-nothing
    // baseline across the ladder (drives per-test score divergence).
    vector<int> p(n + 1), s(n + 1);
    bool skewHome = (testId % 2 == 1);
    for (int i = 1; i <= n; i++) {
        if (skewHome) {
            // triangular skew toward 1: min of two draws
            int a = rnd.next(1, K), b = rnd.next(1, K);
            p[i] = min(a, b);
        } else {
            p[i] = rnd.next(1, K);
        }
        s[i] = rnd.next(1, 4);
    }

    vector<array<int, 4>> edges;            // u, v, w, g
    auto addEdge = [&](int u, int v, int w, int g) {
        if (u == v) return;
        edges.push_back({u, v, w, g});
    };

    // 1) tightly-packed asteroid clusters: cliques of size K+1.
    //    K+1 beacons over K channels -> pigeonhole forces residual cross-talk
    //    for ANY allocation, so F>0 and B>0 are guaranteed and the instance is
    //    genuinely over-constrained (no perfect frequency assignment exists).
    int cliqueSize = K + 1;
    int nClusters = max(1, n / (cliqueSize * 2));
    for (int cl = 0; cl < nClusters; cl++) {
        set<int> st;
        int guard = 0;
        while ((int)st.size() < cliqueSize && guard++ < cliqueSize * 20)
            st.insert(rnd.next(1, n));
        vector<int> vs(st.begin(), st.end());
        for (int a = 0; a < (int)vs.size(); a++)
            for (int b = a + 1; b < (int)vs.size(); b++)
                addEdge(vs[a], vs[b], rnd.next(4, Wmax), rnd.next(1, gmax));
    }

    // 2) diffuse inter-asteroid overlaps (sparse random graph, lighter guards).
    int avgDeg = 4 + testId / 2;
    long long mRand = (long long)n * avgDeg / 2;
    for (long long e = 0; e < mRand; e++) {
        int a = rnd.next(1, n), b = rnd.next(1, n);
        addEdge(a, b, rnd.next(1, Wmax), rnd.next(1, 2));
    }

    // cap edge count within the stated bound
    if ((int)edges.size() > 40000) edges.resize(40000);

    // shuffle so edge order carries no structural hint
    shuffle(edges.begin(), edges.end());

    int m = (int)edges.size();
    printf("%d %d %d\n", n, m, K);
    for (int i = 1; i <= n; i++)
        printf("%d %d\n", p[i], s[i]);
    for (auto& e : edges)
        printf("%d %d %d %d\n", e[0], e[1], e[2], e[3]);
    return 0;
}
