#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

int main(int argc, char** argv) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);

    // ---- structure ladder: rocky-shore tide-pool network ----
    // testId 1 tiny (example scale); grows to a large, dense shore by testId 10.
    int n = 6 + (testId - 1) * 300;                 // 6, 306, ..., 2706
    int C = 2 + (testId % 3);                        // 2..4 candidate niches
    int avgdeg = 6 + testId;                         // 7..16 average channel degree
    long long m = (long long)n * avgdeg / 2;
    if (m < 1) m = 1;
    // cap edges to statement bound
    if (m > 25000) m = 25000;

    int wcap = 12 + testId;                          // typical mixing strength ceiling
    int heavyLo = 60 + 5 * (testId % 4);             // heavy-tailed strong currents
    int heavyHi = heavyLo + 120;
    if (heavyHi > 200) heavyHi = 200;
    double heavyProb = 0.08 + 0.02 * (testId % 3);   // fraction of strong channels

    // Community structure: pools cluster into "shelves"; within-shelf channels are
    // common (dense competition inside a shelf) with occasional cross-shelf links.
    int shelves = max(1, C + testId / 2);            // number of loosely-coupled clusters
    vector<int> shelfOf(n + 1);
    for (int i = 1; i <= n; i++) shelfOf[i] = rnd.next(0, shelves - 1);

    vector<array<int, 3>> edges;
    edges.reserve((size_t)m);

    auto pushEdge = [&](int u, int v) {
        if (u == v) return;
        int w;
        if (rnd.next(0.0, 1.0) < heavyProb)
            w = rnd.next(heavyLo, heavyHi);
        else
            w = rnd.next(1, wcap);
        edges.push_back({u, v, w});
    };

    long long made = 0;
    // guarantee a little intra-shelf backbone so clusters are meaningful
    for (int i = 1; i <= n && made < m; i++) {
        // link i to another pool, biased to same shelf
        int tries = 0;
        while (tries++ < 8) {
            int j;
            if (rnd.next(0.0, 1.0) < 0.75) {
                j = rnd.next(1, n);
                if (shelfOf[j] != shelfOf[i]) { if (tries < 8) continue; }
            } else {
                j = rnd.next(1, n);
            }
            if (j == i) continue;
            pushEdge(i, j);
            made++;
            break;
        }
    }
    // fill the rest with a shelf-biased random attachment
    while (made < m) {
        int u = rnd.next(1, n);
        int v;
        if (rnd.next(0.0, 1.0) < 0.7) {
            // same-shelf partner: resample until same shelf or give up quickly
            v = rnd.next(1, n);
            int guard = 0;
            while (v != u && shelfOf[v] != shelfOf[u] && guard++ < 4) v = rnd.next(1, n);
        } else {
            v = rnd.next(1, n);
        }
        if (u == v) continue;
        pushEdge(u, v);
        made++;
    }

    // shuffle so edge order is unstructured
    shuffle(edges.begin(), edges.end());

    int mm = (int)edges.size();
    printf("%d %d %d\n", n, mm, C);
    for (auto& e : edges)
        printf("%d %d %d\n", e[0], e[1], e[2]);
    return 0;
}
