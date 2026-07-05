#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

// Maximum-weight independent set on a GENERAL conflict graph ("gallery tour").
// testId is a difficulty/structure ladder:
//   testId 1  -> tiny (example scale)
//   testId 10 -> large, dense, clustered, heavy-tailed weights (adversarial)

int main(int argc, char** argv) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);

    int n;
    if (testId == 1) n = 8;                 // tiny example-scale case
    else n = 350 * testId;                  // 700, 1050, ..., 3500

    // ---- vertex values: heavy-tailed so value-aware selection matters ----
    // most installations are modest; a small fraction are "blockbusters".
    double spikeProb = 0.06 + 0.01 * (testId % 4);   // 0.06 .. 0.09
    int loHi = 50 + (testId % 3) * 40;               // modest ceiling 50..130
    vector<int> w(n + 1);
    for (int i = 1; i <= n; i++) {
        if (rnd.next(0.0, 1.0) < spikeProb)
            w[i] = rnd.next(1500, 5000);             // blockbuster
        else
            w[i] = rnd.next(1, loHi);                // modest
    }

    // ---- general conflict graph ----
    // A mix of (a) Erdos-Renyi random conflicts and (b) planted cliques
    // (groups of mutually conflicting installations) that create local
    // structure where weight-aware choices diverge from index order.
    set<pair<int,int>> eset;
    auto addEdge = [&](int a, int b) {
        if (a == b) return;
        if (a > b) swap(a, b);
        eset.insert({a, b});
    };

    // (b) planted conflict cliques (rooms whose installations all clash)
    int nClusters = testId + 2;
    int clusterSz = 4 + (testId % 4);               // 4..7 installations per room
    for (int c = 0; c < nClusters; c++) {
        vector<int> grp;
        for (int j = 0; j < clusterSz; j++) grp.push_back(rnd.next(1, n));
        for (size_t a = 0; a < grp.size(); a++)
            for (size_t b = a + 1; b < grp.size(); b++)
                addEdge(grp[a], grp[b]);
    }

    // (a) random background conflicts up to a target average degree
    int avgDeg = 4 + (testId % 5) * 2;              // 4..12
    long long target = (long long)n * avgDeg / 2;
    long long cap = min<long long>(60000, target);
    int guard = 0;
    while ((long long)eset.size() < cap && guard < 20 * cap + 1000) {
        int a = rnd.next(1, n), b = rnd.next(1, n);
        addEdge(a, b);
        guard++;
    }

    // materialise & shuffle edge order so index != structural position
    vector<pair<int,int>> edges(eset.begin(), eset.end());
    shuffle(edges.begin(), edges.end());
    for (auto& e : edges)
        if (rnd.next(0, 1)) swap(e.first, e.second);   // randomise endpoint order

    int m = (int)edges.size();
    printf("%d %d\n", n, m);
    for (int i = 1; i <= n; i++)
        printf("%d%c", w[i], i == n ? '\n' : ' ');
    for (auto& e : edges)
        printf("%d %d\n", e.first, e.second);
    return 0;
}
