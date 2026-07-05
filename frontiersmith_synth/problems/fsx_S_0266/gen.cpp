#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

int main(int argc, char** argv) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);

    // ---- structure ladder: cluster of star systems on a hyperlane graph ----
    // testId 1 is tiny (example scale); grows to a large cluster by testId 10.
    int N = 12 + 30 * (testId - 1);          // 12, 42, ..., 282
    if (N < 2) N = 2;

    // radius: small tests use r=1 (dense-ish), the largest tests use r=2 on a
    // sparser near-tree so coverage balls stay moderate (keeps scores un-capped).
    int r = (testId >= 8) ? 2 : 1;

    // build a random connected graph: spanning tree + extra hyperlanes.
    set<pair<int,int>> es;
    auto addEdge = [&](int a, int b) {
        if (a == b) return;
        if (a > b) swap(a, b);
        es.insert({a, b});
    };

    // random spanning tree (attach each new system to a random earlier one)
    for (int i = 2; i <= N; i++) {
        int p = rnd.next(1, i - 1);
        addEdge(p, i);
    }

    // extra hyperlanes: denser for r=1, sparse for r=2 so balls stay moderate
    int extra = (r == 1) ? N : N / 3;
    for (int e = 0; e < extra; e++) {
        int a = rnd.next(1, N), b = rnd.next(1, N);
        addEdge(a, b);
    }

    vector<pair<int,int>> edges(es.begin(), es.end());
    shuffle(edges.begin(), edges.end());
    int M = (int)edges.size();

    // per-system relay costs (skewed a little: a few cheap hubs, most mid-range)
    vector<int> cost(N + 1);
    for (int i = 1; i <= N; i++) {
        if (rnd.next(0, 4) == 0) cost[i] = rnd.next(1, 4);   // cheap hub
        else                     cost[i] = rnd.next(5, 20);  // ordinary system
    }

    printf("%d %d %d\n", N, M, r);
    for (auto& e : edges) printf("%d %d\n", e.first, e.second);
    for (int i = 1; i <= N; i++) printf("%d\n", cost[i]);
    return 0;
}
