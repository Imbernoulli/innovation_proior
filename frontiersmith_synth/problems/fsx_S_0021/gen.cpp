#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

// Minimum-interference channel assignment (weighted graph coloring under a color
// budget K), skinned as orbital debris-collector transponder assignment.
//
// testId is a difficulty ladder:
//   testId 1 : tiny (example scale, n=6, K=2)
//   testId 2..10 : n grows 40..200, denser interference, small channel budget so
//                  the swarm is NOT K-colorable -> residual static is guaranteed and
//                  greedy / local-search heuristics diverge.

int main(int argc, char** argv) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);

    int n, K;
    long long targetM;
    if (testId == 1) {
        n = 6; K = 2; targetM = 9;
    } else {
        n = 20 * testId;                 // 40, 60, ..., 200
        K = 3 + (testId % 4);            // 3..6
        int avgdeg = 5 + testId;         // denser as testId grows
        targetM = (long long)n * avgdeg / 2;
    }

    // penalty model varies a little per test
    int pHi = 8 + (testId % 3);          // 8..10 co-channel ceiling

    // collect edges (u,v,p,q) with a dedup set to keep the instance clean-ish;
    // duplicates on the guaranteed clique are allowed but we just avoid exact repeats.
    set<pair<int,int>> seen;
    vector<array<int,4>> edges;

    auto addEdge = [&](int u, int v, int p, int q) {
        if (u == v) return false;
        if (u > v) swap(u, v);
        if (seen.count({u, v})) return false;
        seen.insert({u, v});
        edges.push_back({u, v, p, q});
        return true;
    };

    // ---- guaranteed clique of size K+2 so chromatic number > K: at least one
    // co-channel collision must survive any K-channel assignment (F > 0). ----
    {
        int cs = min(n, K + 2);
        vector<int> nodes(n);
        iota(nodes.begin(), nodes.end(), 1);
        shuffle(nodes.begin(), nodes.end());
        for (int i = 0; i < cs; i++)
            for (int j = i + 1; j < cs; j++) {
                int p = rnd.next(pHi - 2, pHi);      // high, dominant penalties
                int q = rnd.next(0, p);
                addEdge(nodes[i], nodes[j], p, q);
            }
    }

    // ---- random interference pairs up to the target density ----
    int guard = 0;
    while ((long long)edges.size() < targetM && guard < 50 * targetM + 1000) {
        guard++;
        int u = rnd.next(1, n), v = rnd.next(1, n);
        if (u == v) continue;
        int p = rnd.next(1, pHi);
        int q = rnd.next(0, p);
        addEdge(u, v, p, q);
    }

    // shuffle edge order so listing order carries no structure
    shuffle(edges.begin(), edges.end());

    int m = (int)edges.size();
    printf("%d %d %d\n", n, m, K);
    for (auto& e : edges)
        printf("%d %d %d %d\n", e[0], e[1], e[2], e[3]);
    return 0;
}
