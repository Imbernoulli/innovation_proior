#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

// Interstellar relay channel assignment: minimum-conflict graph labeling.
// testId is a difficulty ladder: testId 1 tiny (example scale), growing to a large,
// dense interference graph with a small channel budget (harder) by testId 10.
int main(int argc, char** argv) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);

    int N = 6 + 16 * (testId - 1);        // 6, 22, 38, ..., 150
    int C = 3 + ((testId - 1) % 4);       // channel budget cycles 3,4,5,6
    int avgDeg = 3 + testId / 2;          // interference density grows: 3..8

    // interference weights: mostly small, occasional heavy crosstalk (heavy tail)
    auto w_rand = [&]() -> int {
        if (rnd.next(0, 9) == 0) return (int)rnd.next(20, 100);
        return (int)rnd.next(1, 20);
    };

    vector<array<int, 3>> edges; // u, v, w

    // random overlapping pairs
    long long target = (long long)N * avgDeg / 2;
    for (long long e = 0; e < target; e++) {
        int u = rnd.next(1, N), v = rnd.next(1, N);
        if (u == v) { e--; continue; }
        edges.push_back({u, v, w_rand()});
    }

    // embed overlap cliques LARGER than the channel budget -> unavoidable conflict,
    // rewarding assignments that spread the load well (open-ended structure).
    int numCliques = 1 + testId / 3;
    for (int c = 0; c < numCliques; c++) {
        int sz = C + 1 + (int)rnd.next(0, 2); // C+1 .. C+3 stations
        sz = min(sz, N);
        vector<int> nodes;
        set<int> used;
        while ((int)nodes.size() < sz) {
            int x = (int)rnd.next(1, N);
            if (used.insert(x).second) nodes.push_back(x);
        }
        for (int i = 0; i < sz; i++)
            for (int j = i + 1; j < sz; j++)
                edges.push_back({nodes[i], nodes[j], w_rand()});
    }

    if (edges.empty()) edges.push_back({1, min(2, N), 10});

    // shuffle so listed order != structural position
    shuffle(edges.begin(), edges.end());

    int M = (int)edges.size();
    printf("%d %d %d\n", N, M, C);
    for (auto& e : edges) printf("%d %d %d\n", e[0], e[1], e[2]);
    return 0;
}
