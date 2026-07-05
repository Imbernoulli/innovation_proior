#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

int main(int argc, char** argv) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);

    // ---- structure ladder: pandemic contact network ----
    // testId 1 tiny (example scale); grows denser / larger by testId 10.
    int n = 6 + (testId - 1) * 20;          // 6 .. 186
    int K = 3 + (testId % 6);               // 3..8 cohorts
    int Wmax = 9;
    int dmax = 3;

    vector<array<int, 4>> edges;            // u, v, w, d

    auto addEdge = [&](int u, int v, int w, int d) {
        if (u == v) return;
        edges.push_back({u, v, w, d});
    };

    // 1) tightly-coupled clusters (households / wards): cliques of size K+1.
    //    K+1 people over K cohorts -> pigeonhole forces residual conflict for
    //    ANY assignment, so F>0 and B>0 are guaranteed and the instance is
    //    genuinely over-constrained (no perfect coloring exists).
    int cliqueSize = K + 1;
    int nClusters = max(1, n / (cliqueSize * 2));
    for (int c = 0; c < nClusters; c++) {
        set<int> s;
        while ((int)s.size() < cliqueSize) s.insert(rnd.next(1, n));
        vector<int> vs(s.begin(), s.end());
        for (int a = 0; a < (int)vs.size(); a++)
            for (int b = a + 1; b < (int)vs.size(); b++)
                addEdge(vs[a], vs[b], rnd.next(4, Wmax), rnd.next(1, dmax));
    }

    // 2) diffuse community contacts (random graph, lighter demands).
    int avgDeg = 4 + testId / 2;
    int mRand = n * avgDeg / 2;
    for (int e = 0; e < mRand; e++) {
        int a = rnd.next(1, n), b = rnd.next(1, n);
        addEdge(a, b, rnd.next(1, Wmax), rnd.next(1, 2));
    }

    // shuffle so edge order carries no structural hint
    shuffle(edges.begin(), edges.end());

    int m = (int)edges.size();
    printf("%d %d %d\n", n, m, K);
    for (auto& e : edges)
        printf("%d %d %d %d\n", e[0], e[1], e[2], e[3]);
    return 0;
}
