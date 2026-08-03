#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

int main(int argc, char** argv) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);

    // ---- structure ladder: rigs scattered across the arena floor ----
    // testId 1 is tiny (example scale); grows quadratically to a large, dense
    // interference graph by testId 10.
    int n = 30 * testId * testId;              // 30,120,270,...,3000
    if (n < 4) n = 4;
    if (n > 4000) n = 4000;

    // channel budget kept small so the graph is not freely colorable.
    int C = 3 + (testId % 4);                   // 3..6

    // arena side length keeps rig density (and hence average degree) stable.
    int L = (int)(sqrt((double)n) * 10.0) + 1;
    // interference radius; a couple of tests are made denser/sparser.
    double R = 17.0 + 2.0 * (double)((testId + 1) % 3); // 17,19,21 cycling
    double R2 = R * R;

    // heavy-link probability introduces a few high-penalty "hero rig" pairs,
    // rewarding heuristics that prioritise the costliest conflicts.
    double heavyProb = 0.06 + 0.02 * (double)(testId % 3);

    // ---- place rigs ----
    vector<int> X(n + 1), Y(n + 1);
    for (int i = 1; i <= n; i++) {
        X[i] = rnd.next(0, L);
        Y[i] = rnd.next(0, L);
    }

    // ---- build interference edges within radius ----
    struct E { int u, v, p, q; };
    vector<E> edges;
    edges.reserve(200000);
    long long CAP = 40000;
    for (int i = 1; i <= n && (long long)edges.size() <= CAP + 5000; i++) {
        for (int j = i + 1; j <= n; j++) {
            double dx = X[i] - X[j], dy = Y[i] - Y[j];
            double d2 = dx * dx + dy * dy;
            if (d2 <= R2 && d2 > 0.0) {
                double d = sqrt(d2);
                int p;
                if (rnd.next(0.0, 1.0) < heavyProb) {
                    p = rnd.next(200, 1000);           // heavy hero-rig conflict
                } else {
                    p = 1 + (int)((R - d) * 3.0);      // closer -> stronger, ~1..52
                    if (p < 1) p = 1;
                    if (p > 1000) p = 1000;
                }
                int q = p / 3; if (q < 1) q = 1; if (q > p) q = p;
                edges.push_back({i, j, p, q});
            }
        }
    }

    // if the geometry produced too many edges, keep a random subset.
    if ((long long)edges.size() > CAP) {
        shuffle(edges.begin(), edges.end());
        edges.resize(CAP);
    }
    // guarantee at least one edge for tiny degenerate cases.
    if (edges.empty()) {
        int u = 1, v = 2;
        edges.push_back({u, v, 10, 3});
    }

    shuffle(edges.begin(), edges.end());

    int m = (int)edges.size();
    printf("%d %d %d\n", n, m, C);
    for (auto& e : edges)
        printf("%d %d %d %d\n", e.u, e.v, e.p, e.q);
    return 0;
}
