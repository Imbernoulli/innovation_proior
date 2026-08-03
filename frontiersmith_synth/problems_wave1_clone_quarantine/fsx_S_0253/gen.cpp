#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

int main(int argc, char** argv) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);

    // ---- structure ladder: grid of habitat patches ----
    // testId 1 is tiny (example scale); grows to a large, dense mosaic by testId 10.
    int side = 3 + 2 * (testId - 1);        // 3, 5, 7, ..., 21
    int R = side, C = side;
    int n = R * C;

    auto node = [&](int i, int j) { return i * C + j + 1; };

    // Length model: cheap corridors dominate, sparse expensive corridors form
    // the only long detours -> interdiction is meaningful.
    int cheapHi = 5 + (testId % 3) * 2;     // 5..9 cheap-corridor length ceiling
    int heavyLo = 22 + (testId % 4) * 4;    // 22..34 expensive-corridor floor
    int heavyHi = heavyLo + 14;
    double heavyProb = 0.10 + 0.02 * (testId % 3); // fraction of long detour corridors

    // Clearing-cost model: independent of length so a knapsack tension appears
    // (some cheap-to-walk corridors are expensive to clear and vice versa).
    int costHi = 6 + (testId % 5) * 2;      // 6..14 clearing-cost ceiling

    vector<array<int, 4>> edges; // u, v, w(length), c(clearing cost)

    auto pushEdge = [&](int u, int v) {
        int w;
        if (rnd.next(0.0, 1.0) < heavyProb)
            w = rnd.next(heavyLo, heavyHi);
        else
            w = rnd.next(1, cheapHi);
        int c = rnd.next(1, costHi);
        edges.push_back({u, v, w, c});
    };

    // grid corridors (right + down)
    for (int i = 0; i < R; i++)
        for (int j = 0; j < C; j++) {
            if (j + 1 < C) pushEdge(node(i, j), node(i, j + 1));
            if (i + 1 < R) pushEdge(node(i, j), node(i + 1, j));
        }

    // extra long-range shortcut corridors add parallel routes (more open-ended search)
    int extra = 2 * testId;
    for (int e = 0; e < extra; e++) {
        int a = rnd.next(1, n), b = rnd.next(1, n);
        if (a == b) { e--; continue; }
        int w = rnd.next(1, cheapHi + 3);
        int c = rnd.next(1, costHi + 2);
        edges.push_back({a, b, w, c});
    }

    int s = 1, t = n;
    // budget scales with the instance so several (but not unlimited) clearings fit
    int budget = 8 + 7 * testId;            // 15, 22, ..., 78

    // shuffle edge order so index != structural position
    shuffle(edges.begin(), edges.end());

    int m = (int)edges.size();
    printf("%d %d %d %d %d\n", n, m, s, t, budget);
    for (auto& e : edges)
        printf("%d %d %d %d\n", e[0], e[1], e[2], e[3]);
    return 0;
}
