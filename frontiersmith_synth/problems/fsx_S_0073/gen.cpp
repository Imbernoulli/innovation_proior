#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

int main(int argc, char** argv) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);

    // ---- structure ladder: grid of cave chambers ----
    // testId 1 is tiny (example scale); grows to a large grid by testId 10.
    int side = 3 + 2 * (testId - 1);        // 3, 5, 7, ..., 21
    int R = side, C = side;
    int n = R * C;

    auto node = [&](int i, int j) { return i * C + j + 1; };

    // weight model: a cheap "main gallery" with occasional expensive squeezes so
    // that forcing a detour (by collapsing chambers) is meaningful.
    int cheapHi = 5 + (testId % 3) * 2;     // 5..9 cheap-passage ceiling
    int heavyLo = 22 + (testId % 4) * 5;    // 22..37 tight-squeeze floor
    int heavyHi = heavyLo + 18;
    double heavyProb = 0.10 + 0.02 * (testId % 3); // fraction of costly passages

    vector<array<int, 3>> edges; // u, v, w

    auto pushEdge = [&](int u, int v) {
        int w;
        if (rnd.next(0.0, 1.0) < heavyProb)
            w = rnd.next(heavyLo, heavyHi);
        else
            w = rnd.next(1, cheapHi);
        edges.push_back({u, v, w});
    };

    // grid passages (right + down) -> interior chambers are 2-connected,
    // so most single collapses keep mouth-vault connectivity.
    for (int i = 0; i < R; i++)
        for (int j = 0; j < C; j++) {
            if (j + 1 < C) pushEdge(node(i, j), node(i, j + 1));
            if (i + 1 < R) pushEdge(node(i, j), node(i + 1, j));
        }

    // extra long-range crawlways add parallel routes -> richer search space.
    int extra = 2 * testId + 1;
    for (int e = 0; e < extra; e++) {
        int a = rnd.next(1, n), b = rnd.next(1, n);
        if (a == b) { e--; continue; }
        int w = rnd.next(1, cheapHi + 3);
        edges.push_back({a, b, w});
    }

    int s = 1, t = n;                        // opposite corners
    int k = testId + 1;                      // budget of collapsible chambers

    // shuffle edge order so index != structural position
    shuffle(edges.begin(), edges.end());

    int m = (int)edges.size();
    printf("%d %d %d %d %d\n", n, m, s, t, k);
    for (auto& e : edges)
        printf("%d %d %d\n", e[0], e[1], e[2]);
    return 0;
}
