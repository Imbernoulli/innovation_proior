#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

int main(int argc, char** argv) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);

    // ---- structure ladder: grid of bakery supply hubs ----
    // testId 1 is tiny (example scale); grows to a large, dense grid by testId 10.
    int side = 3 + 2 * (testId - 1);        // 3, 5, 7, ..., 21
    int R = side, C = side;
    int n = R * C;

    auto node = [&](int i, int j) { return i * C + j + 1; };

    // weight model varies per test so the shortest "flour highway" is cheap
    // while detours are expensive -> interdiction is meaningful.
    int cheapHi = 6 + (testId % 3) * 2;     // 6..10 cheap-road ceiling
    int heavyLo = 20 + (testId % 4) * 5;    // 20..35 expensive-road floor
    int heavyHi = heavyLo + 15;
    double heavyProb = 0.08 + 0.02 * (testId % 3); // fraction of costly roads

    vector<array<int, 3>> edges; // u, v, w

    auto pushEdge = [&](int u, int v) {
        int w;
        if (rnd.next(0.0, 1.0) < heavyProb)
            w = rnd.next(heavyLo, heavyHi);
        else
            w = rnd.next(1, cheapHi);
        edges.push_back({u, v, w});
    };

    // grid roads (right + down)
    for (int i = 0; i < R; i++)
        for (int j = 0; j < C; j++) {
            if (j + 1 < C) pushEdge(node(i, j), node(i, j + 1));
            if (i + 1 < R) pushEdge(node(i, j), node(i + 1, j));
        }

    // extra long-range shortcut roads add parallel paths (more open-ended search)
    int extra = 2 * testId;
    for (int e = 0; e < extra; e++) {
        int a = rnd.next(1, n), b = rnd.next(1, n);
        if (a == b) { e--; continue; }
        int w = rnd.next(1, cheapHi + 4);
        edges.push_back({a, b, w});
    }

    int s = 1, t = n;
    int k = testId + 1;                     // budget of closable links

    // shuffle edge order so index != structural position
    shuffle(edges.begin(), edges.end());

    int m = (int)edges.size();
    printf("%d %d %d %d %d\n", n, m, s, t, k);
    for (auto& e : edges)
        printf("%d %d %d\n", e[0], e[1], e[2]);
    return 0;
}
