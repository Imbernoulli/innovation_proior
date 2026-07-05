#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

int main(int argc, char** argv) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);

    // ---- structure ladder: grid of aquarium plumbing junctions ----
    // testId 1 is tiny (example scale); grows to a large, dense grid by testId 10.
    int side = 3 + 2 * (testId - 1);        // 3, 5, 7, ..., 21
    int R = side, C = side;
    int n = R * C;

    auto node = [&](int i, int j) { return i * C + j + 1; };

    // weight model varies per test so a cheap "clean-water trunk" exists while
    // detour pipes are expensive -> interdiction meaningfully lengthens routes.
    int cheapHi = 5 + (testId % 3) * 2;     // 5..9 cheap-pipe ceiling
    int heavyLo = 22 + (testId % 4) * 4;    // 22..34 expensive-pipe floor
    int heavyHi = heavyLo + 18;
    double heavyProb = 0.10 + 0.03 * (testId % 3); // fraction of costly pipes

    vector<array<int, 3>> edges; // u, v, w

    auto pushEdge = [&](int u, int v) {
        int w;
        if (rnd.next(0.0, 1.0) < heavyProb)
            w = rnd.next(heavyLo, heavyHi);
        else
            w = rnd.next(1, cheapHi);
        edges.push_back({u, v, w});
    };

    // grid pipes (right + down)
    for (int i = 0; i < R; i++)
        for (int j = 0; j < C; j++) {
            if (j + 1 < C) pushEdge(node(i, j), node(i, j + 1));
            if (i + 1 < R) pushEdge(node(i, j), node(i + 1, j));
        }

    // extra long-range bypass pipes add parallel routes (richer search space)
    int extra = 2 * testId;
    for (int e = 0; e < extra; e++) {
        int a = rnd.next(1, n), b = rnd.next(1, n);
        if (a == b) { e--; continue; }
        int w = rnd.next(1, cheapHi + 4);
        edges.push_back({a, b, w});
    }

    int s = 1;                              // sump pump at the top-left junction
    int k = 1 + testId;                     // valve budget: 2 .. 11

    // number of display tanks grows with the ladder
    int p = min(12, 2 + testId / 2);        // 2 .. up to ~7

    // pick p distinct tank nodes (all != s), biased toward far junctions so
    // interdiction has room to bite.
    set<int> chosen;
    while ((int)chosen.size() < p) {
        int cand;
        if (rnd.next(0, 1) == 0)
            cand = rnd.next(n / 2 + 1, n);  // far half
        else
            cand = rnd.next(2, n);          // anywhere but the pump
        if (cand != s) chosen.insert(cand);
    }
    vector<int> tanks(chosen.begin(), chosen.end());
    shuffle(tanks.begin(), tanks.end());

    // shuffle edge order so index != structural position
    shuffle(edges.begin(), edges.end());

    int m = (int)edges.size();
    printf("%d %d %d %d %d\n", n, m, s, p, k);
    for (auto& e : edges)
        printf("%d %d %d\n", e[0], e[1], e[2]);
    for (int i = 0; i < p; i++)
        printf("%d%c", tanks[i], i + 1 == p ? '\n' : ' ');
    return 0;
}
