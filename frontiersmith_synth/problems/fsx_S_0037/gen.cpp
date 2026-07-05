#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

int main(int argc, char** argv) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);

    // ---- structure ladder: grid of carnival ride junctions ----
    // testId 1 tiny (example scale); grows to a larger, denser midway by testId 10.
    int side = 3 + (testId - 1);            // 3,4,...,12  (small scale)
    int R = side, C = side;
    int n = R * C;

    auto node = [&](int i, int j) { return i * C + j + 1; };

    // travel-time model: a few cheap "main drag" walkways, some heavy detours.
    int cheapHi = 5 + (testId % 3) * 2;     // 5..9  cheap-walk ceiling
    int heavyLo = 18 + (testId % 4) * 4;    // 18..30 slow-detour floor
    int heavyHi = heavyLo + 12;
    double heavyProb = 0.10 + 0.02 * (testId % 3);

    // barricade-cost model: costs are anti-correlated-ish with travel time so
    // that cost-greedy and delay-greedy diverge (fast walkways cost more to close).
    struct E { int u, v, w, c; };
    vector<E> edges;

    auto pushEdge = [&](int u, int v) {
        int w;
        bool heavy = (rnd.next(0.0, 1.0) < heavyProb);
        if (heavy) w = rnd.next(heavyLo, heavyHi);
        else       w = rnd.next(1, cheapHi);
        // cheap-to-walk (fast) walkways tend to be pricier to barricade
        int c;
        if (heavy) c = rnd.next(1, 4);
        else       c = rnd.next(3, 10);
        edges.push_back({u, v, w, c});
    };

    // grid walkways (right + down)
    for (int i = 0; i < R; i++)
        for (int j = 0; j < C; j++) {
            if (j + 1 < C) pushEdge(node(i, j), node(i, j + 1));
            if (i + 1 < R) pushEdge(node(i, j), node(i + 1, j));
        }

    // extra long-range shortcut walkways -> more alternate routes, richer search
    int extra = testId + 2;
    for (int e = 0; e < extra; e++) {
        int a = rnd.next(1, n), b = rnd.next(1, n);
        if (a == b) { e--; continue; }
        int w = rnd.next(1, cheapHi + 3);
        int c = rnd.next(2, 9);
        edges.push_back({a, b, w, c});
    }

    int s = 1, t = n;

    // barricade budget: enough to close several cheap walkways, never everything.
    int P = 6 + 3 * testId;                 // 9,12,...,36

    // shuffle edge order so index != structural position
    shuffle(edges.begin(), edges.end());

    int m = (int)edges.size();
    printf("%d %d %d %d %d\n", n, m, s, t, P);
    for (auto& e : edges)
        printf("%d %d %d %d\n", e.u, e.v, e.w, e.c);
    return 0;
}
