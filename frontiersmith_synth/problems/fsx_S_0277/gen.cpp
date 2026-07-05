// testlib generator for "Draining the Deluge: Reservoir Junction Interdiction"
// Narrow-grid reservoir mesh with a cheap low-cost corridor and heavy detour pipes.
// Node-deletion shortest-path interdiction. testId 1..10 is a difficulty ladder.
#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

int main(int argc, char* argv[]) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);

    // difficulty ladder: (rows R, cols C)
    int R, C;
    switch (testId) {
        case 1:  R = 4;   C = 3; break;
        case 2:  R = 7;   C = 3; break;
        case 3:  R = 12;  C = 4; break;
        case 4:  R = 22;  C = 4; break;
        case 5:  R = 40;  C = 5; break;
        case 6:  R = 65;  C = 5; break;
        case 7:  R = 100; C = 5; break;
        case 8:  R = 150; C = 6; break;
        case 9:  R = 210; C = 6; break;
        default: R = 260; C = 6; break;
    }

    int n = R * C;
    auto id = [&](int r, int c) { return r * C + c + 1; };
    int s = id(0, 0);
    int t = id(R - 1, C - 1);

    // node removal costs: corridor nodes get a low-but-varied cost so a budgeted
    // knapsack over which junctions to shut is genuinely non-trivial; the rest are
    // expensive so shutting them is rarely worthwhile.
    vector<int> cost(n + 1, 0);
    for (int r = 0; r < R; r++)
        for (int c = 0; c < C; c++) {
            int v = id(r, c);
            if (c == 0) cost[v] = rnd.next(1, 5);          // corridor column: cheap, varied
            else        cost[v] = rnd.next(12, 50);        // rest: expensive
        }

    // edges. cheap "L-shaped" corridor: down column 0, then across the bottom row.
    // corridor pipes have transit 1; every other pipe is a heavy detour.
    struct E { int u, v, w; };
    vector<E> edges;
    auto isCorridor = [&](int r1, int c1, int r2, int c2) -> bool {
        // vertical edge inside column 0
        if (c1 == 0 && c2 == 0) return true;
        // horizontal edge inside bottom row
        if (r1 == R - 1 && r2 == R - 1) return true;
        return false;
    };

    // vertical pipes
    for (int r = 0; r + 1 < R; r++)
        for (int c = 0; c < C; c++) {
            int w = isCorridor(r, c, r + 1, c) ? 1 : rnd.next(15, 100);
            edges.push_back({id(r, c), id(r + 1, c), w});
        }
    // horizontal pipes
    for (int r = 0; r < R; r++)
        for (int c = 0; c + 1 < C; c++) {
            int w = isCorridor(r, c, r, c + 1) ? 1 : rnd.next(15, 100);
            edges.push_back({id(r, c), id(r, c + 1), w});
        }

    // a handful of heavy long-range "bypass" pipes (grid-plus-shortcut mesh) so the
    // network stays robustly connected under node deletion and paths are less regular.
    int extra = min(n / 4, 3 * C + testId);
    for (int e = 0; e < extra; e++) {
        int a = rnd.next(1, n), b = rnd.next(1, n);
        if (a == b) { e--; continue; }
        edges.push_back({a, b, rnd.next(30, 100)});
    }

    // budget: enough to shut several corridor junctions but not all of them, so the
    // choice of which subset to shut actually matters.
    int corridorNodes = R - 2;               // interior of column-0 corridor
    int target = max(2, min(corridorNodes, 3 + R / 12));
    int K = target * rnd.next(3, 5);

    // emit
    printf("%d %d %d %d %d\n", n, (int)edges.size(), s, t, K);
    for (int i = 1; i <= n; i++) printf("%d%c", cost[i], i == n ? '\n' : ' ');
    // shuffle edge order so indices are not structured
    shuffle(edges.begin(), edges.end());
    for (auto& e : edges) printf("%d %d %d\n", e.u, e.v, e.w);
    return 0;
}
