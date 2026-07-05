#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

// Glacier Sensor Net: maximum polyomino coverage packing.
// testId is a size/difficulty ladder: tiny obstructed board at 1, large denser board at 10.
int main(int argc, char** argv) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);

    // ---- board size ladder (medium scale) ----
    int W = 6 + 5 * (testId - 1);   // 6, 11, 16, ..., 51
    int H = 5 + 4 * (testId - 1);   // 5, 9, 13, ..., 41
    if (W > 60) W = 60;
    if (H > 60) H = 60;

    // ---- crevasse (blocked) cells ----
    // density grows a little with testId; boards stay far from fully blocked.
    double dens = 0.06 + 0.012 * (testId % 5);   // 0.06 .. ~0.11
    int total = W * H;
    int Kwant = (int)llround(dens * total);

    // sample distinct blocked cells; leave a guaranteed clear 3x3 pocket at the
    // top-left so footprint 0 (the plus pentomino) always fits at least once.
    set<pair<int,int>> blocked;
    auto inClearPocket = [&](int r, int c) { return r < 3 && c < 3; };
    int guard = 0;
    while ((int)blocked.size() < Kwant && guard < 50 * total) {
        guard++;
        int r = rnd.next(0, H - 1);
        int c = rnd.next(0, W - 1);
        if (inClearPocket(r, c)) continue;   // keep the pocket solid
        blocked.insert({r, c});
    }

    // ---- fixed catalogue of cluster footprints ----
    // Footprint 0 is the plus pentomino: a deliberately poor tiler, so the
    // single-shape baseline is weak and richer coverings score well above it.
    vector<vector<pair<int,int>>> shapes = {
        {{0,1},{1,0},{1,1},{1,2},{2,1}},   // 0: plus pentomino (5)
        {{0,0},{0,1},{1,0},{1,1}},         // 1: O tetromino (4)
        {{0,0},{0,1},{0,2}},               // 2: I tromino (3)
        {{0,0},{1,0},{1,1}},               // 3: L tromino (3)
        {{0,0},{0,1},{0,2},{1,1}},         // 4: T tetromino (4)
        {{0,0},{1,0},{2,0},{2,1}},         // 5: L tetromino (4)
    };

    // ---- emit ----
    printf("%d %d\n", W, H);
    printf("%d\n", (int)blocked.size());
    for (auto& p : blocked) printf("%d %d\n", p.first, p.second);

    printf("%d\n", (int)shapes.size());
    for (auto& s : shapes) {
        printf("%d", (int)s.size());
        for (auto& cell : s) printf(" %d %d", cell.first, cell.second);
        printf("\n");
    }
    return 0;
}
