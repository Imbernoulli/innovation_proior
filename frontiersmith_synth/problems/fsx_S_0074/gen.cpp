#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

int main(int argc, char** argv) {
    registerGen(argc, argv, 1);
    int tid = atoi(argv[1]);

    // ---- scale ladder: tiny 3x3 grid at tid=1, up to 20x20 at tid=10 ----
    int side = 3 + 2 * (tid - 1);   // 3,5,7,...,21
    if (side > 20) side = 20;
    if (side < 3) side = 3;
    int H = side, W = side;

    // ---- obstacle (crevasse) density varies across tests ----
    // more crevasses = more distorted, non-tree coverage balls
    double p = 0.10 + 0.03 * (tid % 6);   // 0.10 .. 0.25

    vector<string> grid(H, string(W, '.'));
    for (int i = 0; i < H; i++)
        for (int j = 0; j < W; j++)
            if (rnd.next(0.0, 1.0) < p) grid[i][j] = '#';

    // number habitable sectors row-major; guarantee at least one open cell
    vector<vector<int>> id(H, vector<int>(W, -1));
    int n = 0;
    for (int i = 0; i < H; i++)
        for (int j = 0; j < W; j++)
            if (grid[i][j] == '.') id[i][j] = ++n;
    if (n == 0) { grid[0][0] = '.'; id[0][0] = ++n; }

    // ---- radii: mostly 1, some 2, sparse radius-3 hubs (keeps balls from covering all) ----
    vector<int> rad(n + 1, 1);
    for (int v = 1; v <= n; v++) {
        if (rnd.next(0, 3) == 0) rad[v] = 2;   // ~25% radius 2
    }
    int hubs = 1 + n / 40;                       // a few radius-3 hubs
    for (int h = 0; h < hubs; h++) {
        int v = rnd.next(1, n);
        rad[v] = 3;
    }

    // ---- costs: a bigger radius needs a bigger generator: cost = base*(1+rad)^2 ----
    // keeps a high-radius relay only a MODEST win over several small ones.
    vector<int> cost(n + 1, 1);
    for (int v = 1; v <= n; v++) {
        int base = rnd.next(2, 12);
        int val = base * (1 + rad[v]) * (1 + rad[v]);   // up to 12*16 = 192
        if (val < 1) val = 1;
        if (val > 200) val = 200;
        cost[v] = val;
    }

    // ---- emit ----
    printf("%d %d\n", H, W);
    for (int i = 0; i < H; i++) printf("%s\n", grid[i].c_str());
    printf("%d\n", n);
    for (int v = 1; v <= n; v++) printf("%d%c", cost[v], v == n ? '\n' : ' ');
    for (int v = 1; v <= n; v++) printf("%d%c", rad[v], v == n ? '\n' : ' ');
    return 0;
}
