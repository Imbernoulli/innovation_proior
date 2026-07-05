#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef pair<int,int> pii;

// generate a random connected polyomino of exactly s cells, anchored near origin
static vector<pii> randPoly(int s) {
    set<pii> cells;
    cells.insert({0,0});
    vector<pii> frontier = {{0,0}};
    int dx[4] = {1,-1,0,0}, dy[4] = {0,0,1,-1};
    while ((int)cells.size() < s) {
        // pick a random existing cell and try to grow
        vector<pii> vc(cells.begin(), cells.end());
        pii base = vc[rnd.next(0, (int)vc.size()-1)];
        int d = rnd.next(0,3);
        pii nc = {base.first + dx[d], base.second + dy[d]};
        cells.insert(nc);
    }
    return vector<pii>(cells.begin(), cells.end());
}

int main(int argc, char** argv) {
    registerGen(argc, argv, 1);
    int t = atoi(argv[1]);

    // difficulty ladder: grid grows, more types, heavier value skew
    int W = 8 + (t - 1) * 4;          // 8 .. 44
    int H = 8 + (t - 1) * 4;          // 8 .. 44
    if (W > 50) W = 50;
    if (H > 50) H = 50;
    int P = 2 + (t + 1) / 2;          // 3 .. 7  (P>=2)
    if (P > 8) P = 8;

    // crevasse density and value-skew fraction grow with t
    double crevP = 0.05 + 0.012 * (t - 1);      // ~0.05 .. ~0.16
    if (crevP > 0.20) crevP = 0.20;
    double hiP   = 0.08 + 0.02 * (t - 1);       // fraction of high-value cells

    // --- heat grid first (need placeable count to size stock) ---
    vector<vector<int>> v(H, vector<int>(W, 0));
    int placeable = 0;
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++) {
            if (rnd.next(0.0, 1.0) < crevP) { v[y][x] = 0; continue; }
            if (rnd.next(0.0, 1.0) < hiP)   v[y][x] = rnd.next(40, 99);   // high-value patch
            else                            v[y][x] = rnd.next(1, 6);     // low background
            placeable++;
        }
    // guarantee an open region for the type-0 baseline: clear a 3x3 block of crevasses
    for (int y = 0; y < min(3,H); y++)
        for (int x = 0; x < min(3,W); x++)
            if (v[y][x] == 0) { v[y][x] = rnd.next(1, 6); placeable++; }

    // --- module types (bounded-knapsack regime: total coverable area << board) ---
    // total coverable area targeted at ~35% of placeable so WHICH cells you cover matters.
    double targetTotal = 0.35 * (double)placeable;
    vector<vector<pii>> shapes;
    vector<int> stock;

    // type 0: fixed L-tromino, value-blind baseline; its stock covers ~15% of placeable.
    shapes.push_back({{0,0},{0,1},{1,1}});
    int stock0 = max(2, (int)llround(0.40 * targetTotal / 3.0));
    stock.push_back(stock0);

    // remaining ~60% of the area budget spread over the other types.
    double restArea = 0.60 * targetTotal;
    for (int i = 1; i < P; i++) {
        int s = rnd.next(3, 6);
        shapes.push_back(randPoly(s));
        double share = restArea / (double)(P - 1);
        int c = max(1, (int)llround(share / (double)s));
        // make roughly half the extra types scarce for knapsack tension
        if (i % 2 == 1) c = max(1, c / 2);
        stock.push_back(c);
    }

    // --- print ---
    printf("%d %d\n", W, H);
    printf("%d\n", P);
    for (int i = 0; i < P; i++) {
        printf("%d %d\n", stock[i], (int)shapes[i].size());
        for (auto &c : shapes[i]) printf("%d %d\n", c.first, c.second);
    }
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            printf("%d", v[y][x]);
            printf(x + 1 < W ? " " : "\n");
        }
    }
    return 0;
}
