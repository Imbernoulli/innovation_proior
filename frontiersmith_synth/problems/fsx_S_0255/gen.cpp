#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef pair<int,int> pii;

// generate a random connected polyomino of exactly s cells, anchored near origin
static vector<pii> randPoly(int s) {
    set<pii> cells;
    cells.insert({0,0});
    int dx[4] = {1,-1,0,0}, dy[4] = {0,0,1,-1};
    while ((int)cells.size() < s) {
        vector<pii> vc(cells.begin(), cells.end());
        pii base = vc[rnd.next(0, (int)vc.size()-1)];
        int dd = rnd.next(0,3);
        cells.insert({base.first + dx[dd], base.second + dy[dd]});
    }
    return vector<pii>(cells.begin(), cells.end());
}

int main(int argc, char** argv) {
    registerGen(argc, argv, 1);
    int t = atoi(argv[1]);

    // difficulty ladder: plot grows, more bay types, deeper-regolith requirements
    int W = 8 + (t - 1) * 6;          // 8 .. 62
    int H = 8 + (t - 1) * 6;
    if (W > 60) W = 60;
    if (H > 60) H = 60;
    int P = 3 + (t - 1) / 2;          // 3 .. 7
    if (P > 8) P = 8;

    // bedrock fraction and deep-regolith fraction evolve with t
    double rockP = 0.05 + 0.010 * (t - 1);   // ~0.05 .. ~0.14
    if (rockP > 0.18) rockP = 0.18;
    double deepP = 0.14 + 0.010 * (t - 1);   // fraction of deep pockets, ~0.14 .. ~0.23

    // --- regolith-depth grid first (need buildable count to size stock) ---
    // background is shallow (1..3); deep pockets (4..9) are scarce and give shielding.
    vector<vector<int>> d(H, vector<int>(W, 0));
    int buildable = 0;
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++) {
            if (rnd.next(0.0, 1.0) < rockP) { d[y][x] = 0; continue; }   // bedrock
            if (rnd.next(0.0, 1.0) < deepP) d[y][x] = rnd.next(4, 9);    // deep pocket
            else                            d[y][x] = rnd.next(1, 3);    // shallow background
            buildable++;
        }
    // guarantee an open shallow region so the type-0 baseline always installs copies.
    for (int y = 0; y < min(3,H); y++)
        for (int x = 0; x < min(3,W); x++)
            if (d[y][x] == 0) { d[y][x] = rnd.next(1, 3); buildable++; }

    // --- bay types (bounded-knapsack regime: total coverable area << plot) ---
    double targetTotal = 0.40 * (double)buildable;   // total footprint budget across all stock
    vector<vector<pii>> shapes;
    vector<int> stock, wv, rv;

    // type 0: fixed L-tromino, low value, minimal shielding -> the value-blind baseline.
    shapes.push_back({{0,0},{0,1},{1,1}});
    wv.push_back(2);
    rv.push_back(1);
    int stock0 = max(3, (int)llround(0.45 * targetTotal / 3.0));
    stock.push_back(stock0);

    // remaining budget over the other types; higher shielding -> higher value, scarcer stock.
    double restArea = 0.55 * targetTotal;
    for (int i = 1; i < P; i++) {
        int s = rnd.next(3, 6);
        shapes.push_back(randPoly(s));
        // shielding requirement grows across types (needs deep regolith)
        int r = 2 + rnd.next(0, min(6, 1 + (i * 6) / max(1, P - 1)));
        if (r > 9) r = 9;
        rv.push_back(r);
        // value rewards deep+large bays; keep within [1,60]
        int w = min(60, s * 2 + r * 3 + rnd.next(0, 4));
        wv.push_back(w);
        double share = restArea / (double)(P - 1);
        int c = max(1, (int)llround(share / (double)s));
        if (i % 2 == 1) c = max(1, c / 2);   // roughly half the extra types scarce
        stock.push_back(c);
    }

    // --- print ---
    printf("%d %d\n", W, H);
    printf("%d\n", P);
    for (int i = 0; i < P; i++) {
        printf("%d %d %d %d\n", wv[i], rv[i], stock[i], (int)shapes[i].size());
        for (auto &c : shapes[i]) printf("%d %d\n", c.first, c.second);
    }
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            printf("%d", d[y][x]);
            printf(x + 1 < W ? " " : "\n");
        }
    }
    return 0;
}
