#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef pair<int,int> P;

// generate a random connected polyomino of size s, returned normalized (min offsets 0)
vector<P> randPoly(int s) {
    set<P> cells;
    cells.insert({0,0});
    vector<P> frontier = {{1,0},{-1,0},{0,1},{0,-1}};
    while ((int)cells.size() < s) {
        // collect candidate empty cells adjacent to current set
        set<P> cand;
        int dx4[4]={1,-1,0,0}, dy4[4]={0,0,1,-1};
        for (auto &c : cells)
            for (int k=0;k<4;k++){
                P np={c.first+dx4[k], c.second+dy4[k]};
                if (!cells.count(np)) cand.insert(np);
            }
        vector<P> cv(cand.begin(), cand.end());
        P pick = cv[rnd.next((int)cv.size())];
        cells.insert(pick);
    }
    int mnx=INT_MAX, mny=INT_MAX;
    for (auto &c : cells){ mnx=min(mnx,c.first); mny=min(mny,c.second);}
    vector<P> out;
    for (auto &c : cells) out.push_back({c.first-mnx, c.second-mny});
    sort(out.begin(), out.end());
    return out;
}

int main(int argc, char** argv) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);

    // ---- structure ladder ----
    int W = 6 + (testId - 1) * 2 / 3;   // 6 .. ~12
    int H = 6 + (testId - 1) * 2 / 3;
    if (W > 12) W = 12; if (H > 12) H = 12;
    // late tests: make non-square to add asymmetry
    if (testId >= 7) { W = 12; H = 10; }
    if (testId == 10) { W = 11; H = 12; }

    bool dense = (testId % 2 == 0);          // even: more hotspots
    bool heavyStock = (testId >= 6);         // later: larger stock -> more room to beat baseline

    // ---- value grid ----
    vector<vector<int>> val(H, vector<int>(W));
    for (int y=0;y<H;y++)
        for (int x=0;x<W;x++)
            val[y][x] = rnd.next(1, 20);
    // hotspots
    int nHot = (dense ? rnd.next(4,7) : rnd.next(2,4)) + testId/3;
    for (int i=0;i<nHot;i++){
        int hx = rnd.next(0, W-1), hy = rnd.next(0, H-1);
        val[hy][hx] = rnd.next(60, 99);
    }

    // ---- clear a guaranteed 3x3 region for design-0 baseline to fit ----
    int cx = rnd.next(0, W-3), cy = rnd.next(0, H-3);
    set<P> clearReg;
    for (int dy=0;dy<3;dy++) for (int dx=0;dx<3;dx++) clearReg.insert({cx+dx, cy+dy});

    // ---- vents (avoid the clear region) ----
    int maxV = (W*H*14)/100;
    int K = rnd.next(0, maxV);
    set<P> vents;
    int tries = 0;
    while ((int)vents.size() < K && tries < 10000) {
        tries++;
        int vx = rnd.next(0, W-1), vy = rnd.next(0, H-1);
        if (clearReg.count({vx,vy})) continue;
        vents.insert({vx,vy});
    }

    // ---- pod designs ----
    int Pn = 2 + (testId - 1) / 3;   // 2 .. 5
    if (Pn > 5) Pn = 5;

    // design 0: small footprint, small stock. L-tromino (or domino for tiny test).
    vector<vector<P>> shapes;
    vector<int> stock;
    // design 0
    vector<P> d0 = {{0,0},{1,0},{0,1}}; // L-tromino, 3 cells
    shapes.push_back(d0);
    stock.push_back(rnd.next(1, 1 + testId/4)); // 1..3

    for (int t=1;t<Pn;t++){
        int s = rnd.next(3,5);
        shapes.push_back(randPoly(s));
        int st = heavyStock ? rnd.next(6, W*H) : rnd.next(3, max(3,(W*H)/2));
        stock.push_back(st);
    }

    // ---- print ----
    printf("%d %d\n", W, H);
    for (int y=0;y<H;y++){
        for (int x=0;x<W;x++) printf("%d%c", val[y][x], x+1<W?' ':'\n');
    }
    printf("%d\n", (int)vents.size());
    for (auto &v : vents) printf("%d %d\n", v.first, v.second);
    printf("%d\n", Pn);
    for (int t=0;t<Pn;t++){
        printf("%d %d\n", stock[t], (int)shapes[t].size());
        for (auto &c : shapes[t]) printf("%d %d\n", c.first, c.second);
    }
    return 0;
}
