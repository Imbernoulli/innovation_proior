#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

// generate a random connected polyomino of s cells, normalized, bounding box <= (maxR,maxC)
vector<pair<int,int>> genPoly(int s, int maxR, int maxC) {
    for (int attempt = 0; attempt < 2000; attempt++) {
        set<pair<int,int>> cells;
        vector<pair<int,int>> cur;
        cells.insert({0,0}); cur.push_back({0,0});
        int dr[] = {0,0,1,-1}, dc[] = {1,-1,0,0};
        int guard = 0;
        while ((int)cells.size() < s && guard < 100000) {
            guard++;
            auto cell = cur[rnd.next((int)cur.size())];
            int d = rnd.next(4);
            pair<int,int> nb = {cell.first + dr[d], cell.second + dc[d]};
            if (!cells.count(nb)) { cells.insert(nb); cur.push_back(nb); }
        }
        int mr = INT_MAX, mc = INT_MAX, Mr = INT_MIN, Mc = INT_MIN;
        for (auto &p : cells) { mr = min(mr,p.first); mc = min(mc,p.second); Mr = max(Mr,p.first); Mc = max(Mc,p.second); }
        if ((Mr - mr) >= maxR || (Mc - mc) >= maxC) continue;
        vector<pair<int,int>> res;
        for (auto &p : cells) res.push_back({p.first - mr, p.second - mc});
        return res;
    }
    // fallback: a straight bar clipped to bounds
    vector<pair<int,int>> res;
    for (int i = 0; i < s; i++) res.push_back({0, min(i, maxC - 1)});
    return res;
}

int main(int argc, char** argv) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);

    // ---- board size ladder (small scale) ----
    int base = 5 + testId;                 // 6 .. 15
    int H = min(15, base);
    int W = min(15, base + (testId % 2));   // slightly non-square on odd tests

    // big rectangular "golden super" comb section -> defines the baseline B
    int rh = max(3, (H * 2) / 5);
    int rw = max(3, (W * 2) / 5);
    rh = min(rh, H); rw = min(rw, W);

    // ---- frame: keep top-left rh x rw corner clear so the big comb always fits ----
    vector<string> grid(H, string(W, '.'));
    double blockFrac = 0.08 + 0.02 * (testId % 4);   // 0.08 .. 0.14
    vector<pair<int,int>> candidates;
    for (int i = 0; i < H; i++)
        for (int j = 0; j < W; j++)
            if (!(i < rh && j < rw)) candidates.push_back({i, j});
    shuffle(candidates.begin(), candidates.end());
    int nblock = (int)round(blockFrac * (double)candidates.size());
    for (int b = 0; b < nblock && b < (int)candidates.size(); b++)
        grid[candidates[b].first][candidates[b].second] = '#';

    // ---- comb-section types ----
    // type 1 = big rectangle (largest area -> B)
    struct Piece { vector<pair<int,int>> off; int cnt; };
    vector<Piece> pieces;
    {
        Piece bigp; bigp.cnt = 1 + testId / 5;   // 1 .. 3
        for (int i = 0; i < rh; i++)
            for (int j = 0; j < rw; j++)
                bigp.off.push_back({i, j});
        pieces.push_back(bigp);
    }
    int nSmall = 3 + testId / 3;                 // 4 .. 6 small types
    nSmall = min(nSmall, 5);
    int maxBB = min(H, W);                        // small pieces fit comfortably
    for (int k = 0; k < nSmall; k++) {
        int s = rnd.next(3, 5);
        Piece p;
        p.off = genPoly(s, maxBB, maxBB);
        p.cnt = rnd.next(2, 5);
        pieces.push_back(p);
    }
    int P = (int)pieces.size();

    // ---- emit ----
    printf("%d %d %d\n", H, W, P);
    for (int i = 0; i < H; i++) printf("%s\n", grid[i].c_str());
    for (int t = 0; t < P; t++) {
        printf("%d %d\n", (int)pieces[t].off.size(), pieces[t].cnt);
        for (auto &o : pieces[t].off) printf("%d %d\n", o.first, o.second);
    }
    return 0;
}
