#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int H, W, D;
vector<vector<pair<int,int>>> base; // base cells per shape (normalized)
vector<int> supply;
vector<vector<char>> blocked;

static const int LATTICE = 6; // reference baseline lattice spacing

// transform shape i by orientation o, normalize to bounding box origin.
vector<pair<int,int>> transformShape(int i, int o) {
    vector<pair<int,int>> v;
    v.reserve(base[i].size());
    for (auto& p : base[i]) {
        int r = p.first, c = p.second;
        if (o >= 4) c = -c;              // reflect
        for (int t = 0; t < (o % 4); t++) { int nr = c, nc = -r; r = nr; c = nc; } // rotate
        v.push_back({r, c});
    }
    int mr = INT_MAX, mc = INT_MAX;
    for (auto& p : v) { mr = min(mr, p.first); mc = min(mc, p.second); }
    for (auto& p : v) { p.first -= mr; p.second -= mc; }
    return v;
}

int main(int argc, char** argv) {
    registerTestlibCmd(argc, argv);

    H = inf.readInt();
    W = inf.readInt();
    D = inf.readInt();

    base.resize(D);
    supply.assign(D, 0);
    for (int i = 0; i < D; i++) {
        int A = inf.readInt();
        supply[i] = inf.readInt();
        for (int j = 0; j < A; j++) {
            int dr = inf.readInt();
            int dc = inf.readInt();
            base[i].push_back({dr, dc});
        }
    }
    int Q = inf.readInt();
    blocked.assign(H, vector<char>(W, 0));
    for (int j = 0; j < Q; j++) {
        int r = inf.readInt();
        int c = inf.readInt();
        blocked[r][c] = 1;
    }

    // ---- internal reference baseline B: sparse lattice of shape-0 (2x2) squares ----
    // must match trivial.cpp exactly.
    ll B = 0;
    {
        auto s0 = transformShape(0, 0); // the 2x2 square, orientation 0
        int usedS0 = 0;
        for (int r = 0; r + 1 < H; r += LATTICE) {
            for (int c = 0; c + 1 < W; c += LATTICE) {
                if (usedS0 >= supply[0]) break;
                bool ok = true;
                for (auto& p : s0) {
                    int rr = r + p.first, cc = c + p.second;
                    if (rr < 0 || rr >= H || cc < 0 || cc >= W || blocked[rr][cc]) { ok = false; break; }
                }
                if (ok) { B += (ll)s0.size(); usedS0++; }
            }
        }
    }
    if (B <= 0) quitf(_fail, "bad instance: baseline B=%lld", B);

    // ---- read & validate participant packing ----
    vector<vector<char>> occ(H, vector<char>(W, 0));
    vector<int> uses(D, 0);
    ll F = 0;

    int P = ouf.readInt(0, H * W, "P");
    for (int k = 0; k < P; k++) {
        int i = ouf.readInt(0, D - 1, "shape");
        int o = ouf.readInt(0, 7, "orient");
        int r = ouf.readInt(0, H - 1, "r");
        int c = ouf.readInt(0, W - 1, "c");
        uses[i]++;
        if (uses[i] > supply[i])
            quitf(_wa, "shape %d used %d times, exceeds supply %d", i, uses[i], supply[i]);
        auto cells = transformShape(i, o);
        for (auto& p : cells) {
            int rr = r + p.first, cc = c + p.second;
            if (rr < 0 || rr >= H || cc < 0 || cc >= W)
                quitf(_wa, "placement %d (shape %d orient %d anchor %d,%d) leaves the roof at (%d,%d)",
                      k, i, o, r, c, rr, cc);
            if (blocked[rr][cc])
                quitf(_wa, "placement %d covers obstructed tile (%d,%d)", k, rr, cc);
            if (occ[rr][cc])
                quitf(_wa, "placement %d overlaps a previous bed at tile (%d,%d)", k, rr, cc);
            occ[rr][cc] = 1;
            F++;
        }
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output tokens");

    double sc = min(1000.0, 100.0 * (double)F / (double)max((ll)1, B));
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}
