#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// -----------------------------------------------------------------------------
// Checker / scorer for "Coral Skin Against the Current".
//
// Input:  W H S M Fx Fy
// Output: K, then K lines "x y" -- occupied cells (1-indexed, x in [1,W], y in [1,H]).
//
// Feasibility: 0<=K<=M, in-bounds, no duplicates, and every occupied cell must be
// SUPPORTED: y=1 cells rest on solid bedrock; a y>1 cell (x,y) is supported iff some
// occupied, supported cell (x',y-1) exists with |x-x'|<=S. Any unsupported cell -> 0.
//
// Objective (MAX): F = sum over occupied cells, over the 4 unit faces (L,R,U,D),
// of BASE + max(0, Fx*dx+Fy*dy) for every face whose neighbor is empty (out of grid
// counts empty; the D face at y=1 is covered by bedrock, never exposed).
//
// Baseline B (checker-computed): a single 1-wide pillar of height h0=min(M,H) at the
// middle column x0=(W+1)/2. Its exposed faces are exactly: L and R on all h0 cells,
// U on the top cell only (D is covered by bedrock at y=1, and by the cell below for
// y>1). B = h0*(BASE+bonus(L)) + h0*(BASE+bonus(R)) + (BASE+bonus(U)).
// Score (max): sc = min(1000, 100*F/max(1,B)); ratio = sc/1000.
// -----------------------------------------------------------------------------

static const ll BASE = 5;

int main(int argc, char* argv[]) {
    registerTestlibCmd(argc, argv);

    ll W = inf.readLong((ll)8, (ll)40, "W");
    ll H = inf.readLong((ll)4, (ll)40, "H");
    ll S = inf.readLong((ll)1, (ll)4, "S");
    ll M = inf.readLong(W, W * H, "M");
    ll Fx = inf.readLong((ll)-6, (ll)6, "Fx");
    ll Fy = inf.readLong((ll)-6, (ll)6, "Fy");

    auto bonus = [&](ll dx, ll dy) -> ll {
        ll v = Fx * dx + Fy * dy;
        return v > 0 ? v : 0;
    };
    // ---- internal baseline B: single pillar of height h0 at middle column ----
    ll h0 = min(M, H);
    if (h0 < 1) h0 = 1;
    ll B = h0 * (BASE + bonus(-1, 0)) + h0 * (BASE + bonus(1, 0)) + (BASE + bonus(0, 1));
    if (B <= 0) B = 1;

    // ---- read participant output ----
    ll K = ouf.readLong(0, M, "K");
    vector<vector<char>> occ(W + 2, vector<char>(H + 2, 0));
    for (ll i = 0; i < K; i++) {
        ll x = ouf.readLong(1, W, "x");
        ll y = ouf.readLong(1, H, "y");
        if (occ[x][y]) quitf(_wa, "cell (%lld,%lld) output more than once", x, y);
        occ[x][y] = 1;
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output tokens after the cell list");

    // ---- support check, row by row (row 1 rests on bedrock) ----
    vector<vector<char>> supported(W + 2, vector<char>(H + 2, 0));
    for (ll x = 1; x <= W; x++) {
        if (occ[x][1]) supported[x][1] = 1;
    }
    for (ll y = 2; y <= H; y++) {
        for (ll x = 1; x <= W; x++) {
            if (!occ[x][y]) continue;
            bool sup = false;
            ll lo = max((ll)1, x - S), hi = min(W, x + S);
            for (ll xp = lo; xp <= hi && !sup; xp++) {
                if (supported[xp][y - 1]) sup = true;
            }
            supported[x][y] = sup ? 1 : 0;
        }
    }
    for (ll y = 1; y <= H; y++) {
        for (ll x = 1; x <= W; x++) {
            if (occ[x][y] && !supported[x][y]) {
                quitf(_wa, "cell (%lld,%lld) lacks a support path to bedrock (max overhang S=%lld)", x, y, S);
            }
        }
    }

    // ---- compute exposed-skin objective F ----
    ll F = 0;
    // directions: L=(-1,0) R=(1,0) U=(0,1) D=(0,-1)
    ll dxs[4] = {-1, 1, 0, 0};
    ll dys[4] = {0, 0, 1, -1};
    for (ll x = 1; x <= W; x++) {
        for (ll y = 1; y <= H; y++) {
            if (!occ[x][y]) continue;
            for (int d = 0; d < 4; d++) {
                ll nx = x + dxs[d], ny = y + dys[d];
                bool neighborSolid;
                if (ny == 0) {
                    neighborSolid = true; // bedrock covers the down face at y=1
                } else if (nx < 1 || nx > W || ny > H) {
                    neighborSolid = false; // outside the grid = open water
                } else {
                    neighborSolid = occ[nx][ny];
                }
                if (!neighborSolid) {
                    F += BASE + bonus(dxs[d], dys[d]);
                }
            }
        }
    }

    double sc = min(1000.0, 100.0 * (double)F / (double)max((ll)1, B));
    quitp(sc / 1000.0, "OK F=%lld B=%lld K=%lld Ratio: %.6f", F, B, K, sc / 1000.0);
    return 0;
}
