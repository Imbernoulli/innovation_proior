#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// -----------------------------------------------------------------------------
// Checker / scorer for "Firebreak on the Downwind Line" (family:
// firebreak-wind-anisotropy).  objective: MAXIMIZE unburned fuel value.
//
// Input:  H W T S D BUDGET WX WY IY IX ; then H rows of W terrain chars
//         ('R'=rock, 'G'=grass, 'B'=brush) ; then H rows of W integer fuel
//         values (0 on every 'R' cell).
// Output: M, then M lines "y x" -- cells the rangers clear (cannot burn).
// Feasible iff: all (y,x) in range, pairwise distinct, terrain != 'R', not
// the ignition cell, Manhattan distance to ignition >= D, and M <= BUDGET.
//
// Physics: fire spreads on the 4-connected grid from (IY,IX). Entering a
// non-rock, non-cleared cell v from a neighbor u costs BASE[terrain(v)] *
// factor(dot), where dot = (vx-ux)*WX + (vy-uy)*WY in {-1,0,1} is the
// step's alignment with the wind vector (WX,WY): downwind steps (dot=1)
// cost factor 1, crosswind steps (dot=0) cost factor CF = max(2,S/2),
// upwind steps (dot=-1) cost factor S -- an anisotropic arrival-time field
// (shortest-path / eikonal spread), computed by Dijkstra. Entering a rock
// or cleared cell costs infinity (impassable). A cell "burns" iff its
// shortest arrival time arr <= T. Objective F = sum of value[y][x] over
// cells that are non-rock, NOT cleared, and NOT burned (arr > T). Cleared
// cells and rock cells and burned cells all contribute 0.
//
// Internal baseline B = F with the empty clearing (natural burn, T ticks,
// no rangers action) -- by construction the `trivial` solution (M=0)
// reproduces F=B exactly, giving ratio 0.1. Score: sc=min(1000,100*F/B);
// ratio = sc/1000.
// -----------------------------------------------------------------------------

static const ll INF = (ll)1e15;
static const int BASE_GRASS = 2, BASE_BRUSH = 5;

static int H, W, T, S, D, BUDGET, WX, WY, IY, IX;
static vector<string> terrain;
static vector<vector<ll>> value;

static inline int baseCost(int y, int x) {
    char c = terrain[y][x];
    if (c == 'G') return BASE_GRASS;
    if (c == 'B') return BASE_BRUSH;
    return -1; // rock, never used as a positive base
}

// Dijkstra arrival times given a boolean "blocked" grid (rock OR cleared).
static vector<vector<ll>> arrivalTimes(const vector<vector<char>>& blocked) {
    int CF = max(2, S / 2);
    vector<vector<ll>> arr(H, vector<ll>(W, INF));
    priority_queue<pair<ll,int>, vector<pair<ll,int>>, greater<>> pq;
    arr[IY][IX] = 0;
    pq.push({0, IY * W + IX});
    static const int dy[4] = {-1, 1, 0, 0};
    static const int dx[4] = {0, 0, -1, 1};
    while (!pq.empty()) {
        auto [d, node] = pq.top(); pq.pop();
        int uy = node / W, ux = node % W;
        if (d > arr[uy][ux]) continue;
        for (int k = 0; k < 4; k++) {
            int vy = uy + dy[k], vx = ux + dx[k];
            if (vy < 0 || vy >= H || vx < 0 || vx >= W) continue;
            if (blocked[vy][vx]) continue;
            int dot = (vx - ux) * WX + (vy - uy) * WY; // step alignment with wind
            int factor = (dot > 0) ? 1 : (dot == 0 ? CF : S);
            ll w = (ll)baseCost(vy, vx) * factor;
            ll nd = d + w;
            if (nd < arr[vy][vx]) { arr[vy][vx] = nd; pq.push({nd, vy * W + vx}); }
        }
    }
    return arr;
}

static ll objective(const vector<vector<char>>& cleared) {
    vector<vector<char>> blocked(H, vector<char>(W, 0));
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++)
            blocked[y][x] = (terrain[y][x] == 'R') || cleared[y][x];
    auto arr = arrivalTimes(blocked);
    ll F = 0;
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++) {
            if (terrain[y][x] == 'R') continue;
            if (cleared[y][x]) continue;
            if (arr[y][x] <= T) continue; // burned
            F += value[y][x];
        }
    return F;
}

int main(int argc, char* argv[]) {
    registerTestlibCmd(argc, argv);

    H = inf.readInt(4, 4000, "H");
    W = inf.readInt(4, 4000, "W");
    T = inf.readInt(1, 100000, "T");
    S = inf.readInt(1, 100, "S");
    D = inf.readInt(0, 4000, "D");
    BUDGET = inf.readInt(0, 4000000, "BUDGET");
    WX = inf.readInt(-1, 1, "WX");
    WY = inf.readInt(-1, 1, "WY");
    IY = inf.readInt(0, H - 1, "IY");
    IX = inf.readInt(0, W - 1, "IX");
    if (!((WX == 0) != (WY == 0))) quitf(_fail, "generator bug: wind must be one cardinal unit vector");

    terrain.assign(H, "");
    for (int y = 0; y < H; y++) terrain[y] = inf.readToken();
    for (int y = 0; y < H; y++)
        if ((int)terrain[y].size() != W) quitf(_fail, "generator bug: terrain row width");

    value.assign(H, vector<ll>(W, 0));
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++) value[y][x] = inf.readLong(0, (ll)1e9, "value");

    if (terrain[IY][IX] == 'R') quitf(_fail, "generator bug: ignition on rock");

    // ---- internal baseline B: empty clearing, natural burn ----
    vector<vector<char>> emptyCleared(H, vector<char>(W, 0));
    ll B = objective(emptyCleared);
    if (B <= 0) B = 1;

    // ---- participant output ----
    int M = ouf.readInt(0, H * W, "M");
    if (M > BUDGET) quitf(_wa, "clearing budget exceeded: M=%d > BUDGET=%d", M, BUDGET);
    vector<vector<char>> cleared(H, vector<char>(W, 0));
    for (int i = 0; i < M; i++) {
        int y = ouf.readInt(0, H - 1, "y");
        int x = ouf.readInt(0, W - 1, "x");
        if (cleared[y][x]) quitf(_wa, "duplicate cleared cell (%d,%d)", y, x);
        if (terrain[y][x] == 'R') quitf(_wa, "cell (%d,%d) is already rock -- cannot clear rock", y, x);
        if (y == IY && x == IX) quitf(_wa, "cannot clear the ignition cell (%d,%d)", y, x);
        ll dist = llabs((ll)y - IY) + llabs((ll)x - IX);
        if (dist < D) quitf(_wa, "cell (%d,%d) is inside the standoff distance (dist=%lld < D=%d) -- crews cannot approach the flame front that closely", y, x, dist, D);
        cleared[y][x] = 1;
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing tokens after the clearing list");

    ll F = objective(cleared);

    double sc = min(1000.0, 100.0 * (double)F / (double)max(1LL, B));
    quitp(sc / 1000.0, "OK F=%lld B=%lld M=%d/%d Ratio: %.6f", F, B, M, BUDGET, sc / 1000.0);
    return 0;
}
