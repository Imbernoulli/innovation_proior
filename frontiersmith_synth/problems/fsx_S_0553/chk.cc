#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// -----------------------------------------------------------------------------
// Checker / scorer for "Laser cutter that warps its own sheet".
// family: pierce-heat-choreography
//
// Input:  N M DA DB PW PG THR ; then M lines  w h v q  (part w x h, value v, heat q).
// Output: K ; then K lines  id x y  -- the placement list IN CUTTING ORDER (the
//         k-th line is cut at contour-tick k, 1-based). Each id in [1,M] used once;
//         part id placed with top-left (x,y), 0<=x, x+w<=N, 0<=y, y+h<=N; parts must
//         not overlap. Placing nothing (K=0) is feasible and scores 0.
//
// Thermal field (deterministic, integer): when part i is cut at tick t_i, and part j
// is cut at tick t_j >= t_i, the heat part i contributes to part j is
//     floor( q_i / (1 + DA*(t_j - t_i) + DB*gap(i,j)) )  +  max(0, PW - PG*gap(i,j)),
// the first term a transient that decays with elapsed ticks AND cell separation, the
// second a PERSISTENT warp (time-independent) a nearby cut leaves in the metal.
// gap(i,j) is the Manhattan gap between the two rectangles (0 if they touch).
// Only parts cut EARLIER (t_i < t_j) plus j's own pierce q_j count toward j. Part j
// SCRAPS (value lost) iff its total field exceeds THR. A scrapped part still emitted
// its heat to others.
// Objective (MAX):  F = sum of v over placed parts that do NOT scrap.
//
// Internal baseline B (checker-computed, reproduced by solutions/trivial.cpp):
//   a widely-spaced grid drop -- cell = 3*maxW by 3*maxH, place the first
//   min(M, cols*rows, 25) parts in INPUT ORDER one per cell, cut in input order. The
//   spacing is large enough that every dropped part survives, so B = sum of those
//   parts' values. This is the "lay a few parts far apart and cut them" reference.
//   B > 0 by construction (fallback: value of part 1, else 1).
// Score (max):  sc = min(1000, 100 * F / max(1,B)); ratio = sc/1000  (trivial -> 0.1).
// -----------------------------------------------------------------------------

static ll gapDist(ll xi, ll yi, ll wi, ll hi, ll xj, ll yj, ll wj, ll hj){
    ll gx = max(0LL, max(xi - (xj + wj), xj - (xi + wi)));
    ll gy = max(0LL, max(yi - (yj + hj), yj - (yi + hi)));
    return gx + gy;
}

int main(int argc, char* argv[]){
    registerTestlibCmd(argc, argv);

    ll N   = inf.readLong();
    int M  = inf.readInt();
    ll DA  = inf.readLong();
    ll DB  = inf.readLong();
    ll PW  = inf.readLong();
    ll PG  = inf.readLong();
    ll THR = inf.readLong();
    vector<ll> w(M + 1), h(M + 1), v(M + 1), q(M + 1);
    ll maxW = 1, maxH = 1;
    for (int i = 1; i <= M; i++){
        w[i] = inf.readLong(); h[i] = inf.readLong();
        v[i] = inf.readLong(); q[i] = inf.readLong();
        maxW = max(maxW, w[i]); maxH = max(maxH, h[i]);
    }

    // ---- heat simulation over an ordered list of placed parts ----
    // parts given as (x,y,id) in cut order -> returns surviving value.
    auto simulate = [&](const vector<array<ll,3>>& pl) -> ll {
        int K = (int)pl.size();
        ll F = 0;
        for (int j = 0; j < K; j++){
            ll idj = pl[j][2];
            ll heat = q[idj];
            for (int i = 0; i < j && heat <= THR; i++){
                ll idi = pl[i][2];
                ll dt  = (ll)(j - i);
                ll gp  = gapDist(pl[i][0], pl[i][1], w[idi], h[idi],
                                 pl[j][0], pl[j][1], w[idj], h[idj]);
                ll denom = 1 + DA * dt + DB * gp;
                heat += q[idi] / denom;                 // transient (time+space decay)
                ll warp = PW - PG * gp;                  // persistent, time-INDEPENDENT
                if (warp > 0) heat += warp;
            }
            if (heat <= THR) F += v[idj];
        }
        return F;
    };

    // ---- internal baseline B: widely-spaced grid, input order ----
    ll cellW = 3 * maxW, cellH = 3 * maxH;
    ll cols = (cellW > 0) ? N / cellW : 0;
    ll rows = (cellH > 0) ? N / cellH : 0;
    ll cap  = cols * rows;
    ll B = 0;
    {
        vector<array<ll,3>> bpl;
        ll place = min<ll>(min<ll>(M, cap), 25);      // K0 = 25 reference parts
        for (ll k = 0; k < place; k++){
            ll col = k % cols, row = k / cols;
            ll x = col * cellW, y = row * cellH;
            bpl.push_back({x, y, k + 1});
        }
        B = simulate(bpl);
    }
    if (B <= 0) B = (M >= 1 ? v[1] : 1);
    if (B <= 0) B = 1;

    // ---- parse & validate participant output ----
    int K = ouf.readInt(0, M, "K");
    vector<char> used(M + 1, 0);
    vector<array<ll,3>> pl;                       // (x,y,id) in cut order
    // occupancy grid for overlap detection (N up to 500 -> <=250k cells)
    vector<unsigned char> grid((size_t)N * (size_t)N, 0);
    for (int k = 0; k < K; k++){
        int id = ouf.readInt(1, M, "id");
        if (used[id]) quitf(_wa, "part %d listed more than once", id);
        used[id] = 1;
        ll x = ouf.readLong(0, N - w[id], "x");
        ll y = ouf.readLong(0, N - h[id], "y");
        // overlap check via stamping
        for (ll yy = y; yy < y + h[id]; yy++){
            size_t base = (size_t)yy * (size_t)N;
            for (ll xx = x; xx < x + w[id]; xx++){
                if (grid[base + (size_t)xx])
                    quitf(_wa, "part %d overlaps a previously placed part at (%lld,%lld)", id, xx, yy);
                grid[base + (size_t)xx] = 1;
            }
        }
        pl.push_back({x, y, (ll)id});
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output tokens after the placement list");

    ll F = simulate(pl);

    double sc = min(1000.0, 100.0 * (double)F / (double)max((ll)1, B));
    quitp(sc / 1000.0, "OK F=%lld B=%lld K=%d Ratio: %.6f", F, B, K, sc / 1000.0);
    return 0;
}
