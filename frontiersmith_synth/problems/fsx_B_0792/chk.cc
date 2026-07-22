#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// -----------------------------------------------------------------------------
// Checker / scorer for "Village Shield: Deflectors Over the Avalanche Line".
// family: granular-deflector-village-shield    objective: MINIMIZE
//
// Input:  H W K L SCALE DISS ; then W column weights w_0..w_{W-1} ; then K
//         lines "c m p" (distinct release columns, mass, initial momentum).
// Output: M (number of barriers) then M lines "r c o h"
//         (0<=r<H, 0<=c<W, o in {0=WALL,1=LEFT-deflect,2=RIGHT-deflect},
//         1<=h<=HMAX). Feasible iff cells are distinct and sum(h) <= L.
//
// Physics: a packet starts at (row=0, col=c_i, dCol=0, momentum=p_i). At each
// row, if a barrier occupies its current cell:
//   WALL:    must absorb the FULL momentum p (cap=h*SCALE >= p) to arrest the
//            packet (momentum -> 0, permanently stopped). Otherwise: FAILS,
//            packet passes through completely unchanged ("overtopped").
//   DEFLECT: must absorb only normal_load = ceil(p*frac(turn)/100), where
//            turn = |target_dCol - current_dCol| in {0,1,2} and frac is a
//            fixed table (15/40/85). On success the packet turns to the
//            barrier's direction AND its momentum shrinks further by a fixed
//            DISSIPATION factor: p <- floor((p-normal_load)*DISS/100).
//            Otherwise: FAILS, packet passes through completely unchanged.
// The packet advances one row per step (dRow=+1, dCol as current), and stops
// mattering as soon as it drifts off the lateral grid (col<0 or col>=W,
// harmless) or its momentum reaches 0 (buried in the runout, harmless). If it
// survives to exit below row H-1 at an in-range column c, it deposits impact
// w[c]*mass*momentum onto the objective.
//
// Objective F = sum of impacts over all K packets (minimize). Internal
// baseline B = F with an empty barrier set (every packet reaches the bottom
// at its own release column, undissipated). Score: sc=min(1000,100*B/max(1,F));
// ratio = sc/1000.  Empty output -> F=B -> ratio=0.1 exactly.
// -----------------------------------------------------------------------------

static const int HMAX = 100;
static const int FRAC0 = 15, FRAC1 = 40, FRAC2 = 85;

static int H, W, K;
static ll L;
static int SCALEg, DISSg;
static vector<int> wt;
static vector<ll> relC, relM, relP;

struct Barrier { bool exists = false; int o = 0; int h = 0; };

static ll simulate(vector<vector<Barrier>>& grid) {
    ll F = 0;
    for (int i = 0; i < K; i++) {
        ll row = 0, col = relC[i], dCol = 0, p = relP[i];
        ll mass = relM[i];
        while (true) {
            if (row >= H) {
                if (col >= 0 && col < W) F += (ll)wt[col] * mass * p;
                break;
            }
            if (col < 0 || col >= W) break;
            if (p <= 0) break;
            Barrier& b = grid[(size_t)row][(size_t)col];
            if (b.exists) {
                ll cap = (ll)b.h * SCALEg;
                if (b.o == 0) {
                    ll normal = p;
                    if (cap >= normal) p = 0; // arrested
                    // else: overtopped, unchanged
                } else {
                    ll target = (b.o == 1) ? -1 : 1;
                    ll turn = llabs(target - dCol);
                    int frac = (turn == 0) ? FRAC0 : (turn == 1) ? FRAC1 : FRAC2;
                    ll normal = (p * (ll)frac + 99) / 100;
                    if (cap >= normal) {
                        ll remainder = p - normal;
                        p = (remainder * (ll)DISSg) / 100;
                        dCol = target;
                    }
                    // else: unchanged
                }
            }
            row += 1; col += dCol;
        }
    }
    return F;
}

int main(int argc, char* argv[]) {
    registerTestlibCmd(argc, argv);

    H = inf.readInt(1, 60, "H");
    W = inf.readInt(2, 80, "W");
    K = inf.readInt(1, 40, "K");
    L = inf.readLong(1LL, 10000LL, "L");
    SCALEg = inf.readInt(1, 8, "SCALE");
    DISSg = inf.readInt(50, 95, "DISS");

    wt.resize(W);
    for (int c = 0; c < W; c++) wt[c] = inf.readInt(0, 10, "w");

    relC.resize(K); relM.resize(K); relP.resize(K);
    vector<char> colUsed(W, 0);
    for (int i = 0; i < K; i++) {
        int c = inf.readInt(0, W - 1, "c");
        if (colUsed[c]) quitf(_fail, "generator bug: duplicate release column %d", c);
        colUsed[c] = 1;
        relC[i] = c;
        relM[i] = inf.readLong(1LL, 10000LL, "m");
        relP[i] = inf.readLong(1LL, 3000LL, "p");
    }

    // ---- internal baseline B: empty barrier set ----
    vector<vector<Barrier>> emptyGrid(H, vector<Barrier>(W));
    ll Braw = simulate(emptyGrid);
    if (Braw <= 0) Braw = 1;
    // Ambient background hazard: even a season with flawless barrier coverage
    // still carries a small irreducible risk (residual wind-drift, settling
    // snow off the mapped release zones) that no placement can address. This
    // is a FIXED function of the input alone (never of the participant's
    // output), added identically to both B and F, so it does not touch the
    // trivial (M=0) calibration (F=B there regardless) but keeps a ceiling
    // strictly above even a flawless defense of every modeled release.
    ll ambient = (Braw * 15 + 99) / 100;
    ll B = Braw + ambient;

    // ---- participant output ----
    vector<vector<Barrier>> grid(H, vector<Barrier>(W));
    int M = ouf.readInt(0, H * W, "M");
    ll spent = 0;
    for (int i = 0; i < M; i++) {
        int r = ouf.readInt(0, H - 1, "r");
        int c = ouf.readInt(0, W - 1, "c");
        int o = ouf.readInt(0, 2, "o");
        int h = ouf.readInt(1, HMAX, "h");
        if (grid[(size_t)r][(size_t)c].exists) quitf(_wa, "duplicate barrier at cell (%d,%d)", r, c);
        grid[(size_t)r][(size_t)c] = Barrier{true, o, h};
        spent += h;
        if (spent > L) quitf(_wa, "barrier height budget exceeded: spent=%lld > L=%lld", spent, L);
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing tokens after the barrier list");
    if (spent > L) quitf(_wa, "barrier height budget exceeded: spent=%lld > L=%lld", spent, L);

    // ---- participant objective F ----
    ll F = simulate(grid) + ambient;

    double sc = min(1000.0, 100.0 * (double)B / (double)max(1LL, F));
    quitp(sc / 1000.0, "OK F=%lld B=%lld spent=%lld/%lld Ratio: %.6f", F, B, spent, L, sc / 1000.0);
    return 0;
}
