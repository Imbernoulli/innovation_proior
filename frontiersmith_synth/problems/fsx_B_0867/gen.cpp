#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// -----------------------------------------------------------------------------
// "Firebreak on the Downwind Line"   (generator)
// family: firebreak-wind-anisotropy
//
// A valley is an H x W grid. Fire starts at one ignition cell and spreads on
// the 4-connected grid; a step INTO a cell costs BASE[terrain] * factor(dot),
// where dot is the step's alignment with a fixed wind vector: downwind steps
// are cheap (factor 1), crosswind steps cost CF=max(2,S/2), upwind steps cost
// S (S in [4,10]) -- an anisotropic eikonal spread whose T-ticks reachable
// set is a cone: WIDE near the source, narrowing toward its downwind tip.
// Rangers may clear up to BUDGET cells (each removes that cell's own fuel
// value AND blocks spread) but MAY NOT approach within Manhattan distance D
// of the ignition point (the flame front's standoff distance) -- a tight
// "seal the source" ring is therefore either impossible (too few affordable
// cells at radius D) or, if attempted, wastes cells on crosswind/upwind arcs
// the fire barely reaches within the T-tick horizon anyway.
//
// PLANTED TRAP (>=4 of the 10 tests, "trapMode"): wind is fixed WEST, the
// ignition sits east of center, and a corridor of GRASS (fast, cheap) is
// carved due west of it, flanked above/below by ROCK walls (free natural
// anchors) for a good stretch -- opening onto a high-value BRUSH cluster
// further west. The corridor's cross-section (band height only, since the
// flanks are already rock) is cheap to seal ANYWHERE past the standoff
// distance, and sealing it anywhere blocks the whole cluster beyond. An
// "encircle the ignition" instinct instead spends the SAME budget building a
// standoff-radius ring in a fixed, wind-oblivious raster order (by column,
// east to west -- see solutions/greedy.cpp) -- which reaches the (fixed-
// west) downwind arc dead last, so it is left open every time.
// -----------------------------------------------------------------------------

static const int BASE_GRASS = 2, BASE_BRUSH = 5;

struct Params {
    int H, W, S, D, BUDGET, T, WX, WY, IY, IX;
    bool trap;
};

int H, W;
vector<string> terrain;
vector<vector<ll>> value;

static void setCell(int y, int x, char t, ll v) {
    terrain[y][x] = t;
    value[y][x] = v;
}

int main(int argc, char* argv[]) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);

    Params p{};
    int halfBand = 2; // corridor band half-height (trap tests), fixed and small
    switch (testId) {
        // trap tests: H, W are DERIVED below from D (must leave room, beyond
        // the mandatory standoff, for a legal stretch of cheap corridor).
        case 1:  p = {0, 0, 5, 2, 5,  0, 0,0,0,0, true};  break;
        case 2:  p = {12, 16, 5, 2, 10, 0, 0,0,0,0, false}; break;
        case 3:  p = {0, 0, 6, 3, 7,  0, 0,0,0,0, true};  break;
        case 4:  p = {18, 24, 6, 3, 14, 0, 0,0,0,0, false}; break;
        case 5:  p = {0, 0, 7, 4, 9,  0, 0,0,0,0, true};  break;
        case 6:  p = {24, 30, 6, 4, 18, 0, 0,0,0,0, false}; break;
        case 7:  p = {0, 0, 8, 5, 11, 0, 0,0,0,0, true};  break;
        case 8:  p = {32, 40, 7, 5, 22, 0, 0,0,0,0, false}; break;
        case 9:  p = {0, 0, 9, 6, 13, 0, 0,0,0,0, true};  break;
        default: p = {0, 0, 9, 7, 15, 0, 0,0,0,0, true};  break; // testId 10
    }
    int margin = p.D + 2;
    if (p.trap) {
        // corridorLen must clear the standoff with room to spare: D+halfBand+5.
        int corridorLen = p.D + halfBand + 5;
        int clusterWidthMin = 16;
        p.W = 2 * margin + clusterWidthMin + corridorLen + 3;
        // H is kept just large enough for the standoff ring's margin (the
        // corridor/cluster band itself is only 2*halfBand+1 rows tall) --
        // deliberately NOT padded with a lot of extra low-value background
        // rows, so the map's total value stays dominated by the band the
        // insight actually controls, not by incidental scattered terrain.
        p.H = 2 * margin + 2 * halfBand + 5;
    }
    H = p.H; W = p.W;

    // wind direction
    if (p.trap) { p.WX = -1; p.WY = 0; }
    else {
        int dir = rnd.next(0, 3);
        static const int dxs[4] = {1, -1, 0, 0};
        static const int dys[4] = {0, 0, 1, -1};
        p.WX = dxs[dir]; p.WY = dys[dir];
    }

    // ignition point: always placed so the WIND's own downwind axis has a
    // long runway ahead of it (near the upwind edge along that axis) -- the
    // crosswind coordinate is centered. This keeps the anisotropic-cone
    // geometry meaningful regardless of which of the 4 directions the wind
    // happens to blow (not just the hand-built trap corridor's West).
    int downwindRunway;
    if (p.WX == 1)       { p.IX = margin;              downwindRunway = (W - 1 - margin) - margin; p.IY = H / 2; }
    else if (p.WX == -1) { p.IX = W - 1 - margin;       downwindRunway = (W - 1 - margin) - margin; p.IY = H / 2; }
    else if (p.WY == 1)  { p.IY = margin;               downwindRunway = (H - 1 - margin) - margin; p.IX = W / 2; }
    else                 { p.IY = H - 1 - margin;       downwindRunway = (H - 1 - margin) - margin; p.IX = W / 2; }
    p.IY = max(margin, min(H - 1 - margin, p.IY));
    p.IX = max(margin, min(W - 1 - margin, p.IX));

    // ---- base terrain: random texture, deliberately LOW-value background so
    // the objective's headroom is dominated by the planted high-value cluster
    // (trap tests) rather than by incidental scattered brush.
    terrain.assign(H, string(W, 'G'));
    value.assign(H, vector<ll>(W, 0));
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            double r = rnd.next(0.0, 1.0);
            if (r < 0.14) setCell(y, x, 'R', 0);
            else if (r < 0.24) setCell(y, x, 'B', rnd.next(2, 3));
            else setCell(y, x, 'G', 1);
        }
    }

    int corridorLen = -1, xClusterEnd = -1, xClusterStart = -1;

    if (p.trap) {
        corridorLen = p.D + halfBand + 5; // matches the sizing formula above
        xClusterEnd = p.IX - corridorLen - 1;
        xClusterStart = margin;
        if (xClusterEnd < xClusterStart) xClusterEnd = xClusterStart;

        // The whole west strip [xClusterStart, IX-1] is a single band (same
        // halfBand height as the standoff corridor, flanked by rock on both
        // sides) so the fire's path west is a straight 1-D corridor: cheap
        // GRASS near the ignition (the legal, cheap cut point), high-value
        // BRUSH cluster further out. Keeping the band height constant (no
        // vertical fan-out) means the required cut width is EXACTLY the
        // band height anywhere along the strip -- the only thing that
        // changes downwind is what's being protected, not what it costs.
        for (int x = xClusterStart; x <= p.IX - 1; x++) {
            bool inCluster = (x <= xClusterEnd);
            for (int y = 0; y < H; y++) {
                if (y >= p.IY - halfBand && y <= p.IY + halfBand) {
                    if (inCluster) setCell(y, x, 'B', rnd.next(9, 10));
                    else setCell(y, x, 'G', rnd.next(1, 2));
                } else setCell(y, x, 'R', 0);
            }
        }
    }

    setCell(p.IY, p.IX, 'G', 0); // ignition must be flammable; its own value never counts

    // ---- T: fire horizon, generous enough to naturally burn THROUGH the
    // corridor and the full cluster depth (so the baseline genuinely loses
    // that value), and to eat well into the generic background in every
    // test (so a naive, mis-placed clearing has little safe value left to
    // accidentally "protect").
    if (p.trap) {
        int clusterDepth = xClusterEnd - xClusterStart + 1;
        // Deliberately burns most (not all) of the cluster naturally, so the
        // baseline is a real, sizable retention -- not a near-total wipe --
        // yet `strong` (which saves the WHOLE cluster with one cheap cut
        // right past the standoff) still lands at a healthy multiple of it.
        p.T = 2 * corridorLen + 5 * ((clusterDepth * 99) / 100);
    } else {
        p.T = (int)llround(4.5 * (downwindRunway + 4));
    }

    printf("%d %d %d %d %d %d %d %d %d %d\n",
           p.H, p.W, p.T, p.S, p.D, p.BUDGET, p.WX, p.WY, p.IY, p.IX);
    for (int y = 0; y < H; y++) printf("%s\n", terrain[y].c_str());
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) printf("%lld%c", value[y][x], x + 1 < W ? ' ' : '\n');
    }
    return 0;
}
