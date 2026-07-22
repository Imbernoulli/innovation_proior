#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// -----------------------------------------------------------------------------
// "Village Shield: Deflectors Over the Avalanche Line"   (generator)
// family: granular-deflector-village-shield
//
// A slope is a grid of H rows (downslope steps, row 0 = release line, row H-1
// = last row before the village) x W columns (lateral position). Every column
// c carries a settlement weight w[c] (village core columns are heavily
// weighted, open slope columns lightly weighted -- but NEVER zero, so nowhere
// is perfectly "safe", only cheaper to be hit at).
//
// K avalanche release events start at row 0, each at a DISTINCT column c_i
// with mass m_i and initial downslope momentum p_i (moving straight down,
// lateral drift dCol = 0). The solver places barrier segments (row, col,
// orientation, height) under a total-height budget L. A WALL barrier (o=0)
// must absorb the packet's FULL momentum (the flow's strong axis) to arrest
// it; a DEFLECT barrier (o=1 left / o=2 right) only needs to absorb a MUCH
// smaller fraction (the flow's weak axis) proportional to how sharp a turn it
// asks for, and on success it also DISSIPATES a further fixed fraction of the
// remaining momentum (energy-dissipation cascade) while routing the packet
// onward in the new lateral direction.
//
// PLANTED TRAP: every barrier has a hard per-cell height cap HMAX (a real dam
// can only be built so tall). Release momenta on the "trap" events are
// calibrated (p = SCALE*HMAX*M, M in [1.3,2.3]) so that NO wall cell, at ANY
// height up to HMAX and regardless of total budget L, can ever reach the
// required capacity (h_needed = HMAX*M > HMAX) -- an "arresting dam directly
// uphill of the village" is *structurally* overtopped, full stop. A cheap
// oblique deflector (turn=1, needs only ~40% of the momentum) comfortably
// fits under HMAX for the same M range, and after one successful deflection
// the packet's momentum shrinks (dissipation) so it drifts to open, low-
// weight ground. A minority of "stoppable" events (M in [0.3,0.85]) are
// planted so that a naive full-width wall recipe DOES pick up some real
// credit -- keeping the greedy tier a plausible, not a strawman, first draft.
// -----------------------------------------------------------------------------

static const int HMAX = 100;

int main(int argc, char* argv[]) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);

    int H, W, K, SCALE, DISS;
    double trapFrac, noiseFrac;

    switch (testId) {
        case 1:  H = 4;  W = 8;   K = 2;  trapFrac = 0.50; noiseFrac = 0.00; break;
        case 2:  H = 6;  W = 14;  K = 5;  trapFrac = 0.50; noiseFrac = 0.20; break;
        case 3:  H = 8;  W = 16;  K = 8;  trapFrac = 0.70; noiseFrac = 0.10; break;
        case 4:  H = 10; W = 18;  K = 8;  trapFrac = 0.20; noiseFrac = 0.30; break;
        case 5:  H = 16; W = 26;  K = 14; trapFrac = 0.60; noiseFrac = 0.20; break;
        case 6:  H = 20; W = 32;  K = 18; trapFrac = 0.75; noiseFrac = 0.15; break;
        case 7:  H = 28; W = 44;  K = 24; trapFrac = 0.60; noiseFrac = 0.20; break;
        case 8:  H = 34; W = 56;  K = 30; trapFrac = 0.65; noiseFrac = 0.45; break;
        case 9:  H = 44; W = 68;  K = 34; trapFrac = 0.80; noiseFrac = 0.15; break;
        default: H = 60; W = 80;  K = 40; trapFrac = 0.70; noiseFrac = 0.20; break; // testId 10
    }

    SCALE = rnd.next(2, 8);
    DISS = rnd.next(55, 90);
    ll capacityFull = (ll)SCALE * HMAX;

    // ---- village core: a contiguous band of high-weight columns, centered ----
    int villageWidth = max(2, W * 35 / 100);
    if (villageWidth > W) villageWidth = W;
    int villageStart = (W - villageWidth) / 2;
    int villageEnd = villageStart + villageWidth - 1;

    vector<int> w(W);
    for (int c = 0; c < W; c++) {
        if (c >= villageStart && c <= villageEnd) w[c] = rnd.next(6, 9);
        else w[c] = rnd.next(1, 3);
    }

    vector<int> villageCols, otherCols;
    for (int c = 0; c < W; c++) {
        if (c >= villageStart && c <= villageEnd) villageCols.push_back(c);
        else otherCols.push_back(c);
    }
    // Fisher-Yates shuffle both column pools (deterministic via rnd, seeded by testId).
    for (int i = (int)villageCols.size() - 1; i > 0; i--) swap(villageCols[i], villageCols[rnd.next(0, i)]);
    for (int i = (int)otherCols.size() - 1; i > 0; i--) swap(otherCols[i], otherCols[rnd.next(0, i)]);

    int noiseCount = min((int)llround(K * noiseFrac), (int)otherCols.size());
    int threatCount = min(K - noiseCount, (int)villageCols.size());
    // If the village band was too narrow to host all threats, spill the remainder
    // into extra noise columns (still deterministic, still bounded by otherCols size).
    int shortfall = (K - noiseCount) - threatCount;
    if (shortfall > 0) noiseCount = min(noiseCount + shortfall, (int)otherCols.size());

    vector<ll> relC, relM, relP;

    for (int i = 0; i < threatCount; i++) {
        int c = villageCols[i];
        bool isTrap = (rnd.next(0.0, 1.0) < trapFrac);
        double M = isTrap ? rnd.next(1.30, 2.30) : rnd.next(0.30, 0.85);
        ll p = max(1LL, (ll)llround(capacityFull * M));
        ll m = rnd.next(50, 10000);
        relC.push_back(c); relM.push_back(m); relP.push_back(p);
    }
    for (int i = 0; i < noiseCount; i++) {
        int c = otherCols[i];
        double M = rnd.next(0.05, 0.40);
        ll p = max(1LL, (ll)llround(capacityFull * M));
        ll m = rnd.next(1, 500);
        relC.push_back(c); relM.push_back(m); relP.push_back(p);
    }

    int Kact = (int)relC.size();
    if (Kact < 1) { // ultra-defensive fallback, should not trigger given the sizes above
        relC.push_back(villageStart); relM.push_back(100); relP.push_back(capacityFull * 2);
        Kact = 1;
    }
    ll L = (ll)Kact * HMAX;

    printf("%d %d %d %lld %d %d\n", H, W, Kact, L, SCALE, DISS);
    for (int c = 0; c < W; c++) printf("%d%c", w[c], c + 1 < W ? ' ' : '\n');
    for (int i = 0; i < Kact; i++) printf("%lld %lld %lld\n", relC[i], relM[i], relP[i]);

    return 0;
}
