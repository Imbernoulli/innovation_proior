#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

// Wildlife Corridor Relocation Run generator.
// testId is a difficulty/structure ladder from tiny (example scale) to large.

static long long manh(long long ax, long long ay, long long bx, long long by){
    return llabs(ax-bx) + llabs(ay-by);
}

int main(int argc, char* argv[]) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);

    // ---- size ladder ----
    int Ptable[11] = {0, 3, 15, 40, 70, 100, 140, 180, 220, 260, 300};
    int P = (testId >= 1 && testId <= 10) ? Ptable[testId] : 100;

    // ---- capacity ladder: alternate tight vs loose ----
    int Q;
    if (testId % 3 == 1) Q = 6;        // tight -> batching contended
    else if (testId % 3 == 2) Q = 14;  // medium
    else Q = 30;                       // loose

    // coordinate range
    int LO = 0, HI = 10000;

    // depot roughly central-ish (deterministic per test)
    int x0 = rnd.next(LO, HI);
    int y0 = rnd.next(LO, HI);

    // corridor structure: for larger/odd tests use clustered sites along a
    // diagonal "corridor"; for even tests use uniform scatter.
    bool clustered = (testId % 2 == 1);
    int nClusters = 3 + testId; // a handful of stepping-stone patches

    struct Cl { int cx, cy; };
    vector<Cl> cls(nClusters);
    for (int c = 0; c < nClusters; c++) {
        // spread cluster centres along the main diagonal of the map
        double frac = (nClusters == 1) ? 0.5 : (double)c / (nClusters - 1);
        int base = (int)round(frac * (HI - LO)) + LO;
        int jitter = 800;
        cls[c].cx = min(HI, max(LO, base + rnd.next(-jitter, jitter)));
        cls[c].cy = min(HI, max(LO, base + rnd.next(-jitter, jitter)));
    }

    auto samplePoint = [&](int spread)->pair<int,int>{
        if (clustered) {
            int c = rnd.next(0, nClusters - 1);
            int x = cls[c].cx + rnd.next(-spread, spread);
            int y = cls[c].cy + rnd.next(-spread, spread);
            x = min(HI, max(LO, x));
            y = min(HI, max(LO, y));
            return {x, y};
        } else {
            return {rnd.next(LO, HI), rnd.next(LO, HI)};
        }
    };

    // penalty-scale ladder: some tests mostly-profitable, some mostly-not.
    // wFactorPct is a percentage applied to an estimated solo haulage cost.
    // range width also varies so a mix of profitable/unprofitable requests appears.
    int loPct, hiPct;
    switch (testId % 4) {
        case 0: loPct = 30;  hiPct = 90;  break;  // mostly unprofitable
        case 1: loPct = 60;  hiPct = 160; break;  // balanced
        case 2: loPct = 90;  hiPct = 220; break;  // mostly profitable
        default: loPct = 45; hiPct = 130; break;  // balanced-low
    }

    // build requests
    struct Req { int px, py, dx, dy, q; long long w; };
    vector<Req> reqs(P);
    int qmax = min(Q, 8);
    for (int i = 0; i < P; i++) {
        auto pk = samplePoint(600);
        auto dl = samplePoint(600);
        int q = rnd.next(1, qmax);
        // estimate solo haulage: depot->pk (load 0), pk->dl (load q), dl->depot (load 0)
        long long solo = manh(x0, y0, pk.first, pk.second) * 1
                       + manh(pk.first, pk.second, dl.first, dl.second) * (1 + q)
                       + manh(dl.first, dl.second, x0, y0) * 1;
        if (solo < 1) solo = 1;
        int pct = rnd.next(loPct, hiPct);
        long long w = (solo * (long long)pct) / 100;
        if (w < 1) w = 1;
        if (w > 50000000LL) w = 50000000LL;
        reqs[i] = {pk.first, pk.second, dl.first, dl.second, q, w};
    }

    // ---- output ----
    printf("%d %d\n", P, Q);
    printf("%d %d\n", x0, y0);
    for (int i = 0; i < P; i++) {
        printf("%d %d %d %d %d %lld\n",
               reqs[i].px, reqs[i].py, reqs[i].dx, reqs[i].dy,
               reqs[i].q, reqs[i].w);
    }
    return 0;
}
