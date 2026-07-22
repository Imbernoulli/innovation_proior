#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// -----------------------------------------------------------------------------
// "Ducting Days: Channels That Survive the Sweep"  (generator)
// family: ducting-channel-gradient
//
// N stations sit at integer positions x_1<...<x_N along ONE corridor axis.
// Base interference edges connect pairs within R_base; each edge's weight is
// an independent random positive integer (NOT a function of distance -- this
// matters, see below). K=40 ducting scenarios: scenario s activates edges
// between pairs whose separation distance lies in the band
// [Lo_s, Hi_s] = [D0+(s-1), D0+(s-1)+WD-1]. Because the step between
// consecutive bands is exactly 1, the 40 scenarios are ONE sliding window
// sweeping across a contiguous range of separation distances
// [D0, D0+39+WD-1] -- a single geometric family, not 40 unrelated graphs.
//
// NON-TRAP tests (mode 0, testIds 1-4,8): D0 is placed far beyond any
// reachable separation distance, so zero ducting edges are ever generated --
// these cases isolate LOCAL coloring quality. Spacing is irregular (1..Rbase/2)
// and C is kept well BELOW Rbase+1, forcing real pigeonhole conflicts almost
// everywhere. Because edge weights are randomized (not distance-based), the
// round-robin baseline's fixed index-period C is no longer a coincidental
// near-optimum here (it would be, for uniform spacing + distance-decreasing
// weights): an adaptive per-station min-conflict greedy coloring reliably
// beats it by picking up the locally cheapest collisions.
//
// TRAP tests (mode 1, testIds 5,6,9,10; NEEDLE mode 2, testId 7): stations lie
// on a UNIFORM unit-spacing stretch. A base-graph-only greedy coloring there
// settles into a periodic pattern with a SMALL period (about R_base+1, the
// local clique size). Because the 40-scenario sweep covers a wide (42-unit)
// contiguous range of separation distances, some multiple of that small
// period is guaranteed (pigeonhole) to fall inside the sweep, causing a mass
// simultaneous collision on the worst-case scenario -- while round-robin's
// LARGER period C can be, and is, deliberately placed outside the swept
// range (testIds 5,6,9: C > 42) or only partially exposed (testIds 7,10:
// C < 42), giving genuine, tunable headroom between greedy and strong.
// testId 7 additionally embeds the uniform trap block inside mostly random
// spacing, so the failure is localized (a NEEDLE) rather than global.
// -----------------------------------------------------------------------------

struct TC { int N, C, Rbase; ll D0; int mode; };
// mode: 0 = irregular spacing, no ducting (local-coloring-only tension)
//       1 = uniform spacing, ducting trap (D0 chosen per comment above)
//       2 = needle: uniform trap block embedded in random spacing
static TC PARAMTAB[10] = {
    /*1*/ {  15,  3, 10, 999999, 0},
    /*2*/ {  30,  4, 12, 999999, 0},
    /*3*/ {  50,  4, 14, 999999, 0},
    /*4*/ {  70,  5, 16, 999999, 0},
    /*5*/ { 150, 50,  5,      7, 1},
    /*6*/ { 220, 55,  6,      8, 1},
    /*7*/ { 260, 20,  5,     18, 2},
    /*8*/ { 400,  6, 20, 999999, 0},
    /*9*/ { 650, 60,  7,      9, 1},
    /*10*/{ 900, 34,  8,     30, 1},
};

static const int K = 40;
static const int WD = 3;   // duct band width
static const int DUCTW = 10;

int main(int argc, char* argv[]) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);
    TC tc = PARAMTAB[(testId - 1) % 10];

    int N = tc.N, C = tc.C, Rbase = tc.Rbase;
    ll D0 = tc.D0;

    // ---- build positions (strictly increasing) ----
    vector<ll> x(N);
    ll cur = 0;
    if (tc.mode == 1) {
        // uniform unit spacing over the whole corridor
        for (int i = 0; i < N; i++) { x[i] = cur; cur += 1; }
    } else if (tc.mode == 2) {
        // needle: random / uniform(trap) / random
        int needleLen = N * 35 / 100;          // ~35% embedded trap block
        int preLen = (N - needleLen) / 2;
        int postLen = N - needleLen - preLen;
        for (int i = 0; i < preLen; i++) { x[i] = cur; cur += rnd.next(4, 9); }
        for (int i = 0; i < needleLen; i++) { x[preLen + i] = cur; cur += 1; }
        for (int i = 0; i < postLen; i++) { x[preLen + needleLen + i] = cur; cur += rnd.next(4, 9); }
    } else {
        // Irregular spacing (1..Rbase): local density varies a lot, so the
        // channel budget C (kept well BELOW Rbase+1) forces genuine pigeonhole
        // conflicts almost everywhere. Combined with per-edge weights that do
        // NOT simply decrease with distance, the periodic round-robin pattern
        // is no longer a coincidental optimum here -- an adaptive local
        // min-conflict coloring can do meaningfully better.
        int hi = max(1, Rbase / 2);
        for (int i = 0; i < N; i++) { x[i] = cur; cur += rnd.next(1, hi); }
    }

    // ---- base interference edges (two-pointer, sorted positions) ----
    vector<array<ll,3>> baseE; // i,j,w (1-indexed)
    {
        int r = 0;
        for (int i = 0; i < N; i++) {
            if (r < i + 1) r = i + 1;
            while (r < N && x[r] - x[i] <= Rbase) r++;
            for (int j = i + 1; j < r; j++) {
                ll w = rnd.next(1, 9); // weight is NOT simply distance-decreasing
                baseE.push_back({(ll)i + 1, (ll)j + 1, w});
            }
        }
    }

    // ---- K duct scenarios: band s = [D0+(s-1), D0+(s-1)+WD-1] ----
    vector<vector<array<ll,3>>> ductE(K + 1);
    for (int s = 1; s <= K; s++) {
        ll Lo = D0 + (s - 1), Hi = Lo + WD - 1;
        auto &vec = ductE[s];
        for (int i = 0; i < N; i++) {
            ll loX = x[i] + Lo, hiX = x[i] + Hi;
            int j0 = (int)(lower_bound(x.begin() + i + 1, x.end(), loX) - x.begin());
            int j1 = (int)(upper_bound(x.begin() + i + 1, x.end(), hiX) - x.begin());
            for (int j = j0; j < j1; j++)
                vec.push_back({(ll)i + 1, (ll)j + 1, (ll)DUCTW});
        }
    }

    // ---- emit ----
    printf("%d %d %d\n", N, C, K);
    for (int i = 0; i < N; i++) printf("%lld%c", x[i], i + 1 < N ? ' ' : '\n');
    printf("%d\n", (int)baseE.size());
    for (auto &e : baseE) printf("%lld %lld %lld\n", e[0], e[1], e[2]);
    for (int s = 1; s <= K; s++) {
        printf("%d\n", (int)ductE[s].size());
        for (auto &e : ductE[s]) printf("%lld %lld %lld\n", e[0], e[1], e[2]);
    }
    return 0;
}
