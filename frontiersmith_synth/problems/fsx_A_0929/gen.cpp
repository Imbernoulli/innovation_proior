#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// -----------------------------------------------------------------------------
// Generator for "Collapsing Pigeonhole Cabinets"
// family: variable-stride-trie-budget
//
// Emits: K M / per cabinet c: (D_c W_c) then T_c[0..D_c]
//
// Each cabinet is parameterized by a "branch density" p in (0,1): starting
// from T[0]=1, T[d] = clamp(T[d-1] + floor(T[d-1]*p) + jitter, 1, 2*T[d-1]).
// p near 1 -> cabinet is "bushy" (dense): a SINGLE wide probe is both
// cheapest AND fewest-probes for it -- no real tension, just recognize it.
// p small  -> cabinet is a "thin chain" (sparse): merging probes is a real
// memory-for-probes TRADE-OFF (monotonically more memory buys fewer probes).
//
// The shared budget M = baseline_mem + alpha*(fullcollapse_mem - baseline_mem)
// is always >= the do-nothing baseline (with margin) but a small fraction of
// full single-level collapse everywhere, so the budget is a real bottleneck.
//
// Ladder (testId 1..10):
//  1-2  tiny sanity, mild density/weight, generous alpha.
//  3,5,6,8  TRAP: two classes -- "dense+popular" (expensive, high W_c) vs
//           "sparse+modest" (cheap, lower W_c). A popularity-proportional
//           budget split over-funds the expensive class and starves the
//           cheap class, even though the cheap class converts memory into
//           probe-savings far more efficiently.
//  4,9  NEEDLE: many small-weight filler cabinets plus 1-2 planted cabinets
//       that are BOTH cheap (sparse) AND highly popular -- easy to miss if
//       budget is split by a coarse per-cabinet rule.
//  7    general random-correlation stress test.
//  10   largest scale: K up to ~450, D up to 20, zipfian popularity
//       correlated with density (trap at scale), tight alpha.
// -----------------------------------------------------------------------------

static vector<ll> buildT(int D, double p, double jitterAmt) {
    // Track a continuous "expected" occupancy val = (1+p)^d (so p genuinely
    // compounds even while val is small) and round to the nearest feasible
    // integer T[d] in [1, 2*T[d-1]] each step, with light jitter noise.
    vector<ll> T(D + 1);
    T[0] = 1;
    double val = 1.0;
    for (int d = 1; d <= D; d++) {
        val *= (1.0 + p);
        double noise = 1.0 + (rnd.next(0, 1000) / 1000.0 - 0.5) * jitterAmt * 0.3;
        ll t = (ll)llround(val * noise);
        ll maxT = 2 * T[d - 1];
        if (t < 1) t = 1;
        if (t > maxT) t = maxT;
        if (t > (1LL << 20)) t = (1LL << 20);
        T[d] = t;
    }
    return T;
}

struct Cabinet {
    int D;
    ll W;
    vector<ll> T;
};

static ll baselineMem(const Cabinet& c) {
    ll m = 0;
    for (int d = 0; d < c.D; d++) m += c.T[d] * 2;
    return m;
}

static ll fullCollapseMem(const Cabinet& c) {
    return c.T[0] * (1LL << c.D);   // T[0] == 1 always -> 2^D
}

int main(int argc, char* argv[]) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);

    int K;
    int Dlo, Dhi;
    double alpha;               // fraction of (full - baseline) gap granted as budget surplus
    // mode: 0 = plain random, 1 = trap (dense-popular vs sparse-cheap), 2 = needle
    int mode;

    switch (testId) {
        case 1: K = 3;   Dlo = 3;  Dhi = 5;  alpha = 0.30; mode = 0; break;
        case 2: K = 5;   Dlo = 4;  Dhi = 7;  alpha = 0.22; mode = 0; break;
        case 3: K = 9;   Dlo = 7;  Dhi = 11; alpha = 0.07; mode = 1; break;
        case 4: K = 16;  Dlo = 8;  Dhi = 14; alpha = 0.05; mode = 2; break;
        case 5: K = 13;  Dlo = 8;  Dhi = 14; alpha = 0.055; mode = 1; break;
        case 6: K = 18;  Dlo = 10; Dhi = 16; alpha = 0.018; mode = 1; break;
        case 7: K = 22;  Dlo = 6;  Dhi = 18; alpha = 0.06; mode = 0; break;
        case 8: K = 26;  Dlo = 4;  Dhi = 20; alpha = 0.045; mode = 1; break;
        case 9: K = 65;  Dlo = 8;  Dhi = 16; alpha = 0.014; mode = 2; break;
        default: K = 420; Dlo = 6; Dhi = 20; alpha = 0.007; mode = 1; break;
    }

    vector<Cabinet> cabs(K);

    // needle-mode planted indices
    set<int> needles;
    if (mode == 2) {
        int nn = max(1, K / 30 + (testId == 9 ? 1 : 0));
        while ((int)needles.size() < nn) needles.insert(rnd.next(0, K - 1));
    }

    for (int c = 0; c < K; c++) {
        int D = rnd.next(Dlo, Dhi);
        double p, jitter;
        ll W;

        if (mode == 1) {
            // dense-popular vs sparse-cheap split
            bool densePopular = (rnd.next(0, 99) < 45);
            if (densePopular) {
                p = 0.40 + rnd.next(0, 18) / 100.0;          // 0.40..0.58 (bushy, poor ROI, never a free lunch)
                W = 8000 + rnd.next(0, 16000);                 // high popularity
            } else {
                p = 0.08 + rnd.next(0, 18) / 100.0;          // 0.08..0.26 (thin chain, cheap)
                W = 300 + rnd.next(0, 2500);                    // modest popularity
            }
            jitter = 0.4;
        } else if (mode == 2) {
            if (needles.count(c)) {
                p = 0.06 + rnd.next(0, 12) / 100.0;          // very cheap to compress
                W = 25000 + rnd.next(0, 25000);                 // very popular
            } else {
                p = 0.25 + rnd.next(0, 45) / 100.0;          // filler, moderate density
                W = 20 + rnd.next(0, 400);                      // low popularity
            }
            jitter = 0.35;
        } else {
            p = 0.10 + rnd.next(0, 55) / 100.0;              // 0.10..0.65 spread
            W = 1 + rnd.next(0, 6000);
            jitter = 0.3;
        }

        // zipfian tail on the largest tests: a few cabinets get a big weight bump
        if (testId == 10 && rnd.next(0, 99) < 3) W += 40000 + rnd.next(0, 60000);

        vector<ll> T = buildT(D, p, jitter);
        cabs[c] = {D, W, T};
    }

    ll baseTotal = 0, fullTotal = 0;
    for (auto& c : cabs) { baseTotal += baselineMem(c); fullTotal += fullCollapseMem(c); }
    if (fullTotal < baseTotal) fullTotal = baseTotal;   // safety, shouldn't happen

    ll surplusPool = (ll)llround(alpha * (double)(fullTotal - baseTotal));
    ll M = baseTotal + surplusPool + 4;   // small fixed margin so baseline always fits exactly
    if (M < baseTotal + 1) M = baseTotal + 1;

    // ---- emit ----
    printf("%d %lld\n", K, M);
    for (auto& c : cabs) {
        printf("%d %lld\n", c.D, c.W);
        for (int d = 0; d <= c.D; d++) printf("%lld%c", c.T[d], d == c.D ? '\n' : ' ');
    }
    return 0;
}
