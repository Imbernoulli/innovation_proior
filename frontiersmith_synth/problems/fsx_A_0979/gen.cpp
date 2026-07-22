#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
typedef unsigned long long u64;

// -----------------------------------------------------------------------------
// "Nested Customs Stamps" (generator)
// family: bloom-bit-cascade-splitter
//
// T=8 fixed nested checkpoints. Checkpoint i owns a true watch-list C_i (sizes
// n_i pairwise disjoint IDs) and, in the checker, a Bloom filter sized/keyed by
// the participant's chosen (m_i,k_i). Container-ID universe is split into two
// DISJOINT ranges so ground truth is unambiguous:
//   member range  [1, 5*10^8]        -- every C_i element lives here
//   clean range   [1_000_000_001, 2*10^9]  -- pure noise IDs, never a true member
//
// Watch-list sizes n_i are drawn i.i.d. (independent of tier index i) -- by
// design NOT correlated with checkpoint depth, so a recipe cannot get a free
// pass just because "the expensive tier happens to have few members". The
// trap is purely the fixed exponential cost weights (1,2,4,...,128) plus a
// TIGHT ink budget: any cost-BLIND allocation (equal split, or the classic
// equalize-the-analytic-FP-rate recipe) ends up with a similar false-positive
// rate at every tier, and since checkpoint 8's false alarms cost 128x
// checkpoint 1's, that "fair" rate is ruinous precisely at the tier where it
// matters least to be fair.
//
// testId ladder:
//   t=1..4  : looser ink budget (bits/element ~8.5-10) -- naive recipes are
//             only mildly suboptimal, sanity/warm-up scale.
//   t=5..10 : TRAP-HEAVY -- tight ink budget (bits/element ~3.75-4.5) forces
//             genuine trade-offs, and a high clean-noise trace fraction means
//             most containers pay for whatever false alarms the checkpoints
//             hand them. t=10 also maximizes trace length Q and tier sizes to
//             fill the stated size envelope.
// Trace composition: cleanFrac of the Q arrivals are pure noise (clean
// range); the rest are genuine members drawn with a mild depth-biased tier
// weight i (deep checkpoints see a bit more of their own real traffic too).
// -----------------------------------------------------------------------------

static const int T = 8;
static const ll MEMBER_LO = 1, MEMBER_HI = 500000000LL;
static const ll CLEAN_LO = 1000000001LL, CLEAN_HI = 2000000000LL;

int main(int argc, char* argv[]) {
    registerGen(argc, argv, 1);
    int t = atoi(argv[1]); // 1..10

    bool trapHeavy = (t >= 5);
    ll nBase = 150 + 400LL * t;

    vector<int> n(T + 1, 0);
    ll sumN = 0;
    for (int i = 1; i <= T; i++) {
        double frac = rnd.next(0.15, 2.2); // i.i.d., decoupled from tier index
        int ni = (int)llround((double)nBase * frac);
        ni = max(8, min(20000, ni));
        n[i] = ni;
        sumN += ni;
    }
    // cap sum n_i <= 40000 by uniform scale-down if needed
    if (sumN > 40000) {
        double scale = 40000.0 / (double)sumN;
        sumN = 0;
        for (int i = 1; i <= T; i++) {
            n[i] = max(8, (int)(n[i] * scale));
            sumN += n[i];
        }
    }

    ll Q = min(40000LL, 300LL + 400LL * (ll)t * (ll)t);

    double bitsPerElem = trapHeavy ? (3.25 + 0.125 * t) : (8.0 + 0.5 * t);
    ll B = (ll)llround((double)sumN * bitsPerElem);
    B = max(B, (ll)(T * 64 * 4)); // always comfortably feasible
    B = min(B, (ll)2000000);

    ll SEED1 = rnd.next(1LL, (1LL << 62));
    ll SEED2 = rnd.next(1LL, (1LL << 62));

    printf("%d %lld %lld %lld\n", T, B, SEED1, SEED2);

    // ---- build pairwise-disjoint watch-lists C_i from the member range ----
    unordered_set<ll> used;
    used.reserve((size_t)sumN * 2 + 16);
    vector<vector<ll>> C(T + 1);
    for (int i = 1; i <= T; i++) {
        C[i].reserve(n[i]);
        while ((int)C[i].size() < n[i]) {
            ll x = rnd.next(MEMBER_LO, MEMBER_HI);
            if (used.count(x)) continue;
            used.insert(x);
            C[i].push_back(x);
        }
        sort(C[i].begin(), C[i].end());
        printf("%d\n", n[i]);
        for (int j = 0; j < n[i]; j++) printf("%lld%c", C[i][j], j + 1 == n[i] ? '\n' : ' ');
    }

    // ---- trace: cleanFrac noise + depth-biased genuine members ----
    double cleanFrac = trapHeavy ? 0.85 : 0.60;
    ll tierW[T + 1], totalW = 0;
    for (int i = 1; i <= T; i++) { tierW[i] = (ll)i; totalW += tierW[i]; }

    printf("%lld\n", Q);
    for (ll q = 0; q < Q; q++) {
        ll id;
        if (rnd.next(0.0, 1.0) < cleanFrac) {
            id = rnd.next(CLEAN_LO, CLEAN_HI);
        } else {
            ll r = rnd.next(1LL, totalW);
            int i = 1; ll acc = 0;
            for (; i <= T; i++) { acc += tierW[i]; if (r <= acc) break; }
            if (i > T) i = T;
            int idx = rnd.next(0, n[i] - 1);
            id = C[i][idx];
        }
        printf("%lld%c", id, q + 1 == Q ? '\n' : ' ');
    }
    return 0;
}
