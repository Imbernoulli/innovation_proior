#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// -----------------------------------------------------------------------------
// "Accordion Silhouette Scheduling" (generator)  family: flat-fold-crease-order
//
// n panels, n-1 creases. Crease i is EXPOSED iff i==1, i==n-1, or a neighbor is
// already committed (access-blocking-order): only two frontiers (left pointer
// growing from 1, right pointer shrinking from n-1) ever advance; interior
// creases are buried until a neighbor clears them (fold-order-commit).
//
// Objective rewards LATE commits (prominence multiplier = commit step) and a
// global Mountain/Valley parity target T (layer-parity-invariant, Maekawa/
// Kawasaki-style): missing T halves the WHOLE score.
//
// TRAP: the obvious approach commits whichever exposed crease currently has the
// BIGGEST weight (a natural "grab the best available option now" instinct) --
// but since value = weight * step, grabbing a big weight EARLY wastes it on a
// small multiplier. The insight is the opposite: commit the SMALLER of the two
// currently-exposed weights first, saving big weights for late (high-multiplier)
// steps, PLUS reason about the global M/V parity target (which the naive
// approach, always using req_i and folding everything, never manages).
// -----------------------------------------------------------------------------

static vector<char> genReq(ll km1, int pctM){
    vector<char> req(km1 + 1);
    for (ll i = 1; i <= km1; i++) req[i] = (rnd.next(100) < pctM) ? 'M' : 'V';
    return req;
}

static ll naturalGap(const vector<char>& req, ll km1){
    ll g = 0;
    for (ll i = 1; i <= km1; i++) g += (req[i] == 'M') ? 1 : -1;
    return g;
}

// T with the SAME parity as km1 (reachable by flips alone, no drop needed),
// offset from the natural gap by roughly 2*shift.
static ll tSameParity(ll natGap, ll km1, int shift){
    ll cand = natGap - 2LL * shift;
    if (cand > km1) cand = km1 - ((km1 - cand) % 2 == 0 ? 0 : 1);
    if (cand < -km1) cand = -km1 + ((km1 + cand) % 2 == 0 ? 0 : 1);
    return cand;
}

// T with OPPOSITE parity from km1 (forces dropping exactly one crease).
static ll tOppositeParity(ll natGap, ll km1){
    ll cand = natGap + 1;
    if (cand > km1) cand = natGap - 1;
    if (cand < -km1) cand = natGap + 1; // km1>=2 so one of the two fits
    return cand;
}

static void printCase(ll n, ll T, const vector<char>& req, const vector<ll>& w){
    ll km1 = n - 1;
    printf("%lld %lld\n", n, T);
    for (ll i = 1; i <= km1; i++) printf("%c%c", req[i], (i == km1) ? '\n' : ' ');
    for (ll i = 1; i <= km1; i++) printf("%lld%c", w[i], (i == km1) ? '\n' : ' ');
}

int main(int argc, char* argv[]){
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);

    ll n, km1;
    vector<char> req;
    vector<ll> w;
    ll T;

    if (testId == 1){
        // tiny sanity case
        n = 6; km1 = n - 1;
        req = genReq(km1, 50);
        w.assign(km1 + 1, 0);
        for (ll i = 1; i <= km1; i++) w[i] = 1 + rnd.next(0, 49);
        ll ng = naturalGap(req, km1);
        T = tSameParity(ng, km1, 0);
    } else if (testId == 2){
        n = 10; km1 = n - 1;
        req = genReq(km1, 60);
        w.assign(km1 + 1, 0);
        for (ll i = 1; i <= km1; i++) w[i] = 1 + rnd.next(0, 199);
        ll ng = naturalGap(req, km1);
        T = tSameParity(ng, km1, 1);
    } else if (testId == 3){
        // TRAP: weight strictly increases toward the RIGHT boundary (which is
        // exposed from step 1) -- naive greedy grabs the peak immediately.
        n = 60; km1 = n - 1;
        req = genReq(km1, 50);
        w.assign(km1 + 1, 0);
        for (ll i = 1; i <= km1; i++){
            ll base = 5 + i * 7;
            ll noise = rnd.next(-3, 3);
            w[i] = max(1LL, base + noise);
        }
        ll ng = naturalGap(req, km1);
        T = tOppositeParity(ng, km1); // forces drop + flips
    } else if (testId == 4){
        // TRAP mirror: weight strictly increases toward the LEFT boundary.
        n = 60; km1 = n - 1;
        req = genReq(km1, 50);
        w.assign(km1 + 1, 0);
        for (ll i = 1; i <= km1; i++){
            ll base = 5 + (km1 + 1 - i) * 7;
            ll noise = rnd.next(-3, 3);
            w[i] = max(1LL, base + noise);
        }
        ll ng = naturalGap(req, km1);
        T = ng; // pure position-insight test, no parity confound
    } else if (testId == 5){
        n = 300; km1 = n - 1;
        req = genReq(km1, 45);
        w.assign(km1 + 1, 0);
        for (ll i = 1; i <= km1; i++) w[i] = 1 + rnd.next(0, 999);
        ll ng = naturalGap(req, km1);
        T = tSameParity(ng, km1, 2);
    } else if (testId == 6){
        // TRAP: periodic spikes scattered across the whole strip (near both
        // boundaries AND the interior) to break many local frontier choices.
        n = 800; km1 = n - 1;
        req = genReq(km1, 50);
        w.assign(km1 + 1, 0);
        for (ll i = 1; i <= km1; i++){
            ll base = 1 + rnd.next(0, 9);
            if (i % 11 == 0) base += 3000 + rnd.next(0, 500);
            w[i] = base;
        }
        ll ng = naturalGap(req, km1);
        T = tOppositeParity(ng, km1);
    } else if (testId == 7){
        // TRAP + NEEDLE: mostly tiny weights, a handful of huge spikes at
        // random positions (including near the boundaries).
        n = 3000; km1 = n - 1;
        req = genReq(km1, 55);
        w.assign(km1 + 1, 0);
        for (ll i = 1; i <= km1; i++) w[i] = 1 + rnd.next(0, 4);
        int needles = 6;
        w[1] = 8000 + rnd.next(0, 5000);       // needle right at a boundary
        w[km1] = 8000 + rnd.next(0, 5000);     // needle at the other boundary
        for (int q = 0; q < needles - 2; q++){
            ll pos = 2 + rnd.next(0, (int)(km1 - 3));
            w[pos] = 8000 + rnd.next(0, 5000);
        }
        ll ng = naturalGap(req, km1);
        T = ng;
    } else if (testId == 8){
        n = 20000; km1 = n - 1;
        req = genReq(km1, 62);
        w.assign(km1 + 1, 0);
        for (ll i = 1; i <= km1; i++) w[i] = 1 + rnd.next(0, 999);
        ll ng = naturalGap(req, km1);
        T = tSameParity(ng, km1, 3);
    } else if (testId == 9){
        // TRAP at scale: big spikes clustered near BOTH boundaries (decaying
        // inward), tiny elsewhere in the middle.
        n = 80000; km1 = n - 1;
        req = genReq(km1, 50);
        w.assign(km1 + 1, 0);
        for (ll i = 1; i <= km1; i++) w[i] = 1 + rnd.next(0, 19);
        ll edge = min<ll>(300, km1 / 4);
        for (ll j = 0; j < edge; j++){
            ll val = 5000 + rnd.next(0, 15000) - (j * (15000 / max<ll>(1, edge)));
            val = max(1LL, val);
            w[1 + j] = val;
            w[km1 - j] = val + rnd.next(-50, 50);
            if (w[km1 - j] < 1) w[km1 - j] = 1;
        }
        ll ng = naturalGap(req, km1);
        T = tOppositeParity(ng, km1);
    } else { // testId == 10, max scale: mixed random + needle spikes + drift
        n = 200000; km1 = n - 1;
        req = genReq(km1, 58);
        w.assign(km1 + 1, 0);
        for (ll i = 1; i <= km1; i++){
            ll drift = 1 + (i * 300) / km1; // mild monotonic drift, small vs spikes
            w[i] = drift + rnd.next(0, 300);
        }
        int needles = 12;
        for (int q = 0; q < needles; q++){
            ll pos = 1 + rnd.next(0, (int)(km1 - 1));
            w[pos] = 5000 + rnd.next(0, 25000);
        }
        ll ng = naturalGap(req, km1);
        T = tSameParity(ng, km1, 4);
    }

    printCase(n, T, req, w);
    return 0;
}
