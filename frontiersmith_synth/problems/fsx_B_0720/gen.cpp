#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// -----------------------------------------------------------------------------
// "Brewery batches sharing one thermal storage tank"  (generator)
// family: batch-heat-handshake-scheduling
//
// G CRITICAL consumers (vol=100, req 70..95, price 2..5) each have a narrow
// window [slot+6, slot+11] tied to their own cluster slot. Heat for them
// comes from a SHARED POOL of only Hcnt (~60% of G) HOT donors (vol=100,
// temp 85..100) -- donors are NOT tied to any particular consumer; each has
// a WIDE window [2, H] and can be aimed at whichever consumer is worth it.
// Because donors are scarce, not every critical consumer can be served --
// deciding WHICH ones (by value = price*req) is a genuine choice, and even
// the best schedule leaves some consumers unserved (an intrinsic cost floor
// that keeps the score from saturating).
//
// Around this: many COLD donors (vol=80, temp 5..30) and BULK consumers
// (vol=150, req 3..15, cheap to under-serve) have an "open" window: an
// INDEPENDENT random anchor spread across roughly the first half of the
// horizon, extending all the way to H. An "earliest start" default places
// each one at its own scattered anchor -- a persistent trickle of
// contamination throughout the horizon, not a one-shot event that fades --
// while deferring them to the LATEST point always means the same thing:
// push them past every cluster, out of the way, to H.
//
// The trap: an "obvious" matching heuristic assigns scarce hot donors to
// critical consumers in ENCOUNTER order (earliest slot / input order), not
// by how much each one is actually worth (price*req) -- so it happily burns
// a donor on a cheap, low-value consumer while an expensive one goes unmet.
// The insight is to rank by VALUE and match greedily downward, and to keep
// every donor's deposit immediately adjacent to the consumer it was aimed
// at (minimizing what else can land between them).
//
// One "forced-open" consumer has window [1,1]; no donor's window can start
// at 1 (e>=2), so it ALWAYS draws from an empty tank -- a fixed cost floor
// present in every solution, which keeps F away from 0.
// -----------------------------------------------------------------------------

struct Task { int type; ll vol, e, l, p1, p2; };

static ll Hmax_g;
// Background window: an INDEPENDENT random anchor (spread across roughly the
// first half of the horizon), open all the way to H.
pair<ll,ll> openWindow(){
    ll a = 2 + rnd.next(0, (int)max(1LL, Hmax_g / 2));
    return {a, Hmax_g};
}

int main(int argc, char* argv[]){
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);
    double f = (testId - 1) / 9.0;

    int G = 3 + (int)llround(f * 27.0);                // 3..30 critical consumers
    int GAP = 55;
    ll slot0 = 200;
    Hmax_g = slot0 + (ll)G * GAP + 400;
    ll Hmax = Hmax_g;
    ll C = 260 + 5 * testId;                            // 265..310 (moderately tight)

    int Hcnt = (int)((ll)G * 60 / 100);                 // scarce shared donor pool
    int nCold = 2 * G, nBulk = G;
    int nExtraCold = (int)llround(f * 30.0);            // background noise
    int nExtraBulk = (int)llround(f * 22.0);

    vector<Task> tasks;

    // forced-open consumer: always draws from an empty tank.
    tasks.push_back({1, 50, 1, 1, 20, 2});

    for (int c = 0; c < G; c++){
        ll slot = slot0 + (ll)c * GAP;
        ll req = 70 + rnd.next(0, 25);                  // 70..95
        ll price = 2 + rnd.next(0, 8);                   // 2..10
        tasks.push_back({1, 100, slot + 6, slot + 11, req, price});     // critical consumer
    }
    for (int j = 0; j < Hcnt; j++){
        ll hotTemp = 85 + rnd.next(0, 15);              // 85..100
        tasks.push_back({0, 100, 2, Hmax, hotTemp, 0});  // shared hot donor (wide window)
    }

    for (int j = 0; j < nCold; j++){
        ll coldTemp = 5 + rnd.next(0, 25);               // 5..30
        auto w = openWindow();
        tasks.push_back({0, 80, w.first, w.second, coldTemp, 0});   // cold donor
    }
    for (int j = 0; j < nBulk; j++){
        ll bReq = 3 + rnd.next(0, 12);                   // 3..15
        ll bPrice = 1 + rnd.next(0, 2);                   // 1..3
        auto w = openWindow();
        tasks.push_back({1, 150, w.first, w.second, bReq, bPrice});
    }
    for (int j = 0; j < nExtraCold; j++){
        ll coldTemp = 5 + rnd.next(0, 25);
        ll vol = 40 + rnd.next(0, 60);
        auto w = openWindow();
        tasks.push_back({0, vol, w.first, w.second, coldTemp, 0});
    }
    for (int j = 0; j < nExtraBulk; j++){
        ll bReq = 3 + rnd.next(0, 12);
        ll bPrice = 1 + rnd.next(0, 2);
        ll vol = 40 + rnd.next(0, 100);
        auto w = openWindow();
        tasks.push_back({1, vol, w.first, w.second, bReq, bPrice});
    }

    // shuffle input order (id-tiebreak should not leak the pool structure)
    for (int i = (int)tasks.size() - 1; i > 0; i--) swap(tasks[i], tasks[rnd.next(0, i)]);

    int N = (int)tasks.size();
    printf("%d %lld %lld\n", N, Hmax, C);
    for (auto &t : tasks)
        printf("%d %lld %lld %lld %lld %lld\n", t.type, t.vol, t.e, t.l, t.p1, t.p2);
    return 0;
}
