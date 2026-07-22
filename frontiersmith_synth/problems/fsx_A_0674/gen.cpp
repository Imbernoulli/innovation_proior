#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// -----------------------------------------------------------------------------
// "Blizzard Fleet Bootstrap" (generator)  family: plow-backbone-bootstrap
//
// PLANTED STRUCTURE (the checker never sees these labels -- solutions must
// re-derive it from the raw graph via bridge-finding):
//   node 1 = depot, node 2 = hub. A single TRUNK street (bridge) joins them.
//   R "regions" hang off the hub, each via its own SPOKE street (bridge,
//   length 15..60). Inside a region the streets form a small dense subgraph
//   (random tree + extra edges -> NOT bridges, only reachable via its spoke).
//   Region sizes are SKEWED from test 3 onward: one region gets roughly half
//   of the node budget, the rest are small -- so naive round-robin-by-order
//   fleet splitting badly overloads whichever crew inherits the big region,
//   while a workload-aware (LPT bin-packing) split balances it.
// -----------------------------------------------------------------------------

struct Edge { int u, v; ll len; };

int main(int argc, char* argv[]) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);

    struct Cfg { int R, K; ll nodeBudget; bool skewed; };
    static const Cfg table[11] = {
        {0,0,0,false}, // unused index 0
        {2, 2, 6,   false},
        {5, 2, 30,  true},
        {6, 2, 50,  true},
        {8, 2, 70,  true},
        {8, 3, 90,  true},
        {9, 3, 120, true},
        {10,3, 160, true},
        {12,4, 210, true},
        {16,4, 280, true},
        {22,5, 385, true},
    };
    Cfg cfg = table[testId];
    int R = cfg.R, K = cfg.K;
    ll nodeBudget = cfg.nodeBudget;
    bool skewed = cfg.skewed;

    // region sizes n_r (>=2 each)
    vector<ll> sz(R);
    vector<bool> isBig(R, false);
    if (skewed) {
        // PLANTED TRAP: mark every K-th region (indices 0, K, 2K, ...) "big"
        // and give the rest of the spoke lengths a strictly increasing
        // sequence, so sorting regions by spoke length reproduces exactly
        // this generation order. A round-robin-by-sorted-spoke assignment
        // then piles ALL the big regions onto the very same crew (positions
        // 0, K, 2K, ... all map to crew index 0 mod K) -- while a
        // workload-aware bin packing spreads them one-per-crew.
        int numBig = 0;
        for (int r = 0; r < R; r += K) { isBig[r] = true; numBig++; }
        double bigFrac = 0.90;
        ll bigBudget = (ll)llround(nodeBudget * bigFrac);
        ll smallBudget = nodeBudget - bigBudget;
        int numSmall = R - numBig;

        vector<double> wb(R, 0), ws(R, 0);
        double wbSum = 0, wsSum = 0;
        for (int r = 0; r < R; r++) {
            double u = 1.0 + rnd.next(0, 19);
            if (isBig[r]) { wb[r] = u; wbSum += u; }
            else { ws[r] = u; wsSum += u; }
        }
        ll assignedBig = 0, assignedSmall = 0;
        int seenBig = 0, seenSmall = 0;
        for (int r = 0; r < R; r++) {
            if (isBig[r]) {
                seenBig++;
                ll s = (seenBig == numBig) ? (bigBudget - assignedBig)
                                            : (ll)llround(bigBudget * wb[r] / max(1e-9, wbSum));
                s = max((ll)2, s);
                sz[r] = s;
                assignedBig += s;
            } else {
                seenSmall++;
                ll s = (seenSmall == numSmall) ? (smallBudget - assignedSmall)
                                                : (ll)llround(smallBudget * ws[r] / max(1e-9, wsSum));
                s = max((ll)2, s);
                sz[r] = s;
                assignedSmall += s;
            }
        }
    } else {
        ll leftover = nodeBudget;
        int remR = R;
        for (int r = 0; r < R; r++) {
            ll avg = max((ll)2, leftover / max(1, remR));
            ll s = (r == R - 1) ? max((ll)2, leftover) : max((ll)2, avg + rnd.next(-1, 1));
            sz[r] = s;
            leftover -= s;
            remR--;
        }
    }

    vector<Edge> edges;
    int nextId = 3; // 1=depot, 2=hub
    ll trunkLen = rnd.next(35, 80);
    edges.push_back({1, 2, trunkLen});

    ll spokeCursor = 12 + rnd.next(0, 3);
    // Keep the strictly-increasing spoke sequence inside [1,80] regardless of
    // R by shrinking the per-step increment as R grows.
    ll spokeStepLo = 1, spokeStepHi = max((ll)1, (75 - spokeCursor) / max(1, R));
    if (spokeStepHi > 4) spokeStepHi = 4;
    for (int r = 0; r < R; r++) {
        int entry = nextId++;
        ll spokeLen;
        if (skewed) {
            spokeLen = spokeCursor;
            spokeCursor += spokeStepLo + rnd.next(0, (int)(spokeStepHi - spokeStepLo));
        } else {
            spokeLen = rnd.next(15, 60);
        }
        edges.push_back({2, entry, spokeLen});
        ll nr = sz[r];
        vector<int> regionNodes;
        regionNodes.push_back(entry);
        for (ll j = 1; j < nr; j++) {
            int node = nextId++;
            int parent = regionNodes[rnd.next(0, (int)regionNodes.size() - 1)];
            ll len = rnd.next(1, 5);
            edges.push_back({parent, node, len});
            regionNodes.push_back(node);
        }
        ll extra = nr / 4;
        for (ll k = 0; k < extra; k++) {
            int a = regionNodes[rnd.next(0, (int)regionNodes.size() - 1)];
            int b = regionNodes[rnd.next(0, (int)regionNodes.size() - 1)];
            if (a == b) continue;
            ll len = rnd.next(1, 5);
            edges.push_back({a, b, len});
        }
    }

    int N = nextId - 1;
    int M = (int)edges.size();

    ll FAST = rnd.next(1, 4);
    ll mult = rnd.next(4, 8);
    ll SLOW = FAST * mult;
    if (SLOW > 40) SLOW = 40;
    if (SLOW <= FAST) SLOW = FAST + 1;

    printf("%d %d %d\n", N, M, K);
    for (auto &e : edges) printf("%d %d %lld\n", e.u, e.v, e.len);
    printf("%lld %lld\n", SLOW, FAST);
    printf("1\n");
    return 0;
}
