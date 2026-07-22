#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// ============================================================================
// Family: audit-cutset-inspector  (theme: canal patrols checking cargo manifests)
//
// A hub chain H_0 -> H_1 -> ... -> H_{L-1} is the canal's spine. Between
// consecutive hubs sits a CORRIDOR BUNDLE: a set of parallel, equal-cost canal
// channels. A patrol placed on some (but not all) channels of a bundle is
// FREE to evade -- smugglers just slide over to an unpatrolled channel at the
// exact same cost (evasion-rerouting), so a partial cut of a bundle realizes
// ZERO extra interception no matter how large a fraction of the bundle it is.
// Only a FULLY closed bundle (every channel patrolled) forces a capture. This
// is the innovation hook: interdiction value lives in the CUT (the whole
// bundle), not in individual edges -- an edge-marginal-value greedy cannot
// see the all-or-nothing threshold and wastes budget spreading one patrol
// across many bundles instead of fully closing a few.
//
// Three bundle classes are planted along the chain (mechanism composition):
//   - CHOKEPOINT (size 1): a true bottleneck, no parallel channel at all.
//     Patrolling its one edge is unconditionally mandatory-and-free (cheap,
//     always pays off) -- "inspection-placement" is trivial here.
//   - MEDIUM (size 2..4, no bypass): fully closing it is affordable and
//     forces genuine capture -- rewards a bundle-aware knapsack.
//   - DECOY (size 5..8, PLUS a pricier bypass channel in the same corridor):
//     looks the most "valuable" under naive per-edge accounting (its primary
//     channel carries just as much static traffic as anything else), but
//     fully closing even its k normal channels still lets smugglers slip out
//     via the hidden, unpatrolled bypass (evasion-rerouting) -- so it is
//     essentially never worth its high channel-count price. A trap for any
//     size-blind greedy.
//
// Node 0 is always H_0, and bundle position 0 is ALWAYS a chokepoint (a
// single edge, printed as edge index 0) -- this is the checker's calibration
// anchor.
//
// Flows: MAIN flows span the whole spine (H_0 -> H_last); one LOCAL flow is
// dedicated to every single bundle (spans exactly that bundle, so its value
// is only ever captured there, decoupling per-bundle payoff from ordering
// effects of "first patrolled edge wins"); a few PARTIAL flows add texture;
// a disjoint NOISE cluster with its own small flows adds filler.
// ============================================================================

static const ll PEN = 2000000; // fixed, restated in statement.txt

struct Edge { int u, v; ll cost, cap; };

int main(int argc, char** argv) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);

    int numChokeTab[11]  = {0,2,2,3,3,3,4,4,4,5,5};
    int numMediumTab[11] = {0,1,2,2,3,3,3,4,4,4,5};
    int numDecoyTab[11]  = {0,1,1,1,2,2,2,3,3,3,3};

    int numChoke  = numChokeTab[testId];
    int numMedium = numMediumTab[testId];
    int numDecoy  = numDecoyTab[testId];
    int numBundles = numChoke + numMedium + numDecoy;

    // ---- bundle type sequence: position 0 forced CHOKE, rest shuffled ----
    vector<int> types; // 0=choke 1=medium 2=decoy
    types.push_back(0);
    vector<int> rest;
    for (int i = 0; i < numChoke - 1; i++) rest.push_back(0);
    for (int i = 0; i < numMedium; i++) rest.push_back(1);
    for (int i = 0; i < numDecoy; i++) rest.push_back(2);
    shuffle(rest.begin(), rest.end());
    for (int x : rest) types.push_back(x);

    int L = numBundles + 1; // number of hubs H_0..H_{L-1}

    vector<Edge> edges;
    vector<vector<int>> bundleEdgeIdx(numBundles); // for reference (not printed)

    // bundle 0 MUST be printed first (edge index 0) -- checker anchor
    for (int b = 0; b < numBundles; b++) {
        int u = b, v = b + 1;
        ll bcost = rnd.next(10, 30);
        int size;
        if (types[b] == 0) size = 1;
        else if (types[b] == 1) size = rnd.next(2, 4);
        else size = rnd.next(5, 8);
        for (int j = 0; j < size; j++) {
            ll cap = rnd.next(15, 60);
            bundleEdgeIdx[b].push_back((int)edges.size());
            edges.push_back({u, v, bcost, cap});
        }
        if (types[b] == 2) { // decoy: add a pricier bypass channel
            ll bypassCost = bcost + rnd.next(80, 150);
            ll bypassCap = rnd.next(15, 60);
            bundleEdgeIdx[b].push_back((int)edges.size());
            edges.push_back({u, v, bypassCost, bypassCap});
        }
    }

    // ---- noise cluster: disjoint node ids, own small edges/flows ----
    int numNoiseNodes = 4 + testId / 2;
    int noiseBase = L;
    int numNoiseEdges = 0;
    vector<Edge> noiseEdges;
    // simple chain to guarantee connectivity, then extra random edges
    for (int i = 0; i + 1 < numNoiseNodes; i++) {
        ll cost = rnd.next(1, 10), cap = rnd.next(5, 20);
        noiseEdges.push_back({noiseBase + i, noiseBase + i + 1, cost, cap});
    }
    int extraNoise = numNoiseNodes; // extra random edges for variety
    for (int i = 0; i < extraNoise; i++) {
        int a = rnd.next(0, numNoiseNodes - 1);
        int c = rnd.next(0, numNoiseNodes - 1);
        if (a == c) continue;
        ll cost = rnd.next(1, 10), cap = rnd.next(5, 20);
        noiseEdges.push_back({noiseBase + a, noiseBase + c, cost, cap});
    }
    for (auto& e : noiseEdges) edges.push_back(e);
    int n = L + numNoiseNodes;

    // ---- flows ----
    struct Flow { int s, t; ll vol, val; };
    vector<Flow> flows;

    int numMain = rnd.next(2, 4);
    for (int i = 0; i < numMain; i++) {
        ll val = rnd.next(40, 90), vol = rnd.next(60, 160);
        flows.push_back({0, L - 1, vol, val});
    }
    // one dedicated LOCAL flow per bundle -- MEDIUM bundles get a richer
    // local flow (val/vol scaled up) so that fully closing an affordable
    // medium bundle is clearly worth more than the free chokepoints, which
    // only a bundle-aware strategy can realize (a per-edge greedy still
    // only ever lands ONE channel on a multi-channel bundle, capturing none
    // of this bonus no matter how it is priced).
    for (int b = 0; b < numBundles; b++) {
        ll val, vol;
        if (types[b] == 1) { val = rnd.next(60, 130); vol = rnd.next(80, 180); }
        else { val = rnd.next(20, 60); vol = rnd.next(30, 90); }
        flows.push_back({b, b + 1, vol, val});
    }
    // partial flows spanning a random sub-range of the spine
    int numPartial = 1 + testId / 3;
    for (int i = 0; i < numPartial; i++) {
        int a = rnd.next(0, L - 2);
        int bnd = rnd.next(a + 1, L - 1);
        ll val = rnd.next(15, 50), vol = rnd.next(20, 70);
        flows.push_back({a, bnd, vol, val});
    }
    // noise flows
    int numNoiseFlows = 2 + testId / 2;
    for (int i = 0; i < numNoiseFlows; i++) {
        int a = rnd.next(0, numNoiseNodes - 1);
        int b2 = rnd.next(0, numNoiseNodes - 1);
        if (a == b2) b2 = (b2 + 1) % numNoiseNodes;
        int s = noiseBase + min(a, b2), t = noiseBase + max(a, b2);
        ll val = rnd.next(5, 20), vol = rnd.next(5, 20);
        flows.push_back({s, t, vol, val});
    }

    int m = (int)edges.size();
    int K = (int)flows.size();

    // ---- budget: covers every chokepoint for free, PLUS exactly enough to
    //      fully close the (numMedium-1) SMALLEST medium bundles (all of
    //      them if there is only one) -- guarantees a real, affordable
    //      bundle-closure opportunity on every test while still forcing a
    //      knapsack choice once there are >= 2 mediums, and never enough to
    //      touch a decoy at all. ----
    vector<ll> medSizes;
    for (int b = 0; b < numBundles; b++)
        if (types[b] == 1) medSizes.push_back((ll)bundleEdgeIdx[b].size());
    sort(medSizes.begin(), medSizes.end());
    int fundCount = (int)medSizes.size();
    if (fundCount > 1) fundCount -= 1; // leave the single largest medium out
    ll fundedMedium = 0;
    for (int i = 0; i < fundCount; i++) fundedMedium += medSizes[i];
    ll Bbudget = numChoke + fundedMedium;
    if (Bbudget < numChoke + 1) Bbudget = numChoke + 1;
    if (Bbudget > m) Bbudget = m;

    // ---- emit ----
    string out;
    out.reserve((size_t)(m + K) * 24 + 64);
    char buf[128];
    int len = sprintf(buf, "%d %d %d %lld\n", n, m, K, Bbudget);
    out.append(buf, len);
    for (auto& e : edges) {
        len = sprintf(buf, "%d %d %lld %lld\n", e.u, e.v, e.cost, e.cap);
        out.append(buf, len);
    }
    for (auto& f : flows) {
        len = sprintf(buf, "%d %d %lld %lld\n", f.s, f.t, f.vol, f.val);
        out.append(buf, len);
    }
    fwrite(out.data(), 1, out.size(), stdout);
    return 0;
}
