#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// -----------------------------------------------------------------------------
// "Shatter the Syndicate"  (generator)   family: collective-influence-immunization
//
// Emits a contact network of N people. Each edge carries a fixed integer
// robustness threshold t in [1,999999]; the network is stress-tested under S
// contagion scenarios, each a strength p_s in [1,999999] -- edge j is ACTIVE in
// scenario s iff t_j < p_s (a fixed, deterministic bond-percolation coupling: no
// randomness at judge time). Every person also has an immunization cost c_i.
// The solver picks a set R with sum of costs <= K (a spending budget, not a head
// count) to immunize (delete); the objective is the K-budget-limited,
// scenario-averaged size of the largest still-connected group.
//
// PLANTED STRUCTURE (the trap): the population is built from several dense,
// locally-redundant COMMUNITIES (either plain dense random clusters or
// near-clique HUB CORES with pendant leaves) stitched together by a handful of
// designated BRIDGE PEOPLE -- degree-2 cut vertices whose only two contacts are
// one anchor in each of the communities they connect, wired with a very low
// threshold so they are active in virtually every scenario. Communities are
// internally dense (mean degree 5-9) so they stay supercritical (one big lump)
// under most scenarios: the ONLY way to shrink the network-wide giant lump is to
// sever the bridges. Bridge people always have the SMALLEST degree in the whole
// graph (degree 2), while community hubs/members have degree far above the
// budget's effective "cut count" -- so ranking people by raw degree burns the
// entire budget deep inside a single redundant community (which barely dents the
// lump, since many alternate internal paths survive) and never reaches the
// bridges. The insight is COLLECTIVE INFLUENCE over a radius: a bridge's
// radius-ell frontier lands inside a whole *neighbouring* community, giving it
// an outsized CI score even though its own degree is minimal.
// -----------------------------------------------------------------------------

static int N = 0;
static vector<array<int,3>> edges;              // u, v, threshold (1-based ids)
static unordered_set<ll> eset;

static ll ekey(int a, int b){ if (a > b) swap(a, b); return (ll)a * 2000000LL + b; }

static void addEdgeTh(int a, int b, int t){
    if (a == b) return;
    ll k = ekey(a, b);
    if (eset.count(k)) return;
    eset.insert(k);
    edges.push_back({a, b, t});
}
static void addEdge(int a, int b){ addEdgeTh(a, b, rnd.next(1, 999999)); }
static void addBridgeEdge(int a, int b){ addEdgeTh(a, b, rnd.next(1, 40)); }   // near-always active

static int newNodes(int cnt){ int start = N + 1; N += cnt; return start; }

// dense ER-ish cluster on [start,start+sz-1], target mean internal degree meanDeg.
// returns a "safe" anchor node (near-median degree, not a hub) for bridging.
static int buildCluster(int start, int sz, double meanDeg){
    long long target = (long long)llround(sz * meanDeg / 2.0);
    long long guard = target * 30 + 200;
    long long made = 0;
    while (made < target && guard-- > 0){
        int a = start + rnd.next(0, sz - 1);
        int b = start + rnd.next(0, sz - 1);
        if (a == b) continue;
        ll k = ekey(a, b);
        if (eset.count(k)) continue;
        addEdge(a, b);
        made++;
    }
    return start + sz / 2;
}

// near-clique hub core: hubCount hubs (near-clique, prob hubProb) + leafCount
// leaves each attaching to 1-3 random hubs. Returns a LEAF id (low degree, safe
// bridge anchor) so bridging never touches the high-degree hub set.
static int buildHubCore(int hubCount, int leafCount, double hubProb){
    int hubStart = newNodes(hubCount);
    for (int i = 0; i < hubCount; i++)
        for (int j = i + 1; j < hubCount; j++)
            if (rnd.next(0.0, 1.0) < hubProb) addEdge(hubStart + i, hubStart + j);
    int leafStart = newNodes(leafCount);
    for (int i = 0; i < leafCount; i++){
        int att = 1 + rnd.next(0, 2);
        for (int a = 0; a < att; a++){
            int hub = hubStart + rnd.next(0, hubCount - 1);
            addEdge(leafStart + i, hub);
        }
    }
    return leafStart;              // low-degree leaf: safe anchor
}

static int addBridgeNode(int anchorA, int anchorB){
    int id = newNodes(1);
    addBridgeEdge(id, anchorA);
    addBridgeEdge(id, anchorB);
    return id;
}

int main(int argc, char* argv[]){
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);

    int ELL = 2, K = 6, S = 3;

    if (testId == 1){                                   // tiny sanity trap
        int a = buildCluster(newNodes(10), 10, 4.5);
        int b = buildCluster(newNodes(10), 10, 4.5);
        addBridgeNode(a, b);
        ELL = 2; K = 3; S = 3;
    } else if (testId == 2){                             // small 2-cluster bridge trap
        int a = buildCluster(newNodes(25), 25, 6.5);
        int b = buildCluster(newNodes(25), 25, 6.5);
        addBridgeNode(a, b);
        ELL = 2; K = 11; S = 3;
    } else if (testId == 3){                             // small hub-redundant trap
        int a = buildHubCore(6, 14, 0.85);
        int b = buildHubCore(6, 14, 0.85);
        addBridgeNode(a, b);
        ELL = 2; K = 7; S = 4;
    } else if (testId == 4){                             // medium plain ER, no trap (sanity)
        buildCluster(newNodes(180), 180, 6.0);
        ELL = 2; K = 110; S = 4;
    } else if (testId == 5){                             // medium 4-cluster chain trap
        int c0 = buildCluster(newNodes(45), 45, 7.0);
        int c1 = buildCluster(newNodes(45), 45, 7.0);
        int c2 = buildCluster(newNodes(45), 45, 7.0);
        int c3 = buildCluster(newNodes(45), 45, 7.0);
        addBridgeNode(c0, c1); addBridgeNode(c1, c2); addBridgeNode(c2, c3);
        ELL = 3; K = 50; S = 5;
    } else if (testId == 6){                             // medium hub-redundant trap
        // hub count is MUCH larger than the budget's removal count, so a near-
        // clique stays whp connected after losing any budget-sized subset of hubs.
        int a = buildHubCore(40, 15, 0.8);
        int b = buildHubCore(40, 15, 0.8);
        addBridgeNode(a, b);
        ELL = 2; K = 24; S = 5;
    } else if (testId == 7){                             // large plain ER, no trap (sanity)
        buildCluster(newNodes(700), 700, 7.0);
        ELL = 2; K = 380; S = 5;
    } else if (testId == 8){                             // large STAR needle trap
        int C = 8;
        int centerSize = 200;
        int centerStart = newNodes(centerSize);
        buildCluster(centerStart, centerSize, 6.5);
        for (int i = 0; i < C; i++){
            int satAnchor = buildCluster(newNodes(90), 90, 6.5);
            int centerAnchor = centerStart + (i * (centerSize / C)) % centerSize;
            addBridgeNode(centerAnchor, satAnchor);
        }
        ELL = 3; K = 30; S = 6;
    } else if (testId == 9){                             // large hub-redundant chain trap
        int a = buildHubCore(45, 20, 0.75);
        int b = buildHubCore(45, 20, 0.75);
        int c = buildHubCore(45, 20, 0.75);
        addBridgeNode(a, b); addBridgeNode(b, c);
        ELL = 3; K = 48; S = 6;
    } else {                                              // testId 10: largest combined trap
        int C = 6;
        int centerSize = 300;
        int centerStart = newNodes(centerSize);
        buildCluster(centerStart, centerSize, 7.0);
        for (int i = 0; i < C; i++){
            int satAnchor = buildCluster(newNodes(500), 500, 7.0);
            int centerAnchor = centerStart + (i * (centerSize / C)) % centerSize;
            addBridgeNode(centerAnchor, satAnchor);
        }
        int h1 = buildHubCore(45, 30, 0.75);
        int h2 = buildHubCore(45, 30, 0.75);
        addBridgeNode(h1, h2);
        int hAnchor = h1;
        int satAnchor2 = centerStart;                      // hub wing also chained to the center
        addBridgeNode(hAnchor, satAnchor2);
        ELL = 3; K = 140; S = 6;
    }

    int n = N;
    int m = (int)edges.size();

    // ---- per-node immunization cost ----
    vector<int> cost(n + 1);
    for (int i = 1; i <= n; i++) cost[i] = 1 + rnd.next(0, 2);

    // ---- S scenario strengths, spread mid-to-high so communities stay supercritical
    //      and the bridges remain the dominant lever (adds regime variety too). ----
    vector<int> ps(S);
    for (int s = 0; s < S; s++){
        int lo = 450000, hi = 950000;
        int anchor = lo + (int)((ll)(hi - lo) * s / max(1, S - 1));
        int jit = rnd.next(-15000, 15000);
        int v = anchor + jit;
        if (v < 20000) v = 20000;
        if (v > 999000) v = 999000;
        ps[s] = v;
    }

    printf("%d %d %d %d %d\n", n, m, ELL, K, S);
    for (auto &e : edges) printf("%d %d %d\n", e[0], e[1], e[2]);
    for (int i = 1; i <= n; i++) printf("%d%c", cost[i], i < n ? ' ' : '\n');
    for (int s = 0; s < S; s++) printf("%d%c", ps[s], s + 1 < S ? ' ' : '\n');
    return 0;
}
