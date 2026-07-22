#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

// gen.cpp -- Refreeze Window: Convoy Wave Scheduling
//
// Builds a channel graph (nodes = ports, edges = channels with thickness cap C,
// regrow rate r, fuel fee f, length/travel-time L) plus K convoys (origin, destination,
// ready time). Every test is CONNECTED.
//
// Structure: a random "background" graph provides general connectivity + noise, plus
// zero or more planted SUPERCLUSTERs. A supercluster has TWO parallel corridors, each a
// pair of anchor ports S1[j]-S2[j] (j=0,1) joined by one expensive MAIN[j] channel (large
// cap -> costly to break fresh, the thickness^1.5 term dominates). Each of its n convoys
// has a private origin o_g and destination d_g; g's "home" corridor is g%2. o_g has a CHEAP
// entry edge to its home S1 and a pricier CROSS entry edge to the other corridor's S1
// (symmetrically, d_g has a cheap exit from its home S2 and a pricier cross exit from the
// other S2). d_g and the S2 ports are reachable *only* through a MAIN edge (no background
// link on that side), so a corridor's MAIN is a required cut-edge for any convoy that wants
// to use it -- there is no bypass, and (unlike a shared direct edge) no sibling convoy's
// private edges can ever substitute for it either, since o_g/d_g have no edges to each
// other at all.
//
// Numbers are tuned so a convoy's own-home corridor is always its statically cheapest
// choice (so the "obvious" independent-shortest-path approach always sends every convoy to
// its home corridor and never proactively shares) -- but a *coordinated* plan that detours
// every convoy in the supercluster onto ONE shared corridor (paying the small cross-edge
// premium for whichever half is "foreign" to it) saves an entire second corridor's fresh
// break cost, which is far larger than the total detour premium once n is large enough.
// This is the trap: a per-convoy-independent shortest path never discovers this, since
// consolidating always looks locally worse. "tight" superclusters have ready times close
// enough to fall inside both MAINs' regrow windows (so warm reuse actually happens);
// "spread" ones have gaps larger than the window (so consolidating is a false lead -- ice
// fully regrows between crossings and detouring only wastes fuel).
//
// testId is a difficulty ladder: test 1 is a tiny hand-fixed instance (no randomness) used
// verbatim in the statement's worked example; tests 2..10 grow in node/edge/convoy count
// and add more hub clusters (tight, spread, needle-in-haystack mixes), filling the stated
// constraint envelope by test 10.

struct Edge { int u, v, L, C, r, f; };

static vector<Edge> edges;
static int N = 0;

static int addNode() { return ++N; }
static int addEdge(int u, int v, int L, int C, int r, int f) {
    edges.push_back({u, v, L, C, r, f});
    return (int)edges.size();
}

struct Convoy { int o, d, ready; };
static vector<Convoy> convoys;

struct HubSpec { int n; bool tight; };

int main(int argc, char* argv[]) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);
    int idx = min(max(testId, 1), 10);

    if (idx == 1) {
        // ---- hand-fixed tiny example (matches statement.txt exactly) ----
        // Nodes: 1,2,3,4.  Edges: e1=(1,2) e2=(2,3) e3=(4,2).
        N = 4;
        addEdge(1, 2, 2, 10, 2, 1);   // e1
        addEdge(2, 3, 3, 8, 1, 2);    // e2 (the shared channel)
        addEdge(4, 2, 1, 6, 3, 0);    // e3
        convoys.push_back({1, 3, 0}); // convoy 1: 1 -> 3, ready 0
        convoys.push_back({4, 3, 3}); // convoy 2: 4 -> 3, ready 3 (shares e2 soon after)
        printf("%d %d %d\n", N, (int)edges.size(), (int)convoys.size());
        for (auto& e : edges) printf("%d %d %d %d %d %d\n", e.u, e.v, e.L, e.C, e.r, e.f);
        for (auto& c : convoys) printf("%d %d %d\n", c.o, c.d, c.ready);
        return 0;
    }

    // ---- parameter table per test index (2..10) ----
    int nBackground[11]   = {0, 0, 6, 8, 9, 11, 11, 14, 16, 8, 6};
    int extraBgEdges[11]  = {0, 0, 2, 2, 2, 3, 3, 4, 5, 8, 150};
    int bgConvoys[11]     = {0, 0, 4, 3, 3, 4, 4, 6, 12, 16, 90};

    vector<vector<HubSpec>> hubTable(11);
    hubTable[3] = {{2, true}};
    hubTable[4] = {{5, true}};
    hubTable[5] = {{6, true}, {2, true}};
    hubTable[6] = {{6, false}};
    hubTable[7] = {{5, true}, {2, true}, {6, true}};
    hubTable[8] = {{8, true}, {1, true}, {1, true}, {2, true}, {2, true}};
    hubTable[9] = {{5, true}, {6, true}, {2, true}, {4, true}, {1, true}};
    hubTable[10] = {{5, true}, {4, true}, {2, true}, {2, true}, {1, true}, {5, true}};

    int nb = nBackground[idx];

    // ---- background: spanning tree over nb nodes + extra chord edges ----
    vector<int> bgNodes;
    for (int i = 0; i < nb; i++) bgNodes.push_back(addNode());
    vector<int> perm = bgNodes;
    shuffle(perm.begin(), perm.end());
    for (int i = 1; i < nb; i++) {
        int a = perm[rnd.next(0, i - 1)];
        int b = perm[i];
        addEdge(a, b, rnd.next(1, 8), rnd.next(10, 40), rnd.next(2, 8), rnd.next(1, 10));
    }
    for (int i = 0; i < extraBgEdges[idx] && nb >= 2; i++) {
        int a = bgNodes[rnd.next(0, nb - 1)];
        int b = bgNodes[rnd.next(0, nb - 1)];
        if (a == b) continue;
        addEdge(a, b, rnd.next(1, 8), rnd.next(10, 40), rnd.next(2, 8), rnd.next(1, 10));
    }

    // ---- background convoys (noise / needle cover) ----
    int baseReady = 0;
    for (int i = 0; i < bgConvoys[idx] && nb >= 2; i++) {
        int o = bgNodes[rnd.next(0, nb - 1)];
        int d = bgNodes[rnd.next(0, nb - 1)];
        if (o == d) { i--; continue; }
        convoys.push_back({o, d, rnd.next(0, 4000)});
    }

    // ---- hop-count trap chain: separates the BFS-fewest-hop baseline (trivial) from a
    // real static-cost shortest path (greedy). Each segment offers a 1-hop SHORTCUT edge
    // (few hops, but a large cap -> expensive even virgin) versus a 2-hop DETOUR through a
    // private midpoint (more hops, but each edge is cheap) -- fewest-hop BFS always takes
    // the shortcut, static-cost Dijkstra always takes the detour, and the gap compounds
    // multiplicatively across chained segments.
    int chainLen[11] = {0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2};
    int cl = chainLen[idx];
    if (cl > 0 && nb >= 1) {
        int chainStart = addNode();
        addEdge(chainStart, bgNodes[rnd.next(0, nb - 1)], rnd.next(1, 3), rnd.next(10, 20), rnd.next(3, 8), rnd.next(1, 5));
        int prev = chainStart;
        for (int s = 0; s < cl; s++) {
            int mid = addNode();
            int nxt = addNode();
            addEdge(prev, nxt, 1, rnd.next(45, 75), rnd.next(2, 6), rnd.next(3, 8));   // shortcut (1 hop, pricey)
            addEdge(prev, mid, 1, rnd.next(8, 14), rnd.next(3, 10), rnd.next(0, 2));   // detour hop 1 (cheap)
            addEdge(mid, nxt, 1, rnd.next(8, 14), rnd.next(3, 10), rnd.next(0, 2));    // detour hop 2 (cheap)
            prev = nxt;
        }
        int chainEnd = prev;
        int chainConvoys = 3;
        for (int t = 0; t < chainConvoys; t++)
            convoys.push_back({chainStart, chainEnd, rnd.next(0, 4000)});
    }

    // ---- superclusters (two parallel corridors each) ----
    int clusterBase = 500;
    for (auto& hs : hubTable[idx]) {
        int S1[2], S2[2], MAINe[2];
        // Both corridors' MAIN channels are drawn with the SAME (L,C,r,f): this removes
        // any *random* static-cost asymmetry between the two corridors, so a per-convoy
        // static shortest path always prefers its own home corridor (by entry/exit cost
        // alone, which is a genuine per-convoy asymmetry) rather than accidentally
        // "consolidating" everyone onto whichever corridor happened to roll a cheaper cap.
        int Lm = rnd.next(3, 6), Cm = rnd.next(200, 260), rm = rnd.next(1, 3), fm = rnd.next(15, 40);
        for (int j = 0; j < 2; j++) {
            S1[j] = addNode(); S2[j] = addNode();
            if (nb >= 1)
                addEdge(S1[j], bgNodes[rnd.next(0, nb - 1)], rnd.next(1, 5), rnd.next(10, 30), rnd.next(2, 8), rnd.next(1, 8));
            MAINe[j] = addEdge(S1[j], S2[j], Lm, Cm, rm, fm);
        }
        int window = Cm / max(1, rm);

        int clusterReadyBase = clusterBase;
        clusterBase += 1500;
        for (int g = 0; g < hs.n; g++) {
            int home = g % 2, other = 1 - home;
            int o = addNode(), d = addNode();
            addEdge(o, S1[home],  rnd.next(1, 2), rnd.next(4, 8),   rnd.next(3, 10), rnd.next(0, 2));   // home entry (cheap)
            addEdge(o, S1[other], rnd.next(1, 2), rnd.next(10, 16), rnd.next(3, 10), rnd.next(2, 6));   // cross entry (detour)
            addEdge(S2[home],  d, rnd.next(1, 2), rnd.next(4, 8),   rnd.next(3, 10), rnd.next(0, 2));   // home exit (cheap)
            addEdge(S2[other], d, rnd.next(1, 2), rnd.next(10, 16), rnd.next(3, 10), rnd.next(2, 6));   // cross exit (detour)
            int ready;
            if (hs.tight) ready = clusterReadyBase + rnd.next(0, max(1, window / 6));
            else          ready = clusterReadyBase + rnd.next(0, window * 6 + 300);
            convoys.push_back({o, d, ready});
        }
        (void)MAINe;
    }

    printf("%d %d %d\n", N, (int)edges.size(), (int)convoys.size());
    for (auto& e : edges) printf("%d %d %d %d %d %d\n", e.u, e.v, e.L, e.C, e.r, e.f);
    for (auto& c : convoys) printf("%d %d %d\n", c.o, c.d, c.ready);
    return 0;
}
