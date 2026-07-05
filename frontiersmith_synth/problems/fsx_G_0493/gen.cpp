#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

// gen.cpp -- VLIW bundle-scheduling generator.
// testId is a difficulty/structure ladder (1 tiny .. 10 full envelope).  Produces layered
// dependence DAGs (edges always u<v -> program order is topological) with a mix of:
//   * PLANTED wide layers where good packing wins big,
//   * NEEDLE cases: one long high-latency chain buried in parallel filler,
//   * TRAP cases: a bottleneck functional-unit kind (cap=1) that greedy in-order packing
//     mis-serializes,
//   * envelope-filling large tests with tight resources.

int main(int argc, char** argv) {
    registerGen(argc, argv, 1);
    int tid = atoi(argv[1]);

    int n, widthAvg, maxPar, W, T, maxL;
    bool injectChain = false, bottleneck = false;

    switch (tid) {
        case 1:  n = 6;    widthAvg = 2;  maxPar = 2; W = 2; T = 2; maxL = 3;  break;
        case 2:  n = 45;   widthAvg = 9;  maxPar = 2; W = 4; T = 2; maxL = 3;  break; // wide packing (planted)
        case 3:  n = 160;  widthAvg = 3;  maxPar = 2; W = 4; T = 3; maxL = 8;  injectChain = true; break; // needle
        case 4:  n = 320;  widthAvg = 6;  maxPar = 3; W = 4; T = 3; maxL = 4;  bottleneck = true;  break; // trap
        case 5:  n = 550;  widthAvg = 5;  maxPar = 3; W = 4; T = 3; maxL = 5;  break;
        case 6:  n = 850;  widthAvg = 12; maxPar = 2; W = 6; T = 4; maxL = 4;  break; // dense wide
        case 7:  n = 1200; widthAvg = 40; maxPar = 1; W = 6; T = 3; maxL = 3;  break; // near-independent (bin packing)
        case 8:  n = 1800; widthAvg = 6;  maxPar = 3; W = 6; T = 4; maxL = 6;  bottleneck = true;  break;
        case 9:  n = 2400; widthAvg = 8;  maxPar = 3; W = 8; T = 4; maxL = 8;  injectChain = true; bottleneck = true; break;
        default: n = 3000; widthAvg = 8;  maxPar = 2; W = 8; T = 4; maxL = 8;  injectChain = true; break; // full envelope
    }

    // ---- resource parameters ----
    vector<int> cap(T + 1), rp(T + 1);
    int maxrp = 0;
    for (int t = 1; t <= T; t++) { rp[t] = rnd.next(1, 3); maxrp = max(maxrp, rp[t]); }
    for (int t = 1; t <= T; t++) cap[t] = rnd.next(max(1, W / 2), W);
    if (bottleneck) cap[T] = 1;                 // scarce unit kind
    int P = min(8, max(maxrp, rnd.next(W, W + 2)));   // read-port budget; couples the kinds

    // ---- layers: node id increasing => layer non-decreasing (keeps edges u<v) ----
    auto layerOf = [&](int id) { return (id - 1) / widthAvg; };
    int Ln = layerOf(n) + 1;

    // ---- op kinds and latencies ----
    vector<int> type(n + 1), L(n + 1);
    for (int i = 1; i <= n; i++) {
        if (bottleneck && rnd.next(0, 99) < 30) type[i] = T;       // load the scarce kind
        else type[i] = rnd.next(1, T);
        L[i] = rnd.next(1, maxL);
    }

    // ---- edges ----
    vector<pair<int,int>> edges;
    vector<set<int>> pset(n + 1);
    const int EDGE_CAP = 7500;
    for (int v = 1; v <= n; v++) {
        int lv = layerOf(v);
        if (lv == 0) continue;
        int p = rnd.next(1, maxPar);
        for (int k = 0; k < p; k++) {
            if ((int)edges.size() >= EDGE_CAP) break;
            int el = rnd.next(0, lv - 1);                 // some earlier layer
            int lo = el * widthAvg + 1;
            int hi = min(n, (el + 1) * widthAvg);
            if (lo > hi) continue;
            int u = rnd.next(lo, hi);
            if (u >= v) continue;
            if (pset[v].count(u)) continue;
            pset[v].insert(u);
            edges.push_back(make_pair(u, v));
        }
    }

    // ---- needle: a long high-latency chain threaded through the layers ----
    if (injectChain) {
        int prev = -1;
        for (int l = 0; l < Ln; l++) {
            int node = l * widthAvg + 1;
            if (node > n) break;
            L[node] = maxL;                                // gate the whole schedule
            if (prev != -1 && (int)edges.size() < EDGE_CAP && !pset[node].count(prev)) {
                pset[node].insert(prev);
                edges.push_back(make_pair(prev, node));
            }
            prev = node;
        }
    }

    if (edges.empty() && n >= 2) edges.push_back(make_pair(1, 2));  // guarantee m >= 1
    int m = (int)edges.size();

    // ---- emit ----
    printf("%d %d %d %d %d\n", n, m, W, T, P);
    for (int t = 1; t <= T; t++) printf("%d%c", cap[t], t == T ? '\n' : ' ');
    for (int t = 1; t <= T; t++) printf("%d%c", rp[t], t == T ? '\n' : ' ');
    for (int i = 1; i <= n; i++) printf("%d %d\n", type[i], L[i]);
    for (int j = 0; j < m; j++) printf("%d %d\n", edges[j].first, edges[j].second);

    return 0;
}
