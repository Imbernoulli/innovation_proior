#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// -----------------------------------------------------------------------------
// "Four Looms, One Rope"  (generator)  family: vliw-chain-interleave-scheduling
//
// C independent strand-chains (each a straight-line dependency chain of length L)
// are woven by a machine with K=4 typed issue lanes (looms) per cycle. Each chain
// c>=1 additionally has TWO "splice" nodes (near 1/3 and 2/3 along the chain) that
// also consume the ROOT (position 0) of an EARLIER chain partner(c) < c -- i.e. a
// planted, sparse "coupling" edge that reuses an already-finished root much later.
// Finally all C chain-ends feed a single ROPE node (id = n), the unique required
// final output.
//
// PLANTED TRAP: if a solver runs all C chains roughly in lockstep (the natural
// list-scheduling behaviour once you decide to fill all 4 lanes every cycle), every
// chain's splice points land near the SAME global cycle, so ALL partner roots are
// "still needed later" simultaneously -> holding them all alive spikes the live
// register count to ~2C right when C is large, blowing a modest register cap R and
// forcing the scheduler to leave lanes idle (serialize) until something is freed.
// The fix (selective recompute as a scheduling primitive): discard a root right
// after ITS OWN chain's first use and re-materialize it (a flat, predecessor-free
// "recompute", since roots have no inputs) for each later splice -- this keeps the
// live set tiny and lets every cycle pack all 4 lanes.
//
// Format:
//   n m R K
//   type_1 .. type_n                (0..3, fixed lane for each node)
//   m lines: u v                    (edge u -> v, dependency; u < v always)
// Node n is the unique required final output ("the rope").
// -----------------------------------------------------------------------------

int main(int argc, char* argv[]) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);
    double f = (testId - 1) / 9.0;

    int C = (int)llround(3 + f * 23.0);      // 3 .. 26 chains
    int L = (int)llround(6 + f * 24.0);      // 6 .. 30 chain length
    if (L < 4) L = 4;
    int R = (int)llround(1.12 * C) + 6 + (int)llround(f * 4.0); // tight register cap

    vector<int> len(C);
    for (int c = 0; c < C; c++) {
        int jitter = rnd.next(-1, 1);
        len[c] = max(4, L + jitter);
    }

    vector<int> offset(C);
    int cur = 1;
    for (int c = 0; c < C; c++) { offset[c] = cur; cur += len[c]; }
    int n = cur; // nodes 1..n-1 are chain nodes, node n = rope
    int ropeId = n;

    auto nodeId = [&](int c, int p) { return offset[c] + p; };

    vector<pair<int,int>> edges;
    edges.reserve((size_t)n * 2 + C + 8);

    for (int c = 0; c < C; c++) {
        for (int p = 1; p < len[c]; p++) edges.push_back({nodeId(c, p - 1), nodeId(c, p)});
    }

    // splice ("coupling") edges: chain c (c>=1) reuses the ROOT of chain c-1
    // (single-hop reuse keeps the register-pressure window bounded and local).
    for (int c = 1; c < C; c++) {
        int partner = c - 1;
        int root = nodeId(partner, 0);
        int p1 = max(1, len[c] / 3);
        int p2 = max(p1 + 1, (2 * len[c]) / 3);
        if (p2 >= len[c] - 1) p2 = len[c] - 2;
        if (p2 < p1) p2 = p1;
        if (p1 < len[c] - 1) edges.push_back({root, nodeId(c, p1)});
        if (p2 < len[c] - 1 && p2 != p1) edges.push_back({root, nodeId(c, p2)});
    }

    // rope: fed by every chain's final node
    for (int c = 0; c < C; c++) edges.push_back({nodeId(c, len[c] - 1), ropeId});

    // shuffle edge print order (doesn't affect meaning; u<v already holds for all)
    for (int i = (int)edges.size() - 1; i > 0; i--) swap(edges[i], edges[rnd.next(0, i)]);

    vector<int> type(n + 1);
    for (int i = 1; i <= n; i++) type[i] = (i - 1) % 4;

    int m = (int)edges.size();
    int K = 4;
    printf("%d %d %d %d\n", n, m, R, K);
    for (int i = 1; i <= n; i++) printf("%d%c", type[i], i == n ? '\n' : ' ');
    for (auto &e : edges) printf("%d %d\n", e.first, e.second);
    return 0;
}
