#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// -----------------------------------------------------------------------------
// Checker / scorer for "Four Looms, One Rope"  (family: vliw-chain-interleave-scheduling)
//
// Input:  n m R K
//         type_1 .. type_n              (0..3, fixed issue-lane type per node)
//         m lines: u v                  (dependency edge u -> v, u < v)
// Node n is "the rope", the unique required final output.
//
// Output (participant): T, then T cycle-lines. Cycle line:
//   p_t  (op_1 node_1) .. (op_p node_p)   d_t  disc_1 .. disc_d
// op in {0=compute (first time only), 1=recompute (revive a discarded node)}.
// Both compute and recompute require ALL predecessors of the node to be
// currently LIVE (computed/recomputed at an EARLIER cycle, not yet discarded).
// A node's own result becomes usable starting the NEXT cycle. At most one
// instruction per lane-type per cycle (K lane-types, K given, always 4 here).
// Discards are free bookkeeping, applied after this cycle's instructions.
// After the cycle's instructions+discards, the number of currently-live nodes
// must not exceed R (the register file). Node n must be live once the last
// printed cycle ends.
//
// Objective (MIN): F = T (number of cycles used).
// Baseline B (an actual feasible trivial construction): process every node
// exactly once, one instruction per cycle (only ever 1 of the 4 lanes used,
// never recomputing, discarding a node only once ALL its uses have fired) ->
// this always succeeds in exactly T=n cycles, so B = n.
// Score: sc = min(1000, 100*B/max(1,F)); ratio = sc/1000.
// -----------------------------------------------------------------------------

int main(int argc, char* argv[]) {
    registerTestlibCmd(argc, argv);

    int n = inf.readInt();
    int m = inf.readInt();
    int R = inf.readInt();
    int K = inf.readInt();
    vector<int> type(n + 1);
    for (int i = 1; i <= n; i++) type[i] = inf.readInt(0, K - 1, "type");
    vector<vector<int>> preds(n + 1);
    for (int i = 0; i < m; i++) {
        int u = inf.readInt(1, n, "u");
        int v = inf.readInt(1, n, "v");
        preds[v].push_back(u);
    }

    ll B = n;

    int Tcap = 10 * n + 200;
    int T = ouf.readInt(1, Tcap, "T");

    vector<char> live(n + 1, 0), computedEver(n + 1, 0);
    vector<int> avail(n + 1, -1);
    long long liveCount = 0;

    for (int t = 1; t <= T; t++) {
        int p = ouf.readInt(0, K, "p_t");
        vector<char> laneUsed(K, 0);
        vector<int> committedThisCycle;
        vector<int> usedPredsThisCycle;

        for (int j = 0; j < p; j++) {
            int op = ouf.readInt(0, 1, "op");
            int node = ouf.readInt(1, n, "node");
            int ty = type[node];
            if (laneUsed[ty]) quitf(_wa, "cycle %d: lane type %d used twice", t, ty);
            laneUsed[ty] = 1;

            if (op == 0) {
                if (computedEver[node]) quitf(_wa, "cycle %d: node %d computed twice (op=0)", t, node);
            } else {
                if (!computedEver[node]) quitf(_wa, "cycle %d: recompute of never-computed node %d", t, node);
                if (live[node]) quitf(_wa, "cycle %d: recompute of already-live node %d", t, node);
            }
            for (int u : preds[node]) {
                if (!(live[u] && avail[u] <= t))
                    quitf(_wa, "cycle %d: node %d issued but predecessor %d is not available", t, node, u);
                usedPredsThisCycle.push_back(u);
            }
            computedEver[node] = 1;
            live[node] = 1;
            avail[node] = t + 1;
            liveCount++;
            committedThisCycle.push_back(node);
        }

        int d = ouf.readInt(0, n, "d_t");
        for (int j = 0; j < d; j++) {
            int node = ouf.readInt(1, n, "discard_node");
            if (!live[node]) quitf(_wa, "cycle %d: discard of non-live node %d", t, node);
            live[node] = 0;
            liveCount--;
        }

        if (liveCount > R) quitf(_wa, "cycle %d: register cap exceeded (%lld > %d)", t, liveCount, R);
        if (liveCount < 0) quitf(_wa, "cycle %d: internal negative live count", t); // defensive
    }

    if (!ouf.seekEof()) quitf(_wa, "trailing output tokens after cycle %d", T);
    if (!live[n]) quitf(_wa, "the rope (node %d) is not live when the schedule ends", n);

    ll F = T;
    double sc = min(1000.0, 100.0 * (double)B / (double)max((ll)1, F));
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}
