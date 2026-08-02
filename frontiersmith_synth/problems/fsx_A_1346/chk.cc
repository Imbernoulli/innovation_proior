#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

// chk.cc -- scorer for "Corner Patrol: Dismantling a Graph to Trap an Evader" (MINIMIZE).
//
// Participant prints exactly n-1 lines "v k g_1..g_k": eliminate v (once each), guarded
// by k distinct still-live vertices g_1..g_k (!=v). A step is legal iff every currently
// live neighbor of v, and v itself, lies in the union of the g_j's *live* closed
// neighborhoods. Cost of a step is k^2; F = total cost. Any violated line scores 0.
//
// Internal baseline B: eliminate in a FIXED order (deepest-first under a BFS tree from
// vertex 0, ties by index, root last/never eliminated), guarding every vertex with ALL
// of its currently-live neighbors (always valid: a vertex's own neighbor set trivially
// covers itself, since each live neighbor u has u in its own closed neighborhood, and u
// is adjacent to v so v in N[u] too). This never fails, since every non-root vertex has
// (by BFS shortest-path property) at least one neighbor at depth-1, which under
// deepest-first elimination is still live. B > 0 always (n>=2 forces >=1 step, and the
// first eliminated vertex under deepest-first has degree >=1: it is not the root, and
// its BFS-parent is live).
//
// ratio = min(1, 0.1 * B / max(1,F))  -> matching the baseline exactly scores 0.1.

int n; long long m;
vector<vector<int>> adj;

vector<int> bfsDeepestFirstOrder() {
    vector<int> depth(n, -1);
    queue<int> q;
    depth[0] = 0; q.push(0);
    while (!q.empty()) {
        int u = q.front(); q.pop();
        for (int v : adj[u]) if (depth[v] == -1) { depth[v] = depth[u] + 1; q.push(v); }
    }
    for (int i = 0; i < n; i++)
        if (depth[i] == -1) quitf(_fail, "input graph is not connected (vertex %d unreachable from 0)", i);
    vector<int> order(n);
    iota(order.begin(), order.end(), 0);
    sort(order.begin(), order.end(), [&](int a, int b) {
        if (depth[a] != depth[b]) return depth[a] > depth[b];
        return a < b;
    });
    order.pop_back(); // drop vertex 0 (guaranteed unique minimum depth == 0, sorts last)
    if (!order.empty() && order.back() == 0) { /* sanity: fine either way */ }
    return order;
}

long long computeBaseline(const vector<int>& order) {
    vector<char> alive(n, 1);
    long long B = 0;
    for (int v : order) {
        long long k = 0;
        for (int u : adj[v]) if (alive[u]) k++;
        if (k < 1) quitf(_fail, "internal baseline: vertex %d has no live neighbor (generator bug)", v);
        B += k * k;
        alive[v] = 0;
    }
    return B;
}

int main(int argc, char* argv[]) {
    registerTestlibCmd(argc, argv);

    n = inf.readInt(2, 1000000, "n");
    m = inf.readLong(0LL, 20000000LL, "m");
    adj.assign(n, {});
    for (long long i = 0; i < m; i++) {
        int u = inf.readInt(0, n - 1, "u");
        int v = inf.readInt(0, n - 1, "v");
        adj[u].push_back(v);
        adj[v].push_back(u);
    }

    vector<int> order = bfsDeepestFirstOrder();
    long long B = computeBaseline(order);
    if (B <= 0) B = 1;

    vector<char> alive(n, 1), eliminated(n, 0);
    int aliveCount = n;
    long long F = 0;

    for (int step = 0; step < n - 1; step++) {
        if (ouf.seekEof())
            quitf(_wa, "output ended after %d elimination lines, expected %d", step, n - 1);
        long long v = ouf.readLong(0LL, (long long)(n - 1), "v");
        if (eliminated[v]) quitf(_wa, "vertex %lld eliminated a second time", v);

        long long k = ouf.readLong(1LL, (long long)(aliveCount - 1), "k");
        set<int> gset;
        for (long long j = 0; j < k; j++) {
            long long g = ouf.readLong(0LL, (long long)(n - 1), "g");
            if (g == v) quitf(_wa, "guard equals the eliminated vertex %lld", v);
            if (eliminated[g]) quitf(_wa, "guard %lld is already eliminated", g);
            if (!gset.insert((int)g).second) quitf(_wa, "duplicate guard %lld in step eliminating %lld", g, v);
        }

        vector<char> covered(n, 0);
        for (int g : gset) {
            covered[g] = 1;
            for (int w : adj[g]) if (alive[w]) covered[w] = 1;
        }
        if (!covered[v])
            quitf(_wa, "vertex %lld is not covered by its own guard set (no guard adjacent to it)", v);
        for (int w : adj[(int)v])
            if (alive[w] && !covered[w])
                quitf(_wa, "live neighbor %d of eliminated vertex %lld is not covered by the guard set", w, v);

        F += k * k;
        alive[(int)v] = 0;
        eliminated[(int)v] = 1;
        aliveCount--;
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output after the %d required elimination lines", n - 1);

    double sc = min(1000.0, 100.0 * (double)B / (double)max(1LL, F));
    double ratio = sc / 1000.0;
    quitp(ratio, "OK F=%lld B=%lld Ratio: %.6f", F, B, ratio);
    return 0;
}
