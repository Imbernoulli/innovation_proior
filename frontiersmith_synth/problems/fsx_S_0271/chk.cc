#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

// chk.cc -- scorer for Lunar Habitat Pressure Loops (survivable degree-capped tunnel grid).
// Minimization. Validates: distinct valid endpoints, no duplicate tunnel, per-module
// degree <= cap, and 2-edge-connectivity (spanning + connected + NO bridge). Baseline
// B = length of the input-order Hamiltonian cycle 1-2-...-N-1 (always feasible, positive).
// Participant length F = sum of rounded Euclidean tunnel lengths.
// ratio = min(1, (B / max(1,F)) / 10)  ->  loop baseline scores 0.1, cap 1.0 at F <= B/10.

static vector<long long> X, Y;
static inline long long len(int a, int b) {
    double dx = (double)(X[a] - X[b]);
    double dy = (double)(Y[a] - Y[b]);
    return (long long)llround(sqrt(dx * dx + dy * dy));
}

// Iterative bridge finder (Tarsjan lowlink) over a simple undirected graph given as
// adjacency lists of (neighbor, edgeId). Returns true iff the graph is connected across
// all N vertices AND has no bridge (i.e. is 2-edge-connected).
static bool two_edge_connected(int N, const vector<vector<pair<int,int>>>& adj) {
    vector<int> disc(N, -1), low(N, 0);
    int timer = 0;
    int visitedCount = 0;
    // iterative DFS from vertex 0 (graph must be connected, so one root suffices; if not
    // all reached we return false).
    vector<int> parentEdge(N, -1);
    vector<int> itPtr(N, 0);
    vector<int> stk;
    stk.reserve(N);
    disc[0] = low[0] = timer++;
    stk.push_back(0);
    visitedCount = 1;
    bool bridge = false;
    while (!stk.empty()) {
        int u = stk.back();
        if (itPtr[u] < (int)adj[u].size()) {
            auto [v, eid] = adj[u][itPtr[u]++];
            if (eid == parentEdge[u]) continue;   // skip the edge we came in on
            if (disc[v] == -1) {
                disc[v] = low[v] = timer++;
                parentEdge[v] = eid;
                visitedCount++;
                stk.push_back(v);
            } else {
                low[u] = min(low[u], disc[v]);
            }
        } else {
            stk.pop_back();
            if (!stk.empty()) {
                int p = stk.back();
                low[p] = min(low[p], low[u]);
                if (low[u] > disc[p]) bridge = true;  // edge (p,u) is a bridge
            }
        }
    }
    if (visitedCount != N) return false;   // not connected / not spanning
    return !bridge;
}

int main(int argc, char* argv[]) {
    registerTestlibCmd(argc, argv);

    int N = inf.readInt();
    X.resize(N);
    Y.resize(N);
    vector<int> cap(N);
    for (int i = 0; i < N; i++) {
        X[i] = inf.readInt();
        Y[i] = inf.readInt();
        cap[i] = inf.readInt();
    }

    // A 2-edge-connected simple graph with max degree 4 has at most 2N edges. Bound E so a
    // malformed / over-sized output is rejected (scores 0) rather than read unboundedly.
    int E = ouf.readInt(0, 2 * N, "E");
    vector<int> deg(N, 0);
    set<pair<int,int>> seen;
    vector<vector<pair<int,int>>> adj(N);
    long long F = 0;
    for (int k = 0; k < E; k++) {
        int u = ouf.readInt(1, N, "a") - 1;
        int v = ouf.readInt(1, N, "b") - 1;
        if (u == v) quitf(_wa, "tunnel %d is a self-loop at module %d", k + 1, u + 1);
        int a = min(u, v), b = max(u, v);
        if (!seen.insert({a, b}).second)
            quitf(_wa, "duplicate tunnel between modules %d and %d", a + 1, b + 1);
        adj[u].push_back({v, k});
        adj[v].push_back({u, k});
        deg[u]++; deg[v]++;
        F += len(u, v);
    }
    if (!ouf.seekEof()) quitf(_wa, "extra trailing output after the tunnel list");

    // degree caps
    for (int i = 0; i < N; i++)
        if (deg[i] > cap[i])
            quitf(_wa, "module %d has degree %d exceeding its port cap %d", i + 1, deg[i], cap[i]);

    // survivability: 2-edge-connected (spanning + connected + bridgeless)
    if (N < 3) quitf(_wa, "instance too small for a survivable network");
    if (!two_edge_connected(N, adj))
        quitf(_wa, "network is not 2-edge-connected (disconnected or contains a bridge)");

    if (F <= 0) quitf(_wa, "degenerate zero-length network");

    // baseline B = input-order Hamiltonian cycle length (always feasible, positive)
    long long B = 0;
    for (int i = 0; i + 1 < N; i++) B += len(i, i + 1);
    B += len(N - 1, 0);

    double sc = min(1000.0, 100.0 * (double)B / (double)max(1LL, F));
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}
