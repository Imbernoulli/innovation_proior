#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

// Checker/scorer for Smart-City Adaptive Lighting (min-cost R-multi-cover).
// Baseline B = sum of all costs (build a lamp everywhere). Feasible output cost F.
// ratio = min(1.0, 0.1 * B / F).  Trivial (build-all) -> 0.1; infeasible -> 0.

int N, M, R;
vector<vector<int>> adj;

int main(int argc, char* argv[]) {
    registerTestlibCmd(argc, argv);

    N = inf.readInt();
    M = inf.readInt();
    R = inf.readInt();
    vector<long long> c(N + 1);
    long long B = 0;
    for (int i = 1; i <= N; i++) { c[i] = inf.readInt(); B += c[i]; }
    vector<int> d(N + 1);
    for (int i = 1; i <= N; i++) d[i] = inf.readInt();
    adj.assign(N + 1, {});
    for (int e = 0; e < M; e++) {
        int u = inf.readInt(), v = inf.readInt();
        adj[u].push_back(v);
        adj[v].push_back(u);
    }
    if (B <= 0) B = 1; // safety; costs are >=1 so unreachable.

    // Read participant output.
    int k = ouf.readInt(0, N, "k");
    vector<int> chosen(k);
    vector<char> picked(N + 1, 0);
    long long F = 0;
    for (int i = 0; i < k; i++) {
        int v = ouf.readInt(1, N, "lamp");
        if (picked[v]) quitf(_wa, "duplicate lamp index %d", v);
        picked[v] = 1;
        chosen[i] = v;
        F += c[v];
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing data after lamp list");

    // Coverage counts: for each built lamp, BFS to depth R, increment cov[u].
    vector<int> cov(N + 1, 0);
    vector<int> stamp(N + 1, 0), dist(N + 1, 0);
    int tok = 0;
    for (int lamp : chosen) {
        tok++;
        queue<int> q; q.push(lamp); stamp[lamp] = tok; dist[lamp] = 0;
        cov[lamp]++;
        while (!q.empty()) {
            int x = q.front(); q.pop();
            if (dist[x] == R) continue;
            for (int y : adj[x]) if (stamp[y] != tok) {
                stamp[y] = tok; dist[y] = dist[x] + 1;
                cov[y]++;
                q.push(y);
            }
        }
    }

    // Feasibility: every demand met.
    for (int u = 1; u <= N; u++) {
        if (cov[u] < d[u])
            quitf(_wa, "vertex %d under-lit: has %d lamps within R=%d, needs %d",
                  u, cov[u], R, d[u]);
    }

    if (F <= 0) F = 0; // build-nothing only possible if all demands are 0 (they are >=1).
    long long denom = max(1LL, F);
    double sc = min(1000.0, 100.0 * (double)B / (double)denom);
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}
