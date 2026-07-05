#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main(int argc, char** argv) {
    registerTestlibCmd(argc, argv);

    int N = inf.readInt();
    int M = inf.readInt();
    int r = inf.readInt();

    vector<ll> cost(N + 1);
    ll B = 0;
    for (int i = 1; i <= N; i++) { cost[i] = inf.readInt(); B += cost[i]; }
    if (B <= 0) quitf(_fail, "bad instance: B=%lld", B);

    vector<vector<int>> adj(N + 1);
    for (int e = 0; e < M; e++) {
        int u = inf.readInt(1, N, "u");
        int v = inf.readInt(1, N, "v");
        adj[u].push_back(v);
        adj[v].push_back(u);
    }

    // ---- read & validate participant's watchtower set ----
    int K = ouf.readInt(0, N, "K");
    vector<char> chosen(N + 1, 0);
    vector<int> sel;
    sel.reserve(K);
    for (int i = 0; i < K; i++) {
        int idx = ouf.readInt(1, N, "cellIndex");
        if (chosen[idx]) quitf(_wa, "cell %d built more than once", idx);
        chosen[idx] = 1;
        sel.push_back(idx);
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output tokens");

    // ---- verify full coverage: multi-source BFS bounded to radius r ----
    vector<int> dist(N + 1, -1);
    queue<int> q;
    for (int s : sel) { dist[s] = 0; q.push(s); }
    while (!q.empty()) {
        int u = q.front(); q.pop();
        if (dist[u] == r) continue;          // do not expand beyond radius r
        for (int v : adj[u]) if (dist[v] == -1) {
            dist[v] = dist[u] + 1;
            q.push(v);
        }
    }
    for (int i = 1; i <= N; i++)
        if (dist[i] == -1)
            quitf(_wa, "cell %d is not monitored within radius %d", i, r);

    // ---- objective: total installation cost ----
    ll F = 0;
    for (int s : sel) F += cost[s];
    if (F <= 0) quitf(_wa, "no watchtowers built but coverage claimed"); // guard

    double sc = min(1000.0, 100.0 * (double)B / (double)max((ll)1, F));
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}
