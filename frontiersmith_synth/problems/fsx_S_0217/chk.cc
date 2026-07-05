#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

static const long long INF = (long long)4e18;

struct Graph {
    int n;
    vector<vector<pair<int,int>>> adj; // (to, edgeIdx)
    vector<long long> w;
    Graph(int n_) : n(n_), adj(n_ + 1) {}
};

static long long dijkstra(const Graph& g, int s, int tt, const vector<char>& removed) {
    vector<long long> dist(g.n + 1, INF);
    priority_queue<pair<long long,int>, vector<pair<long long,int>>, greater<>> pq;
    dist[s] = 0;
    pq.push({0, s});
    while (!pq.empty()) {
        auto [d, u] = pq.top(); pq.pop();
        if (d != dist[u]) continue;
        if (u == tt) return d;
        for (auto [v, ei] : g.adj[u]) {
            if (removed[ei]) continue;
            long long nd = d + g.w[ei];
            if (nd < dist[v]) { dist[v] = nd; pq.push({nd, v}); }
        }
    }
    return dist[tt];
}

int main(int argc, char** argv) {
    registerTestlibCmd(argc, argv);

    int n = inf.readInt();
    int m = inf.readInt();
    int s = inf.readInt();
    int tt = inf.readInt();
    long long budget = inf.readLong();

    Graph g(n);
    g.w.assign(m + 1, 0);
    vector<long long> cost(m + 1, 0);
    for (int e = 1; e <= m; e++) {
        int u = inf.readInt();
        int v = inf.readInt();
        long long w = inf.readLong();
        long long c = inf.readLong();
        g.w[e] = w;
        cost[e] = c;
        g.adj[u].push_back({v, e});
        g.adj[v].push_back({u, e});
    }

    vector<char> none(m + 1, 0);
    long long B = dijkstra(g, s, tt, none);
    if (B >= INF) quitf(_fail, "internal: source cannot reach sink in original network");
    if (B <= 0) quitf(_fail, "internal: non-positive baseline");

    // Read participant output.
    int r = ouf.readInt(0, m, "r");
    vector<char> removed(m + 1, 0);
    long long spent = 0;
    for (int i = 0; i < r; i++) {
        int idx = ouf.readInt(1, m, "idx");
        if (removed[idx]) quitf(_wa, "duplicate edge index %d", idx);
        removed[idx] = 1;
        spent += cost[idx];
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing tokens in output");

    if (spent > budget)
        quitf(_wa, "closure cost %lld exceeds budget %lld", spent, budget);

    long long F = dijkstra(g, s, tt, removed);
    if (F >= INF)
        quitf(_wa, "sink disconnected from source after removals");

    double sc = min(1000.0, 100.0 * (double)F / (double)max(1LL, B));
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc / 1000.0);
}
