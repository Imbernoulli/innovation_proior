// TIER: greedy
// One-pass greedy: repeatedly cut the cheapest-cost edge lying on the current
// shortest path (that keeps s-t connected) until no affordable path edge remains.
#include <bits/stdc++.h>
using namespace std;
static const long long INF = (long long)4e18;

int n, m, s, t; long long budget;
vector<long long> w, cost;
vector<vector<pair<int,int>>> adj; // (to, edgeIdx)

// Returns dist and fills parent edge for path recovery.
long long dijkstra(const vector<char>& removed, vector<int>& parEdge, vector<int>& parNode) {
    vector<long long> dist(n + 1, INF);
    parEdge.assign(n + 1, -1);
    parNode.assign(n + 1, -1);
    priority_queue<pair<long long,int>, vector<pair<long long,int>>, greater<>> pq;
    dist[s] = 0; pq.push({0, s});
    while (!pq.empty()) {
        auto [d, u] = pq.top(); pq.pop();
        if (d != dist[u]) continue;
        for (auto [v, ei] : adj[u]) {
            if (removed[ei]) continue;
            long long nd = d + w[ei];
            if (nd < dist[v]) { dist[v] = nd; parEdge[v] = ei; parNode[v] = u; pq.push({nd, v}); }
        }
    }
    return dist[t];
}

int main() {
    scanf("%d %d %d %d %lld", &n, &m, &s, &t, &budget);
    w.assign(m + 1, 0); cost.assign(m + 1, 0);
    adj.assign(n + 1, {});
    for (int e = 1; e <= m; e++) {
        int u, v; long long ww, cc;
        scanf("%d %d %lld %lld", &u, &v, &ww, &cc);
        w[e] = ww; cost[e] = cc;
        adj[u].push_back({v, e});
        adj[v].push_back({u, e});
    }

    vector<char> removed(m + 1, 0);
    vector<int> chosen;
    long long remaining = budget;

    while (true) {
        vector<int> parEdge, parNode;
        long long d = dijkstra(removed, parEdge, parNode);
        if (d >= INF) break; // should not happen; stop
        // Collect path edges.
        vector<int> pathEdges;
        int cur = t;
        while (cur != s && parEdge[cur] != -1) {
            pathEdges.push_back(parEdge[cur]);
            cur = parNode[cur];
        }
        // Pick cheapest-cost affordable path edge whose removal keeps s-t connected.
        int best = -1;
        for (int e : pathEdges) {
            if (removed[e]) continue;
            if (cost[e] > remaining) continue;
            if (best == -1 || cost[e] < cost[best]) best = e;
        }
        if (best == -1) break;
        // Tentatively remove; verify connectivity.
        removed[best] = 1;
        vector<int> pe, pn;
        long long nd = dijkstra(removed, pe, pn);
        if (nd >= INF) { removed[best] = 0; break; } // would disconnect; stop
        chosen.push_back(best);
        remaining -= cost[best];
    }

    printf("%d\n", (int)chosen.size());
    for (int e : chosen) printf("%d\n", e);
    return 0;
}
