// TIER: strong
// Best-improvement local search over shortest-path edges, budget-aware.
// Each round: evaluate the TRUE delay gain of removing each affordable edge on the
// current shortest path (one Dijkstra per candidate) and commit the edge maximizing
// gain-per-cost (knapsack-aware), keeping s-t connected. Repeat until budget exhausted
// or no improving move. A bounded Dijkstra budget keeps it deterministic and fast.
#include <bits/stdc++.h>
using namespace std;
static const long long INF = (long long)4e18;

int n, m, s, t; long long budget;
vector<long long> w, cost;
vector<vector<pair<int,int>>> adj;
long long dijkstraCalls = 0;
const long long MAX_DIJKSTRA = 8000;

long long dijkstra(const vector<char>& removed, vector<int>* parEdge, vector<int>* parNode) {
    dijkstraCalls++;
    vector<long long> dist(n + 1, INF);
    if (parEdge) parEdge->assign(n + 1, -1);
    if (parNode) parNode->assign(n + 1, -1);
    priority_queue<pair<long long,int>, vector<pair<long long,int>>, greater<>> pq;
    dist[s] = 0; pq.push({0, s});
    while (!pq.empty()) {
        auto [d, u] = pq.top(); pq.pop();
        if (d != dist[u]) continue;
        for (auto [v, ei] : adj[u]) {
            if (removed[ei]) continue;
            long long nd = d + w[ei];
            if (nd < dist[v]) {
                dist[v] = nd;
                if (parEdge) (*parEdge)[v] = ei;
                if (parNode) (*parNode)[v] = u;
                pq.push({nd, v});
            }
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

    while (dijkstraCalls < MAX_DIJKSTRA) {
        vector<int> parEdge, parNode;
        long long cur = dijkstra(removed, &parEdge, &parNode);
        if (cur >= INF) break;
        // Path edges.
        vector<int> pathEdges;
        int u = t;
        while (u != s && parEdge[u] != -1) { pathEdges.push_back(parEdge[u]); u = parNode[u]; }

        int bestEdge = -1;
        double bestScore = 0.0;
        long long bestGain = 0;
        for (int e : pathEdges) {
            if (removed[e]) continue;
            if (cost[e] > remaining) continue;
            if (dijkstraCalls >= MAX_DIJKSTRA) break;
            removed[e] = 1;
            long long nd = dijkstra(removed, nullptr, nullptr);
            removed[e] = 0;
            if (nd >= INF) continue; // would disconnect
            long long gain = nd - cur;
            if (gain <= 0) continue;
            double score = (double)gain / (double)cost[e]; // gain per unit budget
            if (bestEdge == -1 || score > bestScore + 1e-12 ||
                (fabs(score - bestScore) <= 1e-12 && gain > bestGain)) {
                bestEdge = e; bestScore = score; bestGain = gain;
            }
        }
        if (bestEdge == -1) break;
        removed[bestEdge] = 1;
        chosen.push_back(bestEdge);
        remaining -= cost[bestEdge];
        if (remaining <= 0) break;
    }

    printf("%d\n", (int)chosen.size());
    for (int e : chosen) printf("%d\n", e);
    return 0;
}
