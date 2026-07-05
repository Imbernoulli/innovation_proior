#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

/*
 * Checker / scorer for "Interstellar Relay: Robust Degree-Bounded Backbone".
 *
 * Feasibility (strict): the chosen candidate edges must be distinct, all node
 * degrees <= D, and the resulting subgraph must span & connect all n nodes.
 *
 * Objective (minimize):  F = total_edge_cost + L * max_node_stress
 *   where max_node_stress = the maximum, over v != hub, of the number of nodes
 *   whose unit relay flow to the hub (node 0) passes through v.  The relay uses
 *   the shortest-path (min total edge cost) tree rooted at the hub, with a fully
 *   deterministic tie-break: each node's parent is the incident shortest-path
 *   predecessor u minimizing (dist[u], u).  For a spanning tree this stress is
 *   exactly the rooted subtree size -- a spectral/betweenness robustness surrogate.
 *
 * Baseline B = objective of the backbone path (candidate edges 1..n-1), which is
 * a fragile chain: cost = sum of backbone edge costs, max_stress = n-1.
 */
int main(int argc, char** argv) {
    registerTestlibCmd(argc, argv);

    int n = inf.readInt();
    int m = inf.readInt();
    int D = inf.readInt();
    long long L = inf.readLong();

    vector<int> eu(m + 1), ev(m + 1);
    vector<long long> ew(m + 1);
    for (int j = 1; j <= m; j++) {
        eu[j] = inf.readInt();
        ev[j] = inf.readInt();
        ew[j] = inf.readLong();
    }

    long long backbone = 0;
    for (int i = 1; i <= n - 1; i++) backbone += ew[i];
    long long B = backbone + L * (long long)(n - 1);

    // ---- read participant output ----
    int k = ouf.readInt(0, m, "k");
    vector<char> seen(m + 1, 0);
    vector<int> deg(n, 0);
    vector<vector<pair<int,long long>>> adj(n);
    long long cost = 0;
    for (int t = 0; t < k; t++) {
        int e = ouf.readInt(1, m, "edge_index");
        if (seen[e]) quitf(_wa, "duplicate edge index %d", e);
        seen[e] = 1;
        int u = eu[e], v = ev[e];
        long long w = ew[e];
        deg[u]++; deg[v]++;
        if (deg[u] > D) quitf(_wa, "degree of node %d exceeds D=%d", u, D);
        if (deg[v] > D) quitf(_wa, "degree of node %d exceeds D=%d", v, D);
        adj[u].push_back({v, w});
        adj[v].push_back({u, w});
        cost += w;
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output");

    // ---- connectivity / spanning from hub 0 ----
    vector<char> vis(n, 0);
    queue<int> q; q.push(0); vis[0] = 1; int reached = 1;
    while (!q.empty()) {
        int x = q.front(); q.pop();
        for (auto& e : adj[x]) if (!vis[e.first]) { vis[e.first] = 1; reached++; q.push(e.first); }
    }
    if (reached != n) quitf(_wa, "chosen network is not connected/spanning (reached %d of %d)", reached, n);

    // ---- shortest-path distances from hub ----
    const long long INF = LLONG_MAX;
    vector<long long> dist(n, INF); dist[0] = 0;
    priority_queue<pair<long long,int>, vector<pair<long long,int>>, greater<pair<long long,int>>> pq;
    pq.push({0, 0});
    while (!pq.empty()) {
        auto top = pq.top(); pq.pop();
        long long d = top.first; int x = top.second;
        if (d > dist[x]) continue;
        for (auto& e : adj[x]) {
            long long nd = d + e.second;
            if (nd < dist[e.first]) { dist[e.first] = nd; pq.push({nd, e.first}); }
        }
    }

    // ---- deterministic shortest-path-tree parent ----
    vector<int> par(n, -1);
    for (int v = 0; v < n; v++) {
        if (v == 0) continue;
        long long bestd = INF; int bu = -1;
        for (auto& e : adj[v]) {
            int u = e.first; long long w = e.second;
            if (dist[u] != INF && dist[u] + w == dist[v]) {
                if (bu == -1 || dist[u] < bestd || (dist[u] == bestd && u < bu)) { bestd = dist[u]; bu = u; }
            }
        }
        par[v] = bu; // guaranteed found for a connected graph
    }

    // ---- rooted subtree sizes = node stress; process by decreasing distance ----
    vector<int> order(n);
    iota(order.begin(), order.end(), 0);
    sort(order.begin(), order.end(), [&](int a, int b){
        if (dist[a] != dist[b]) return dist[a] > dist[b];
        return a > b;
    });
    vector<long long> stress(n, 1);
    long long maxstress = 0;
    for (int v : order) {
        if (v == 0) continue;
        if (par[v] >= 0) stress[par[v]] += stress[v];
        if (stress[v] > maxstress) maxstress = stress[v];
    }

    long long F = cost + L * maxstress;

    double sc = min(1000.0, 100.0 * (double)B / (double)max(1LL, F));
    quitp(sc / 1000.0, "OK F=%lld B=%lld cost=%lld maxstress=%lld Ratio: %.6f",
          F, B, cost, maxstress, sc / 1000.0);
    return 0;
}
