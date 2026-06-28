#include <bits/stdc++.h>
using namespace std;

/*
  Min-cost flow of a fixed value F on a directed graph whose edge costs may be
  NEGATIVE (but the graph has no negative-cost cycle).

  Successive Shortest Paths (SSP) augments along a shortest (cheapest) s->t path
  in the residual graph each round. With negative original costs, plain Dijkstra
  is invalid. Johnson's potentials fix this: maintain h[v] so the reduced cost
  rc(u,v) = cost(u,v) + h[u] - h[v] is >= 0 on every residual arc, then Dijkstra
  on reduced costs is valid. Bellman-Ford ONCE seeds h from true shortest
  distances (the only step that must tolerate negative edges); after each
  Dijkstra, h[v] += dist[v] keeps the invariant for the new residual arcs.
*/

struct MCMF {
    struct Edge { int to, rev; long long cap, cost; };
    int n;
    vector<vector<Edge>> g;
    MCMF(int n_) : n(n_), g(n_) {}
    void add_edge(int u, int v, long long cap, long long cost) {
        g[u].push_back({v, (int)g[v].size(), cap, cost});
        g[v].push_back({u, (int)g[u].size() - 1, 0, -cost});
    }

    static const long long INF = (long long)4e18;

    // Push up to maxf units s->t at minimum cost. Returns {flow_pushed, total_cost}.
    pair<long long,long long> min_cost_flow(int s, int t, long long maxf) {
        vector<long long> h(n, 0);

        // --- Bellman-Ford ONCE to initialize potentials (negative edges allowed) ---
        // Distances from s along edges with positive residual capacity.
        {
            fill(h.begin(), h.end(), INF);
            h[s] = 0;
            // Relax up to n-1 times (no negative cycle assumed -> converges).
            for (int it = 0; it < n - 1; it++) {
                bool changed = false;
                for (int u = 0; u < n; u++) {
                    if (h[u] == INF) continue;
                    for (auto &e : g[u]) {
                        if (e.cap > 0 && h[u] + e.cost < h[e.to]) {
                            h[e.to] = h[u] + e.cost;
                            changed = true;
                        }
                    }
                }
                if (!changed) break;
            }
            // Unreachable vertices get potential 0 so reduced costs stay finite.
            for (int v = 0; v < n; v++) if (h[v] == INF) h[v] = 0;
        }

        long long total_flow = 0, total_cost = 0;
        vector<long long> dist(n);
        vector<int> pv(n), pe(n);

        while (total_flow < maxf) {
            // --- Dijkstra on reduced costs rc = cost + h[u] - h[v] (>= 0) ---
            fill(dist.begin(), dist.end(), INF);
            dist[s] = 0;
            priority_queue<pair<long long,int>, vector<pair<long long,int>>,
                           greater<pair<long long,int>>> pq;
            pq.push({0, s});
            while (!pq.empty()) {
                auto [d, u] = pq.top(); pq.pop();
                if (d > dist[u]) continue;
                for (int i = 0; i < (int)g[u].size(); i++) {
                    Edge &e = g[u][i];
                    if (e.cap <= 0) continue;
                    long long rc = e.cost + h[u] - h[e.to];
                    if (dist[u] + rc < dist[e.to]) {
                        dist[e.to] = dist[u] + rc;
                        pv[e.to] = u; pe[e.to] = i;
                        pq.push({dist[e.to], e.to});
                    }
                }
            }
            if (dist[t] == INF) break; // t unreachable: cannot push more flow

            // Update potentials by the new shortest reduced distances.
            for (int v = 0; v < n; v++)
                if (dist[v] < INF) h[v] += dist[v];

            // Find bottleneck capacity along the s->t path.
            long long push = maxf - total_flow;
            for (int v = t; v != s; v = pv[v])
                push = min(push, g[pv[v]][pe[v]].cap);

            // Apply the augmentation.
            for (int v = t; v != s; v = pv[v]) {
                Edge &e = g[pv[v]][pe[v]];
                e.cap -= push;
                g[v][e.rev].cap += push;
            }
            total_flow += push;
            // True path cost = reduced path dist + (h[t]-h[s]); but after the h
            // update above h[t] already equals the cumulative true distance, so
            // the per-unit cost of this path is exactly h[t] - h[s].
            total_cost += push * (h[t] - h[s]);
        }
        return {total_flow, total_cost};
    }
};

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, m, s, t;
    long long F;
    if (!(cin >> n >> m >> s >> t >> F)) return 0;

    MCMF mc(n);
    for (int i = 0; i < m; i++) {
        int u, v; long long cap, cost;
        cin >> u >> v >> cap >> cost;
        mc.add_edge(u, v, cap, cost);
    }

    auto [flow, cost] = mc.min_cost_flow(s, t, F);
    if (flow < F) cout << "IMPOSSIBLE\n";
    else cout << cost << "\n";
    return 0;
}
