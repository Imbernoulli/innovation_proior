#include <bits/stdc++.h>
using namespace std;

// Min-cost max-flow via successive shortest paths with Johnson potentials
// (Dijkstra on reduced non-negative costs). Capacities are integers, costs are
// non-negative long long.
struct MCMF {
    struct Edge { int to; long long cap, cost; int rev; };
    int n;
    vector<vector<Edge>> g;
    vector<long long> h, dist;      // potentials and Dijkstra distances
    vector<int> prevv, preve;       // path reconstruction
    MCMF(int n_) : n(n_), g(n_), h(n_), dist(n_), prevv(n_), preve(n_) {}
    void add_edge(int from, int to, long long cap, long long cost) {
        g[from].push_back({to, cap, cost, (int)g[to].size()});
        g[to].push_back({from, 0, -cost, (int)g[from].size() - 1});
    }
    // Returns {flow, cost} pushing up to maxf units of flow from s to t.
    pair<long long,long long> min_cost_flow(int s, int t, long long maxf) {
        long long flow = 0, cost = 0;
        fill(h.begin(), h.end(), 0);
        while (flow < maxf) {
            // Dijkstra over reduced costs cost + h[u] - h[v] >= 0.
            priority_queue<pair<long long,int>, vector<pair<long long,int>>,
                           greater<pair<long long,int>>> pq;
            fill(dist.begin(), dist.end(), LLONG_MAX);
            dist[s] = 0;
            pq.push({0, s});
            while (!pq.empty()) {
                auto [d, u] = pq.top(); pq.pop();
                if (d > dist[u]) continue;
                for (int i = 0; i < (int)g[u].size(); i++) {
                    Edge &e = g[u][i];
                    if (e.cap <= 0) continue;
                    if (h[u] == LLONG_MAX) continue; // unreachable potential
                    long long nd = dist[u] + e.cost + h[u] - h[e.to];
                    if (nd < dist[e.to]) {
                        dist[e.to] = nd;
                        prevv[e.to] = u;
                        preve[e.to] = i;
                        pq.push({nd, e.to});
                    }
                }
            }
            if (dist[t] == LLONG_MAX) break;        // sink unreachable
            for (int v = 0; v < n; v++)
                if (dist[v] < LLONG_MAX) h[v] += dist[v];
            // Bottleneck along the found shortest path.
            long long d = maxf - flow;
            for (int v = t; v != s; v = prevv[v])
                d = min(d, g[prevv[v]][preve[v]].cap);
            for (int v = t; v != s; v = prevv[v]) {
                Edge &e = g[prevv[v]][preve[v]];
                e.cap -= d;
                g[v][e.rev].cap += d;
            }
            flow += d;
            cost += d * h[t];   // h[t] == true shortest-path cost s->t
        }
        return {flow, cost};
    }
};

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int W, T;
    if (!(cin >> W >> T)) return 0;

    // c[i][j] = base cost to assign task j to worker i.
    vector<vector<long long>> c(W, vector<long long>(T));
    for (int i = 0; i < W; i++)
        for (int j = 0; j < T; j++)
            cin >> c[i][j];

    vector<long long> q(W), base(W);
    for (int i = 0; i < W; i++) cin >> q[i];     // regular quota of worker i
    for (int i = 0; i < W; i++) cin >> base[i];  // overtime slope of worker i

    // Node layout: 0 = source, 1..T = tasks, T+1..T+W = workers, T+W+1 = sink.
    int S = 0;
    auto TASK = [&](int j) { return 1 + j; };
    auto WORK = [&](int i) { return 1 + T + i; };
    int K = 1 + T + W;
    MCMF mc(K + 1);

    for (int j = 0; j < T; j++) mc.add_edge(S, TASK(j), 1, 0);

    for (int i = 0; i < W; i++)
        for (int j = 0; j < T; j++)
            mc.add_edge(TASK(j), WORK(i), 1, c[i][j]);

    // Convex overtime cost as parallel unit edges worker_i -> sink.
    // Marginal cost of the m-th task on worker i (m = 1..T) is
    //   s_i(m) = base[i] * max(0, m - q[i]),
    // which is non-decreasing in m, so the total per-worker cost is convex.
    for (int i = 0; i < W; i++) {
        for (int m = 1; m <= T; m++) {
            long long over = (long long)max(0LL, (long long)m - q[i]);
            long long marginal = base[i] * over;
            mc.add_edge(WORK(i), K, 1, marginal);
        }
    }

    auto [flow, cost] = mc.min_cost_flow(S, K, T);
    // Every task can always be routed (workers have capacity T each), so flow==T.
    cout << cost << "\n";
    return 0;
}
