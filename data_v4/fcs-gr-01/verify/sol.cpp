#include <bits/stdc++.h>
using namespace std;

// Maximum-profit project selection via max-weight closure -> min cut (Dinic).
//
// Build a flow network:
//   source S -> project i   with capacity profit[i]   (only profit[i] > 0)
//   machine j -> sink T      with capacity cost[j]
//   project i -> machine j   with capacity +infinity   if i requires j
// Then  answer = (sum of positive profits) - maxflow(S, T).
//
// (Projects with profit <= 0 contribute their value as a direct adjustment so
//  that the "sum of positive profits" baseline and the cut stay consistent.)

struct Dinic {
    struct Edge { int to; long long cap; int rev; };
    vector<vector<Edge>> g;
    vector<int> level, iter;
    int n;
    Dinic(int n_) : g(n_), level(n_), iter(n_), n(n_) {}
    void add_edge(int from, int to, long long cap) {
        g[from].push_back({to, cap, (int)g[to].size()});
        g[to].push_back({from, 0, (int)g[from].size() - 1});
    }
    bool bfs(int s, int t) {
        fill(level.begin(), level.end(), -1);
        queue<int> q;
        level[s] = 0;
        q.push(s);
        while (!q.empty()) {
            int v = q.front(); q.pop();
            for (const Edge &e : g[v]) {
                if (e.cap > 0 && level[e.to] < 0) {
                    level[e.to] = level[v] + 1;
                    q.push(e.to);
                }
            }
        }
        return level[t] >= 0;
    }
    long long dfs(int v, int t, long long f) {
        if (v == t) return f;
        for (int &i = iter[v]; i < (int)g[v].size(); ++i) {
            Edge &e = g[v][i];
            if (e.cap > 0 && level[v] < level[e.to]) {
                long long d = dfs(e.to, t, min(f, e.cap));
                if (d > 0) {
                    e.cap -= d;
                    g[e.to][e.rev].cap += d;
                    return d;
                }
            }
        }
        return 0;
    }
    long long max_flow(int s, int t) {
        long long flow = 0;
        const long long INF = (long long)4e18;
        while (bfs(s, t)) {
            fill(iter.begin(), iter.end(), 0);
            long long f;
            while ((f = dfs(s, t, INF)) > 0) flow += f;
        }
        return flow;
    }
};

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, m;
    if (!(cin >> n >> m)) return 0;

    vector<long long> profit(n), cost(m);
    for (int i = 0; i < n; ++i) cin >> profit[i];
    for (int j = 0; j < m; ++j) cin >> cost[j];

    // Node ids: 0 = source, 1..n = projects, n+1..n+m = machines, n+m+1 = sink.
    int S = 0;
    int T = n + m + 1;
    Dinic dinic(n + m + 2);

    const long long INF = (long long)4e18;

    long long base = 0; // sum of positive profits
    for (int i = 0; i < n; ++i) {
        if (profit[i] > 0) {
            base += profit[i];
            dinic.add_edge(S, 1 + i, profit[i]);
        } else if (profit[i] < 0) {
            // A project with non-positive profit behaves like a "cost" that the
            // closure can pay to keep the project itself in (or drop it). Model it
            // as an edge project -> sink with capacity (-profit).
            dinic.add_edge(1 + i, T, -profit[i]);
        }
        // profit[i] == 0 contributes nothing either way.
    }
    for (int j = 0; j < m; ++j) {
        // Machine cost: machine -> sink with capacity cost[j] (cost >= 0).
        dinic.add_edge(1 + n + j, T, cost[j]);
    }

    // Prerequisite edges: project i requires machine j  =>  i -> j with cap INF.
    int E;
    cin >> E;
    for (int e = 0; e < E; ++e) {
        int i, j;
        cin >> i >> j; // 1-based project id, 1-based machine id
        dinic.add_edge(1 + (i - 1), 1 + n + (j - 1), INF);
    }

    long long cut = dinic.max_flow(S, T);
    long long answer = base - cut;

    cout << answer << "\n";
    return 0;
}
