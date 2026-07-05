// TIER: strong
// Weight-over-degree greedy start + local-search swap improvement:
// repeatedly add a vertex and evict its lighter conflicting chosen neighbours
// whenever that yields a net value gain. Diverges per test from plain greedy.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n, m;
    scanf("%d %d", &n, &m);
    vector<long long> w(n + 1);
    for (int i = 1; i <= n; i++) scanf("%lld", &w[i]);
    vector<vector<int>> adj(n + 1);
    for (int i = 0; i < m; i++) {
        int u, v; scanf("%d %d", &u, &v);
        adj[u].push_back(v);
        adj[v].push_back(u);
    }

    // --- start: weight / (1 + degree) greedy ---
    vector<int> order(n);
    for (int i = 0; i < n; i++) order[i] = i + 1;
    sort(order.begin(), order.end(), [&](int a, int b) {
        double ka = (double)w[a] / (1.0 + adj[a].size());
        double kb = (double)w[b] / (1.0 + adj[b].size());
        if (ka != kb) return ka > kb;
        return a < b;
    });
    vector<char> chosen(n + 1, 0), blocked(n + 1, 0);
    for (int u : order) {
        if (blocked[u]) continue;
        chosen[u] = 1;
        for (int v : adj[u]) blocked[v] = 1;
    }

    // --- local search: swap improvement passes ---
    // For each vertex v: let cs = total value of chosen neighbours of v.
    // If v is not chosen and w[v] > cs, evict those neighbours and add v.
    for (int pass = 0; pass < 12; pass++) {
        bool improved = false;
        for (int v = 1; v <= n; v++) {
            if (chosen[v]) continue;
            long long cs = 0;
            for (int u : adj[v]) if (chosen[u]) cs += w[u];
            if (w[v] > cs) {
                for (int u : adj[v]) if (chosen[u]) chosen[u] = 0;
                chosen[v] = 1;
                improved = true;
            }
        }
        if (!improved) break;
    }

    vector<int> pick;
    for (int v = 1; v <= n; v++) if (chosen[v]) pick.push_back(v);
    printf("%d\n", (int)pick.size());
    for (int v : pick) printf("%d\n", v);
    return 0;
}
