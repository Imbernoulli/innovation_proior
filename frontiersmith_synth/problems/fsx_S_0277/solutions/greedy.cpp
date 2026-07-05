// TIER: greedy
// Cost-greedy: repeatedly shut the CHEAPEST affordable junction lying on the current
// shortest route, provided s-t stays connected. Fast one-pass-ish heuristic.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int n, m, s, t;
ll K;
vector<int> cst;
vector<vector<pair<int,int>>> adj;

// Dijkstra with predecessor recovery over the live subgraph (dead[]==0).
// Returns dist to t (-1 if unreachable); fills pred for path recovery.
ll dijkstraPath(const vector<char>& dead, vector<int>& pred) {
    const ll INF = (ll)4e18;
    vector<ll> dist(n + 1, INF);
    pred.assign(n + 1, -1);
    priority_queue<pair<ll,int>, vector<pair<ll,int>>, greater<>> pq;
    dist[s] = 0; pq.push({0, s});
    while (!pq.empty()) {
        auto [d, u] = pq.top(); pq.pop();
        if (d != dist[u]) continue;
        for (auto& [v, w] : adj[u]) {
            if (dead[v]) continue;
            if (d + w < dist[v]) { dist[v] = d + w; pred[v] = u; pq.push({dist[v], v}); }
        }
    }
    return dist[t] >= INF ? -1 : dist[t];
}

// Is t reachable from s over the live subgraph?
bool connected(const vector<char>& dead) {
    vector<char> vis(n + 1, 0);
    vector<int> st = {s}; vis[s] = 1;
    while (!st.empty()) {
        int u = st.back(); st.pop_back();
        if (u == t) return true;
        for (auto& [v, w] : adj[u]) if (!dead[v] && !vis[v]) { vis[v] = 1; st.push_back(v); }
    }
    return vis[t];
}

int main() {
    scanf("%d %d %d %d %lld", &n, &m, &s, &t, &K);
    cst.assign(n + 1, 0);
    for (int i = 1; i <= n; i++) scanf("%d", &cst[i]);
    adj.assign(n + 1, {});
    for (int i = 0; i < m; i++) {
        int u, v, w; scanf("%d %d %d", &u, &v, &w);
        adj[u].push_back({v, w}); adj[v].push_back({u, w});
    }

    vector<char> dead(n + 1, 0);
    ll budget = K;
    vector<int> chosen;

    while (budget > 0) {
        vector<int> pred;
        ll d = dijkstraPath(dead, pred);
        if (d < 0) break;
        // recover interior path nodes
        vector<int> path;
        for (int u = t; u != -1; u = pred[u]) path.push_back(u);
        // candidate interior nodes on the path
        vector<int> cand;
        for (int u : path) if (u != s && u != t && !dead[u] && cst[u] <= budget) cand.push_back(u);
        // cheapest first
        sort(cand.begin(), cand.end(), [&](int a, int b) {
            if (cst[a] != cst[b]) return cst[a] < cst[b];
            return a < b;
        });
        int picked = -1;
        for (int u : cand) {
            dead[u] = 1;
            if (connected(dead)) { picked = u; break; }
            dead[u] = 0;
        }
        if (picked < 0) break;
        budget -= cst[picked];
        chosen.push_back(picked);
    }

    printf("%d\n", (int)chosen.size());
    for (int u : chosen) printf("%d\n", u);
    return 0;
}
