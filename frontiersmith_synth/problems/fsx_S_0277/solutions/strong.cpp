// TIER: strong
// Best-improvement local search: at each step, for every junction on the current
// shortest route, evaluate the TRUE post-shutdown shortest route and shut the one whose
// removal yields the largest delay per remaining budget (keeping s-t connected).
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int n, m, s, t;
ll K;
vector<int> cst;
vector<vector<pair<int,int>>> adj;

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

// distance only (no pred) over live subgraph; -1 if t unreachable.
ll dijkstra(const vector<char>& dead) {
    const ll INF = (ll)4e18;
    vector<ll> dist(n + 1, INF);
    priority_queue<pair<ll,int>, vector<pair<ll,int>>, greater<>> pq;
    dist[s] = 0; pq.push({0, s});
    while (!pq.empty()) {
        auto [d, u] = pq.top(); pq.pop();
        if (d != dist[u]) continue;
        if (u == t) return d;
        for (auto& [v, w] : adj[u]) {
            if (dead[v]) continue;
            if (d + w < dist[v]) { dist[v] = d + w; pq.push({dist[v], v}); }
        }
    }
    return dist[t] >= INF ? -1 : dist[t];
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

    const int MAX_ITERS = 60;      // safety bound
    const int MAX_CAND = 90;       // cap candidates scanned per iteration

    for (int iter = 0; iter < MAX_ITERS && budget > 0; iter++) {
        vector<int> pred;
        ll cur = dijkstraPath(dead, pred);
        if (cur < 0) break;
        vector<int> path;
        for (int u = t; u != -1; u = pred[u]) path.push_back(u);

        vector<int> cand;
        for (int u : path)
            if (u != s && u != t && !dead[u] && cst[u] <= budget) cand.push_back(u);
        if ((int)cand.size() > MAX_CAND) cand.resize(MAX_CAND);

        int best = -1;
        double bestScore = 0.0;   // delay-per-cost; must be strictly positive
        ll bestF = cur;
        for (int u : cand) {
            dead[u] = 1;
            ll f = dijkstra(dead);
            dead[u] = 0;
            if (f < 0) continue;               // would disconnect
            ll gain = f - cur;
            if (gain <= 0) continue;
            double sc = (double)gain / (double)cst[u];   // bang-per-buck
            if (sc > bestScore + 1e-12 ||
                (fabs(sc - bestScore) <= 1e-12 && f > bestF)) {
                bestScore = sc; best = u; bestF = f;
            }
        }
        if (best < 0) break;
        dead[best] = 1;
        budget -= cst[best];
        chosen.push_back(best);
    }

    printf("%d\n", (int)chosen.size());
    for (int u : chosen) printf("%d\n", u);
    return 0;
}
