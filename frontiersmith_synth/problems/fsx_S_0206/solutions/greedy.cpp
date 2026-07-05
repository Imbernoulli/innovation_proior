// TIER: greedy
// Cost-effective greedy weighted set cover: repeatedly build the depot that
// covers the most still-uncovered shops per unit build cost.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int n, m, D, r;
vector<int> cost;
struct AdjE { int to; ll w; };
vector<vector<AdjE>> g;
vector<int> shops;
vector<int> shopId; // node -> shop index, or -1

// radius-limited Dijkstra from src; returns shop indices within r
vector<int> coverFrom(int src) {
    vector<ll> dist(n + 1, LLONG_MAX);
    priority_queue<pair<ll,int>, vector<pair<ll,int>>, greater<pair<ll,int>>> pq;
    dist[src] = 0; pq.push({0, src});
    vector<int> res;
    while (!pq.empty()) {
        auto [d, u] = pq.top(); pq.pop();
        if (d > dist[u]) continue;
        if (shopId[u] >= 0) res.push_back(shopId[u]);
        if (d >= r) continue;
        for (auto& e : g[u]) {
            ll nd = d + e.w;
            if (nd <= r && nd < dist[e.to]) { dist[e.to] = nd; pq.push({nd, e.to}); }
        }
    }
    return res;
}

int main() {
    if (scanf("%d %d %d %d", &n, &m, &D, &r) != 4) return 0;
    cost.assign(n + 1, 0);
    for (int u = 1; u <= n; u++) scanf("%d", &cost[u]);
    g.assign(n + 1, {});
    for (int i = 0; i < m; i++) {
        int u, v, w; scanf("%d %d %d", &u, &v, &w);
        g[u].push_back({v, (ll)w}); g[v].push_back({u, (ll)w});
    }
    shops.resize(D);
    shopId.assign(n + 1, -1);
    for (int i = 0; i < D; i++) { scanf("%d", &shops[i]); shopId[shops[i]] = i; }

    // coverage[u] = shop indices coverable by a depot at u
    vector<vector<int>> coverage(n + 1);
    for (int u = 1; u <= n; u++) coverage[u] = coverFrom(u);

    vector<char> covered(D, 0);
    int remaining = D;
    vector<int> chosen;
    vector<char> used(n + 1, 0);

    while (remaining > 0) {
        int best = -1; double bestRatio = 1e18; int bestNew = 0;
        for (int u = 1; u <= n; u++) {
            if (used[u]) continue;
            int nc = 0;
            for (int s : coverage[u]) if (!covered[s]) nc++;
            if (nc == 0) continue;
            double ratio = (double)cost[u] / nc;
            if (ratio < bestRatio - 1e-12) { bestRatio = ratio; best = u; bestNew = nc; }
        }
        if (best < 0) break; // should not happen (each shop covers itself)
        used[best] = 1;
        chosen.push_back(best);
        for (int s : coverage[best]) if (!covered[s]) { covered[s] = 1; remaining--; }
    }

    printf("%d\n", (int)chosen.size());
    for (int u : chosen) printf("%d\n", u);
    return 0;
}
