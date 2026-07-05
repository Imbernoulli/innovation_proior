// TIER: greedy
// Start from the balance-only baseline, then do ONE improving flip pass over
// venues in descending incident-weight order (balance-aware, single sweep).
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main() {
    int n, m; ll L, U;
    scanf("%d %d %lld %lld", &n, &m, &L, &U);
    vector<ll> p(n + 1);
    for (int i = 1; i <= n; i++) scanf("%lld", &p[i]);
    vector<vector<pair<int,ll>>> adj(n + 1);
    vector<ll> wsum(n + 1, 0);
    for (int i = 0; i < m; i++) {
        int a, b; ll w; scanf("%d %d %lld", &a, &b, &w);
        adj[a].push_back({b, w});
        adj[b].push_back({a, w});
        wsum[a] += w; wsum[b] += w;
    }

    // baseline start
    vector<int> c(n + 1, 0);
    ll popA = 0, sB = 0;
    for (int i = 1; i <= n; i++) {
        if (popA < sB) { c[i] = 1; popA += p[i]; }
        else           { c[i] = 0; sB += p[i]; }
    }

    // order venues by descending incident weight
    vector<int> order(n);
    for (int i = 0; i < n; i++) order[i] = i + 1;
    sort(order.begin(), order.end(), [&](int x, int y){ return wsum[x] > wsum[y]; });

    for (int idx = 0; idx < n; idx++) {
        int i = order[idx];
        ll delta = 0;
        for (auto& e : adj[i]) delta += (c[i] == c[e.first] ? e.second : -e.second);
        if (delta <= 0) continue;
        // feasibility of flip
        if (c[i] == 0) { if (popA + p[i] > U) continue; c[i] = 1; popA += p[i]; }
        else           { if (popA - p[i] < L) continue; c[i] = 0; popA -= p[i]; }
    }

    for (int i = 1; i <= n; i++) printf("%d%c", c[i], i == n ? '\n' : ' ');
    return 0;
}
