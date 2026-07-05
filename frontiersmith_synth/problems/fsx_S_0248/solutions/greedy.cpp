// TIER: greedy
// One-pass greedy assignment: process racks by descending total incident coupling
// weight; place each on whichever loop (with remaining capacity) currently gains the
// most split strength against the already-assigned racks. Keeps loops balanced.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main() {
    int n, m;
    if (scanf("%d %d", &n, &m) != 2) return 0;
    vector<int> eu(m), ev(m);
    vector<ll> ew(m);
    vector<vector<pair<int,ll>>> g(n + 1);
    vector<ll> wsum(n + 1, 0);
    for (int i = 0; i < m; i++) {
        int u, v; ll w; scanf("%d %d %lld", &u, &v, &w);
        eu[i] = u; ev[i] = v; ew[i] = w;
        g[u].push_back({v, w}); g[v].push_back({u, w});
        wsum[u] += w; wsum[v] += w;
    }

    vector<int> order(n);
    for (int i = 0; i < n; i++) order[i] = i + 1;
    sort(order.begin(), order.end(), [&](int x, int y){ return wsum[x] > wsum[y]; });

    vector<int> assign(n + 1, -1);
    int cap0 = n / 2, cap1 = n / 2;
    for (int x : order) {
        // gain of putting x on loop 0 = strength to already-placed neighbors on loop 1
        // gain of putting x on loop 1 = strength to already-placed neighbors on loop 0
        ll g0 = 0, g1 = 0;
        for (auto& e : g[x]) {
            if (assign[e.first] == 1) g0 += e.second;
            else if (assign[e.first] == 0) g1 += e.second;
        }
        int side;
        if (cap0 == 0) side = 1;
        else if (cap1 == 0) side = 0;
        else if (g0 > g1) side = 0;
        else if (g1 > g0) side = 1;
        else side = (cap0 >= cap1) ? 0 : 1; // tie: balance-favoring
        assign[x] = side;
        if (side == 0) cap0--; else cap1--;
    }

    for (int i = 1; i <= n; i++) printf("%d\n", assign[i]);
    return 0;
}
