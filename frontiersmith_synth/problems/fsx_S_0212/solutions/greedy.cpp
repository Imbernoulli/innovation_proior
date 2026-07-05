// TIER: greedy
// Sequential greedy: place cells one at a time (in a fixed shuffled order); assign each
// cell to the fleet OPPOSITE its heavier already-placed neighbor weight (maximizing the
// incremental cut), subject to each fleet holding at most n/2 cells. Result is exactly
// balanced (difference 0), hence always feasible.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main() {
    int n, m, D;
    if (scanf("%d %d %d", &n, &m, &D) != 3) return 0;
    vector<vector<pair<int,ll>>> adj(n + 1);
    for (int i = 0; i < m; i++) {
        int u, v; ll w; scanf("%d %d %lld", &u, &v, &w);
        adj[u].push_back({v, w});
        adj[v].push_back({u, w});
    }

    vector<int> order(n);
    for (int i = 0; i < n; i++) order[i] = i + 1;
    mt19937 rng(12345u);
    shuffle(order.begin(), order.end(), rng);

    vector<int> x(n + 1, -1);
    int cap = n / 2, c0 = 0, c1 = 0;
    for (int u : order) {
        ll s0 = 0, s1 = 0; // weight to already-placed neighbors on each side
        for (auto& e : adj[u]) {
            if (x[e.first] == 0) s0 += e.second;
            else if (x[e.first] == 1) s1 += e.second;
        }
        // want u opposite the heavier side to cut that weight
        int want = (s0 >= s1) ? 1 : 0;
        if (want == 0) {
            if (c0 < cap) { x[u] = 0; c0++; }
            else { x[u] = 1; c1++; }
        } else {
            if (c1 < cap) { x[u] = 1; c1++; }
            else { x[u] = 0; c0++; }
        }
    }
    for (int i = 1; i <= n; i++) printf("%d\n", x[i]);
    return 0;
}
