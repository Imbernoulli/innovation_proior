// TIER: greedy
// Weight-ordered one-pass greedy: order rigs by descending total incident weight,
// then assign each rig the channel minimizing added penalty vs already-assigned
// neighbours. Single pass, no revisiting.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main() {
    int n, m, C;
    if (scanf("%d %d %d", &n, &m, &C) != 3) return 0;
    vector<vector<array<int,3>>> adj(n + 1); // (nbr, p, q)
    vector<ll> wsum(n + 1, 0);
    for (int i = 0; i < m; i++) {
        int u, v, p, q; scanf("%d %d %d %d", &u, &v, &p, &q);
        adj[u].push_back({v, p, q});
        adj[v].push_back({u, p, q});
        wsum[u] += p; wsum[v] += p;
    }

    vector<int> order(n);
    for (int i = 0; i < n; i++) order[i] = i + 1;
    sort(order.begin(), order.end(), [&](int a, int b){ return wsum[a] > wsum[b]; });

    vector<int> col(n + 1, 0); // 0 = unassigned
    for (int idx = 0; idx < n; idx++) {
        int u = order[idx];
        ll best = LLONG_MAX; int bestc = 1;
        for (int ch = 1; ch <= C; ch++) {
            ll add = 0;
            for (auto& e : adj[u]) {
                int nc = col[e[0]];
                if (nc == 0) continue;
                int diff = abs(nc - ch);
                if (diff == 0) add += e[1];
                else if (diff == 1) add += e[2];
            }
            if (add < best) { best = add; bestc = ch; }
        }
        col[u] = bestc;
    }

    for (int i = 1; i <= n; i++) printf("%d%c", col[i], i == n ? '\n' : ' ');
    return 0;
}
