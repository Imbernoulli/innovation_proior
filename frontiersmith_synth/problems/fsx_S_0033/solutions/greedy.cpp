// TIER: greedy
// One-pass min-conflict greedy: process galleries in id order; assign each the
// channel that adds the least annoyance with its ALREADY-assigned overlapping
// neighbors (ties -> lowest channel). No revisiting.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main() {
    int n, k, m;
    if (scanf("%d %d %d", &n, &k, &m) != 3) return 0;
    vector<vector<pair<int,int>>> adj(n + 1); // node -> (neighbor, weight)
    for (int i = 0; i < m; i++) {
        int u, v, w; if (scanf("%d %d %d", &u, &v, &w) != 3) break;
        adj[u].push_back({v, w});
        adj[v].push_back({u, w});
    }
    vector<int> c(n + 1, 0); // 0 = unassigned
    vector<ll> cost(k + 1);
    for (int g = 1; g <= n; g++) {
        fill(cost.begin(), cost.end(), 0);
        for (auto& e : adj[g]) {
            int nb = e.first;
            if (c[nb] != 0) cost[c[nb]] += e.second;
        }
        int best = 1; ll bestCost = cost[1];
        for (int ch = 2; ch <= k; ch++)
            if (cost[ch] < bestCost) { bestCost = cost[ch]; best = ch; }
        c[g] = best;
    }
    for (int g = 1; g <= n; g++) printf("%d%c", c[g], g == n ? '\n' : ' ');
    return 0;
}
