// TIER: greedy
// Sequential greedy labeling: process pools 1..n in order; assign each pool the niche
// that minimizes added conflict weight to its already-labeled neighbors. Single pass.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n, m, C;
    if (scanf("%d %d %d", &n, &m, &C) != 3) return 0;
    vector<vector<pair<int,int>>> adj(n + 1); // node -> (nbr, weight)
    for (int i = 0; i < m; i++) {
        int u, v, w; scanf("%d %d %d", &u, &v, &w);
        adj[u].push_back({v, w});
        adj[v].push_back({u, w});
    }
    vector<int> lab(n + 1, 0); // 0 = unlabeled
    vector<long long> cost(C + 1);
    for (int i = 1; i <= n; i++) {
        for (int c = 1; c <= C; c++) cost[c] = 0;
        for (auto& e : adj[i]) {
            int j = e.first;
            if (lab[j] != 0) cost[lab[j]] += e.second;
        }
        int best = 1; long long bv = cost[1];
        for (int c = 2; c <= C; c++)
            if (cost[c] < bv) { bv = cost[c]; best = c; }
        lab[i] = best;
    }
    // print
    string out;
    out.reserve(n * 2);
    char buf[16];
    for (int i = 1; i <= n; i++) { int len = sprintf(buf, "%d\n", lab[i]); out.append(buf, len); }
    fputs(out.c_str(), stdout);
    return 0;
}
