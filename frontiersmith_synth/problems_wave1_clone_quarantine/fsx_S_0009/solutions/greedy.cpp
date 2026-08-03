// TIER: greedy
// One-pass greedy: color zones 1..N in index order; each zone takes the program that
// adds the least conflict with its already-colored coupled neighbors.
#include <bits/stdc++.h>
using namespace std;
int main() {
    int N, M, K;
    if (scanf("%d %d %d", &N, &M, &K) != 3) return 0;
    vector<vector<pair<int,long long>>> adj(N + 1);
    for (int i = 0; i < M; i++) {
        int u, v; long long w;
        if (scanf("%d %d %lld", &u, &v, &w) != 3) break;
        adj[u].push_back({v, w});
        adj[v].push_back({u, w});
    }
    vector<int> col(N + 1, 0);   // 0 = uncolored
    for (int j = 1; j <= N; j++) {
        vector<long long> cost(K + 1, 0);
        for (auto &e : adj[j])
            if (col[e.first] != 0) cost[col[e.first]] += e.second;
        int best = 1; long long bc = cost[1];
        for (int c = 2; c <= K; c++)
            if (cost[c] < bc) { bc = cost[c]; best = c; }
        col[j] = best;
    }
    for (int j = 1; j <= N; j++) printf("%d\n", col[j]);
    return 0;
}
