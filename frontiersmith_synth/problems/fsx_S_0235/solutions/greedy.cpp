// TIER: greedy
// Shortest-path star: one Dijkstra from the first hut; dig the union of
// shortest-path trenches connecting it to every other hut.
#include <bits/stdc++.h>
using namespace std;
int main() {
    int N, M, K;
    if (scanf("%d %d %d", &N, &M, &K) != 3) return 0;
    vector<int> eu(M + 1), ev(M + 1); vector<long long> ew(M + 1);
    vector<vector<pair<int,int>>> adj(N + 1); // (nbr, edgeIdx)
    for (int k = 1; k <= M; k++) {
        scanf("%d %d %lld", &eu[k], &ev[k], &ew[k]);
        adj[eu[k]].push_back({ev[k], k});
        adj[ev[k]].push_back({eu[k], k});
    }
    vector<int> huts(K);
    for (int i = 0; i < K; i++) scanf("%d", &huts[i]);

    int src = huts[0];
    const long long INF = (long long)4e18;
    vector<long long> dist(N + 1, INF);
    vector<int> parEdge(N + 1, -1);
    priority_queue<pair<long long,int>, vector<pair<long long,int>>, greater<>> pq;
    dist[src] = 0; pq.push({0, src});
    while (!pq.empty()) {
        auto [d, u] = pq.top(); pq.pop();
        if (d != dist[u]) continue;
        for (auto [v, idx] : adj[u]) {
            long long nd = d + ew[idx];
            if (nd < dist[v]) { dist[v] = nd; parEdge[v] = idx; pq.push({nd, v}); }
        }
    }

    set<int> chosen;
    for (int i = 1; i < K; i++) {
        int cur = huts[i];
        while (cur != src && parEdge[cur] != -1) {
            int idx = parEdge[cur];
            chosen.insert(idx);
            cur = (eu[idx] == cur) ? ev[idx] : eu[idx];
        }
    }

    printf("%d\n", (int)chosen.size());
    bool first = true;
    for (int idx : chosen) { printf("%s%d", first ? "" : " ", idx); first = false; }
    printf("\n");
    return 0;
}
