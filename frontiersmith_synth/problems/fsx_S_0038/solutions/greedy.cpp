// TIER: greedy
// Cost-blind max-coverage greedy: repeatedly build where it newly monitors the
// most currently-unmonitored cells (ties broken by smallest index). Minimizes
// the NUMBER of towers, ignoring installation cost.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int N, M, r;
    if (scanf("%d %d %d", &N, &M, &r) != 3) return 0;
    vector<long long> c(N + 1);
    for (int i = 1; i <= N; i++) scanf("%lld", &c[i]);
    vector<vector<int>> adj(N + 1);
    for (int e = 0; e < M; e++) { int u, v; scanf("%d %d", &u, &v); adj[u].push_back(v); adj[v].push_back(u); }

    // coverage set of each candidate site (cells within radius r)
    vector<vector<int>> cover(N + 1);
    vector<int> dist(N + 1, -1);
    for (int s = 1; s <= N; s++) {
        vector<int> touched;
        dist[s] = 0; queue<int> q; q.push(s); touched.push_back(s);
        while (!q.empty()) {
            int u = q.front(); q.pop();
            cover[s].push_back(u);
            if (dist[u] == r) continue;
            for (int v : adj[u]) if (dist[v] == -1) { dist[v] = dist[u] + 1; q.push(v); touched.push_back(v); }
        }
        for (int u : touched) dist[u] = -1;
    }

    vector<char> covered(N + 1, 0);
    int need = N;
    vector<int> sel;
    while (need > 0) {
        int best = -1, bestcnt = -1;
        for (int t = 1; t <= N; t++) {
            int cnt = 0;
            for (int u : cover[t]) if (!covered[u]) cnt++;
            if (cnt > bestcnt) { bestcnt = cnt; best = t; }
        }
        sel.push_back(best);
        for (int u : cover[best]) if (!covered[u]) { covered[u] = 1; need--; }
    }

    printf("%d\n", (int)sel.size());
    for (int x : sel) printf("%d\n", x);
    return 0;
}
