// TIER: greedy
// Coverage-only greedy: repeatedly build the relay that illuminates the most
// still-dark systems (ties broken by lower cost). Ignores the cost/coverage
// trade-off, so it tends to overspend relative to the cost-aware strong solution.
#include <bits/stdc++.h>
using namespace std;

int N, M, r;
vector<vector<int>> g;
vector<int> cost;

// nodes within r hops of src (inclusive)
vector<int> ball(int src) {
    vector<int> dist(N + 1, -1);
    deque<int> q; dist[src] = 0; q.push_back(src);
    vector<int> res;
    while (!q.empty()) {
        int u = q.front(); q.pop_front();
        res.push_back(u);
        if (dist[u] >= r) continue;
        for (int v : g[u]) if (dist[v] < 0) { dist[v] = dist[u] + 1; q.push_back(v); }
    }
    return res;
}

int main() {
    scanf("%d %d %d", &N, &M, &r);
    g.assign(N + 1, {});
    for (int i = 0; i < M; i++) { int u, v; scanf("%d %d", &u, &v); g[u].push_back(v); g[v].push_back(u); }
    cost.assign(N + 1, 0);
    for (int i = 1; i <= N; i++) scanf("%d", &cost[i]);

    vector<vector<int>> B(N + 1);
    for (int i = 1; i <= N; i++) B[i] = ball(i);

    vector<char> covered(N + 1, 0);
    int remaining = N;
    vector<int> chosen;
    while (remaining > 0) {
        int best = -1, bestNew = -1; long bestCost = 0;
        for (int i = 1; i <= N; i++) {
            int nw = 0;
            for (int u : B[i]) if (!covered[u]) nw++;
            if (nw == 0) continue;
            if (nw > bestNew || (nw == bestNew && cost[i] < bestCost)) {
                bestNew = nw; best = i; bestCost = cost[i];
            }
        }
        chosen.push_back(best);
        for (int u : B[best]) if (!covered[u]) { covered[u] = 1; remaining--; }
    }

    printf("%d\n", (int)chosen.size());
    for (int x : chosen) printf("%d\n", x);
    return 0;
}
