// TIER: greedy
// Weighted greedy set cover on graph-radius coverage: repeatedly build the station
// minimizing (cost / number of newly-monitored demand blocks).
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int n, m, D, P;
struct AdjE { int to; ll w; };
vector<vector<AdjE>> g;
vector<int> demIdOf;                 // node -> demand id (1..D) or 0
vector<int> stNode, stCost, stRad;
vector<vector<int>> cover;           // station -> list of covered demand ids

void computeCoverage() {
    cover.assign(P + 1, {});
    for (int i = 1; i <= P; i++) {
        int src = stNode[i]; ll rad = stRad[i];
        vector<ll> dist(n + 1, LLONG_MAX);
        priority_queue<pair<ll,int>, vector<pair<ll,int>>, greater<pair<ll,int>>> pq;
        dist[src] = 0; pq.push({0, src});
        while (!pq.empty()) {
            auto [d, u] = pq.top(); pq.pop();
            if (d > dist[u]) continue;
            if (d > rad) continue;
            if (demIdOf[u]) cover[i].push_back(demIdOf[u]);
            for (auto& e : g[u]) {
                ll nd = d + e.w;
                if (nd <= rad && nd < dist[e.to]) { dist[e.to] = nd; pq.push({nd, e.to}); }
            }
        }
    }
}

int main() {
    if (scanf("%d %d", &n, &m) != 2) return 0;
    g.assign(n + 1, {});
    for (int i = 0; i < m; i++) {
        int u, v; ll w; scanf("%d %d %lld", &u, &v, &w);
        g[u].push_back({v, w}); g[v].push_back({u, w});
    }
    scanf("%d", &D);
    demIdOf.assign(n + 1, 0);
    for (int i = 1; i <= D; i++) { int x; scanf("%d", &x); demIdOf[x] = i; }
    scanf("%d", &P);
    stNode.assign(P + 1, 0); stCost.assign(P + 1, 0); stRad.assign(P + 1, 0);
    for (int i = 1; i <= P; i++) scanf("%d %d %d", &stNode[i], &stCost[i], &stRad[i]);

    computeCoverage();

    vector<char> covered(D + 1, 0);
    vector<char> chosen(P + 1, 0);
    int remaining = D;
    vector<int> pick;
    while (remaining > 0) {
        int best = -1; double bestRatio = 1e18;
        for (int i = 1; i <= P; i++) {
            if (chosen[i]) continue;
            int neu = 0;
            for (int d : cover[i]) if (!covered[d]) neu++;
            if (neu == 0) continue;
            double ratio = (double)stCost[i] / (double)neu;
            if (ratio < bestRatio - 1e-12) { bestRatio = ratio; best = i; }
        }
        if (best == -1) break; // should not happen (build-all covers all)
        chosen[best] = 1; pick.push_back(best);
        for (int d : cover[best]) if (!covered[d]) { covered[d] = 1; remaining--; }
    }

    printf("%d\n", (int)pick.size());
    for (int i : pick) printf("%d\n", i);
    return 0;
}
