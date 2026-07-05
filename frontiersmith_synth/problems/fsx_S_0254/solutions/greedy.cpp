// TIER: greedy
// Coverage-only greedy set cover: repeatedly place a sensor at the pool that newly
// covers the MOST currently-uncovered pools (ignoring cost). One pass, lazy-evaluated.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int N, M; ll R;
vector<ll> cost;
struct E { int to; ll w; };
vector<vector<E>> g;
vector<vector<int>> ball; // ball[v] = pools within radius R of v

void buildBalls() {
    ball.assign(N + 1, {});
    vector<ll> dist(N + 1, -1);
    priority_queue<pair<ll,int>, vector<pair<ll,int>>, greater<pair<ll,int>>> pq;
    for (int s = 1; s <= N; s++) {
        vector<int> touched;
        dist[s] = 0; pq.push({0, s});
        while (!pq.empty()) {
            auto [d, u] = pq.top(); pq.pop();
            if (d != dist[u]) continue;
            touched.push_back(u);
            ball[s].push_back(u);
            for (auto& e : g[u]) {
                ll nd = d + e.w;
                if (nd <= R && (dist[e.to] < 0 || nd < dist[e.to])) { dist[e.to] = nd; pq.push({nd, e.to}); }
            }
        }
        for (int u : touched) dist[u] = -1;
    }
}

int main() {
    scanf("%d %d %lld", &N, &M, &R);
    cost.assign(N + 1, 0);
    for (int v = 1; v <= N; v++) scanf("%lld", &cost[v]);
    g.assign(N + 1, {});
    for (int i = 0; i < M; i++) {
        int u, v; ll w; scanf("%d %d %lld", &u, &v, &w);
        g[u].push_back({v, w}); g[v].push_back({u, w});
    }
    buildBalls();

    vector<char> covered(N + 1, 0);
    int remaining = N;
    // lazy max-heap keyed by (upper-bound gain)
    priority_queue<pair<int,int>> pq;
    for (int v = 1; v <= N; v++) pq.push({(int)ball[v].size(), v});
    vector<char> chosen(N + 1, 0);
    vector<int> sol;
    while (remaining > 0 && !pq.empty()) {
        auto [key, v] = pq.top(); pq.pop();
        if (chosen[v]) continue;
        int real = 0;
        for (int u : ball[v]) if (!covered[u]) real++;
        if (real < key) { pq.push({real, v}); continue; }
        if (real == 0) break;
        chosen[v] = 1; sol.push_back(v);
        for (int u : ball[v]) if (!covered[u]) { covered[u] = 1; remaining--; }
    }
    printf("%d\n", (int)sol.size());
    for (int v : sol) printf("%d\n", v);
    return 0;
}
