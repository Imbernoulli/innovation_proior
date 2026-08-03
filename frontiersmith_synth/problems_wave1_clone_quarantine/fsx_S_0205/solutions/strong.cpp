// TIER: strong
// Best-improvement local search on the true multi-sink objective. Each round, tentatively
// delete every still-available edge, recompute F = sum of pump-to-tank distances with a
// single Dijkstra (rejecting any move that strands a tank), and commit the edge giving the
// largest increase. Repeat until the budget is exhausted or no move helps.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int n, m, s, p, k;
struct AdjE { int to; ll w; int idx; };
vector<vector<AdjE>> g;
vector<int> tanks;

vector<ll> dbuf;
// F = sum of dist(s, tank); -1 if any tank unreachable.
ll sumF(const vector<char>& removed) {
    dbuf.assign(n + 1, LLONG_MAX);
    priority_queue<pair<ll,int>, vector<pair<ll,int>>, greater<pair<ll,int>>> pq;
    dbuf[s] = 0; pq.push({0, s});
    while (!pq.empty()) {
        auto [d, u] = pq.top(); pq.pop();
        if (d > dbuf[u]) continue;
        for (auto& e : g[u]) {
            if (removed[e.idx]) continue;
            ll nd = d + e.w;
            if (nd < dbuf[e.to]) { dbuf[e.to] = nd; pq.push({nd, e.to}); }
        }
    }
    ll tot = 0;
    for (int u : tanks) {
        if (dbuf[u] == LLONG_MAX) return -1;
        tot += dbuf[u];
    }
    return tot;
}

int main() {
    scanf("%d %d %d %d %d", &n, &m, &s, &p, &k);
    g.assign(n + 1, {});
    for (int i = 1; i <= m; i++) {
        int u, v; ll w; scanf("%d %d %lld", &u, &v, &w);
        g[u].push_back({v, w, i});
        g[v].push_back({u, w, i});
    }
    tanks.resize(p);
    for (int i = 0; i < p; i++) scanf("%d", &tanks[i]);

    vector<char> removed(m + 1, 0);
    ll curF = sumF(removed);
    int used = 0;

    while (used < k) {
        int bestE = -1;
        ll bestF = curF;
        for (int e = 1; e <= m; e++) {
            if (removed[e]) continue;
            removed[e] = 1;
            ll f = sumF(removed);
            removed[e] = 0;
            if (f > bestF) { bestF = f; bestE = e; }
        }
        if (bestE == -1) break;      // no feasible improving move
        removed[bestE] = 1;
        curF = bestF;
        used++;
    }

    vector<int> out;
    for (int i = 1; i <= m; i++) if (removed[i]) out.push_back(i);
    printf("%d\n", (int)out.size());
    for (int e : out) printf("%d\n", e);
    return 0;
}
