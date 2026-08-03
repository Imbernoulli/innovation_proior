// TIER: greedy
// Traffic-weighted one-pass greedy: build a single shortest-path tree from the
// hive, score each tree edge by (number of forage patches whose route uses it) x
// (edge cost), then destroy the highest-scoring corridors that keep every patch
// reachable. No re-computation of the tree between removals.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int n, m, s, q, k;
struct AdjE { int to; ll w; int idx; };
vector<vector<AdjE>> g;
vector<int> eu, ev; vector<ll> ew;
vector<int> term;

vector<ll> dijkstra(const vector<char>& removed, vector<int>* parentEdge = nullptr) {
    vector<ll> dist(n + 1, LLONG_MAX);
    if (parentEdge) parentEdge->assign(n + 1, 0);
    priority_queue<pair<ll,int>, vector<pair<ll,int>>, greater<pair<ll,int>>> pq;
    dist[s] = 0; pq.push({0, s});
    while (!pq.empty()) {
        auto [d, u] = pq.top(); pq.pop();
        if (d > dist[u]) continue;
        for (auto& e : g[u]) {
            if (removed[e.idx]) continue;
            ll nd = d + e.w;
            if (nd < dist[e.to]) {
                dist[e.to] = nd;
                if (parentEdge) (*parentEdge)[e.to] = e.idx;
                pq.push({nd, e.to});
            }
        }
    }
    return dist;
}

bool allReachable(const vector<char>& removed) {
    vector<ll> d = dijkstra(removed);
    for (int t : term) if (d[t] == LLONG_MAX) return false;
    return true;
}

int main() {
    scanf("%d %d %d %d %d", &n, &m, &s, &q, &k);
    g.assign(n + 1, {});
    eu.assign(m + 1, 0); ev.assign(m + 1, 0); ew.assign(m + 1, 0);
    for (int i = 1; i <= m; i++) {
        int u, v; ll w; scanf("%d %d %lld", &u, &v, &w);
        eu[i] = u; ev[i] = v; ew[i] = w;
        g[u].push_back({v, w, i});
        g[v].push_back({u, w, i});
    }
    term.assign(q, 0);
    for (int i = 0; i < q; i++) scanf("%d", &term[i]);

    vector<char> removed(m + 1, 0);
    vector<int> parent;
    dijkstra(removed, &parent);

    // traffic per tree edge
    vector<ll> traffic(m + 1, 0);
    for (int t : term) {
        int cur = t;
        while (cur != s && parent[cur] != 0) {
            int e = parent[cur];
            traffic[e]++;
            cur = (eu[e] == cur) ? ev[e] : eu[e];
        }
    }
    // rank candidate edges by (traffic * cost) descending
    vector<int> cand;
    for (int e = 1; e <= m; e++) if (traffic[e] > 0) cand.push_back(e);
    sort(cand.begin(), cand.end(), [&](int a, int b) {
        ll sa = traffic[a] * ew[a], sb = traffic[b] * ew[b];
        if (sa != sb) return sa > sb;
        return a < b;
    });

    int destroyed = 0;
    vector<int> chosen;
    for (int e : cand) {
        if (destroyed >= k) break;
        removed[e] = 1;
        if (allReachable(removed)) { chosen.push_back(e); destroyed++; }
        else removed[e] = 0;
    }

    printf("%d\n", (int)chosen.size());
    for (int e : chosen) printf("%d\n", e);
    return 0;
}
