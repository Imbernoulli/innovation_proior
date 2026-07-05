// TIER: strong
// Best-improvement local search: for up to k rounds, recompute all shortest
// paths, gather the current shortest-path-tree edges that carry forage traffic,
// and for each candidate compute the TRUE objective gain of destroying it (a fresh
// Dijkstra), respecting the reachability constraint. Commit the single best
// feasible destruction each round. This exact marginal search dominates the
// static traffic-weighted greedy.
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

// returns total effort, or -1 if some terminal unreachable
ll totalEffort(const vector<char>& removed, vector<int>* parentEdge = nullptr) {
    vector<ll> d = dijkstra(removed, parentEdge);
    ll F = 0;
    for (int t : term) { if (d[t] == LLONG_MAX) return -1; F += d[t]; }
    return F;
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
    vector<int> chosen;

    for (int round = 0; round < k; round++) {
        vector<int> parent;
        ll curF = totalEffort(removed, &parent);
        if (curF < 0) break; // should not happen (we keep feasibility)

        // candidate edges: shortest-path-tree edges on the routes to terminals
        vector<char> isCand(m + 1, 0);
        vector<int> cand;
        for (int t : term) {
            int cur = t;
            while (cur != s && parent[cur] != 0) {
                int e = parent[cur];
                if (!removed[e] && !isCand[e]) { isCand[e] = 1; cand.push_back(e); }
                cur = (eu[e] == cur) ? ev[e] : eu[e];
            }
        }

        ll bestGain = 0; int bestEdge = -1;
        for (int e : cand) {
            removed[e] = 1;
            ll f = totalEffort(removed);
            removed[e] = 0;
            if (f < 0) continue;              // would disconnect a patch
            ll gain = f - curF;
            if (gain > bestGain) { bestGain = gain; bestEdge = e; }
        }

        if (bestEdge == -1) break;            // no improving feasible destruction
        removed[bestEdge] = 1;
        chosen.push_back(bestEdge);
    }

    printf("%d\n", (int)chosen.size());
    for (int e : chosen) printf("%d\n", e);
    return 0;
}
