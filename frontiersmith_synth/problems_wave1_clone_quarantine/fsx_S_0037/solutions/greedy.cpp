// TIER: greedy
// Cost-greedy: repeatedly barricade the CHEAPEST affordable walkway lying on the
// current shortest stroll, as long as the finale stays reachable and the budget holds.
// Ignores how much delay a closure actually buys -> a fast one-pass heuristic.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int n, m, s, t;
ll P;
struct AdjE { int to; ll w; int idx; };
vector<vector<AdjE>> g;
vector<ll> ecost;

// returns shortest dist and fills path-edge indices (one shortest s-t path) or -1
ll dijkstraPath(const vector<char>& removed, vector<int>& pathEdges) {
    vector<ll> dist(n + 1, LLONG_MAX);
    vector<int> preEdge(n + 1, -1), preNode(n + 1, -1);
    priority_queue<pair<ll,int>, vector<pair<ll,int>>, greater<pair<ll,int>>> pq;
    dist[s] = 0; pq.push({0, s});
    while (!pq.empty()) {
        auto [d, u] = pq.top(); pq.pop();
        if (d > dist[u]) continue;
        for (auto& e : g[u]) {
            if (removed[e.idx]) continue;
            ll nd = d + e.w;
            if (nd < dist[e.to]) {
                dist[e.to] = nd; preEdge[e.to] = e.idx; preNode[e.to] = u;
                pq.push({nd, e.to});
            }
        }
    }
    pathEdges.clear();
    if (dist[t] == LLONG_MAX) return -1;
    int cur = t;
    while (cur != s) { pathEdges.push_back(preEdge[cur]); cur = preNode[cur]; }
    return dist[t];
}

bool connected(const vector<char>& removed) {
    vector<int> pe;
    return dijkstraPath(removed, pe) != -1;
}

int main() {
    int ss, tt, PP;
    if (scanf("%d %d %d %d %d", &n, &m, &s, &t, &PP) != 5) return 0;
    P = PP;
    g.assign(n + 1, {});
    ecost.assign(m + 1, 0);
    for (int i = 1; i <= m; i++) {
        int u, v, w, c; scanf("%d %d %d %d", &u, &v, &w, &c);
        ecost[i] = c;
        g[u].push_back({v, w, i});
        g[v].push_back({u, w, i});
    }

    vector<char> removed(m + 1, 0);
    ll budget = P;
    vector<int> chosen;

    while (true) {
        vector<int> pe;
        if (dijkstraPath(removed, pe) == -1) break;
        // cheapest affordable path edge that keeps connectivity
        int best = -1; ll bestC = LLONG_MAX;
        for (int idx : pe) {
            if (removed[idx]) continue;
            if (ecost[idx] > budget) continue;
            if (ecost[idx] < bestC) {
                removed[idx] = 1;
                bool ok = connected(removed);
                removed[idx] = 0;
                if (ok) { bestC = ecost[idx]; best = idx; }
            }
        }
        if (best == -1) break;
        removed[best] = 1; budget -= ecost[best]; chosen.push_back(best);
    }

    printf("%d\n", (int)chosen.size());
    for (int idx : chosen) printf("%d\n", idx);
    return 0;
}
