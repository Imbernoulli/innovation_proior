// TIER: greedy
// Greedy tree-edge cutting: build one shortest-path tree from the pump, collect the
// edges lying on the tree paths to the tanks, and shut the heaviest of them (up to k)
// as long as every tank stays connected. One-pass proxy; weaker than true local search.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int n, m, s, p, k;
struct AdjE { int to; ll w; int idx; };
vector<vector<AdjE>> g;
vector<int> tanks;

// Dijkstra returning dist and, for each node, the edge index used to reach it.
void dijkstra(const vector<char>& removed, vector<ll>& dist, vector<int>& pe) {
    dist.assign(n + 1, LLONG_MAX);
    pe.assign(n + 1, -1);
    priority_queue<pair<ll,int>, vector<pair<ll,int>>, greater<pair<ll,int>>> pq;
    dist[s] = 0; pq.push({0, s});
    while (!pq.empty()) {
        auto [d, u] = pq.top(); pq.pop();
        if (d > dist[u]) continue;
        for (auto& e : g[u]) {
            if (removed[e.idx]) continue;
            ll nd = d + e.w;
            if (nd < dist[e.to]) { dist[e.to] = nd; pe[e.to] = e.idx; pq.push({nd, e.to}); }
        }
    }
}

bool allTanksReachable(const vector<char>& removed) {
    vector<ll> dist; vector<int> pe;
    dijkstra(removed, dist, pe);
    for (int u : tanks) if (dist[u] == LLONG_MAX) return false;
    return true;
}

int main() {
    scanf("%d %d %d %d %d", &n, &m, &s, &p, &k);
    g.assign(n + 1, {});
    vector<ll> ew(m + 1);
    for (int i = 1; i <= m; i++) {
        int u, v; ll w; scanf("%d %d %lld", &u, &v, &w);
        g[u].push_back({v, w, i});
        g[v].push_back({u, w, i});
        ew[i] = w;
    }
    tanks.resize(p);
    for (int i = 0; i < p; i++) scanf("%d", &tanks[i]);

    // shortest-path tree from the pump
    vector<ll> dist; vector<int> pe;
    dijkstra(vector<char>(m + 1, 0), dist, pe);

    // collect tree edges on the paths to the tanks (walk parent pointers)
    set<int> cand;
    for (int u : tanks) {
        int cur = u;
        while (cur != s && pe[cur] != -1) {
            cand.insert(pe[cur]);
            // move to the other endpoint of the parent edge
            int nxt = -1;
            for (auto& e : g[cur]) if (e.idx == pe[cur]) { nxt = e.to; break; }
            if (nxt == -1 || nxt == cur) break;
            cur = nxt;
        }
    }

    // sort candidates by pipe cost, heaviest first
    vector<int> order(cand.begin(), cand.end());
    sort(order.begin(), order.end(), [&](int a, int b){ return ew[a] > ew[b]; });

    vector<char> removed(m + 1, 0);
    int used = 0;
    for (int e : order) {
        if (used >= k) break;
        removed[e] = 1;
        if (allTanksReachable(removed)) { used++; }
        else removed[e] = 0;   // would strand a tank; skip
    }

    vector<int> out;
    for (int i = 1; i <= m; i++) if (removed[i]) out.push_back(i);
    printf("%d\n", (int)out.size());
    for (int e : out) printf("%d\n", e);
    return 0;
}
