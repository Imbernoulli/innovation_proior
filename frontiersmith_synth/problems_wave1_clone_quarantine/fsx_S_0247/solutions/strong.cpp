// TIER: strong
// Degree-constrained Steiner tree. (1) Kruskal over ALL locations (poles + cabinets) with
// port-cap limits builds a maximal degree-constrained spanning forest. (2) Repair: connect
// remaining components via spare-capacity nodes (leaves of a cap>=2 tree always have spare)
// into one spanning tree. (3) Prune: repeatedly delete junction-cabinet leaves so cabinets
// survive only where they shorten routing. Result spans all light poles, respects caps, and
// exploits junctions -- beating the pole-only heuristics.
#include <bits/stdc++.h>
using namespace std;

struct DSU {
    vector<int> p;
    DSU(int n) : p(n) { for (int i = 0; i < n; i++) p[i] = i; }
    int find(int x) { while (p[x] != x) { p[x] = p[p[x]]; x = p[x]; } return x; }
    bool unite(int a, int b) { a = find(a); b = find(b); if (a == b) return false; p[a] = b; return true; }
};

int main() {
    int N;
    if (scanf("%d", &N) != 1) return 0;
    vector<long long> X(N), Y(N);
    vector<int> term(N), cap(N);
    for (int i = 0; i < N; i++) {
        long long x, y, t, c;
        scanf("%lld %lld %lld %lld", &x, &y, &t, &c);
        X[i] = x; Y[i] = y; term[i] = (int)t; cap[i] = (int)c;
    }
    if (N <= 1) { printf("0\n"); return 0; }

    auto len2 = [&](int a, int b) {
        long long dx = X[a] - X[b], dy = Y[a] - Y[b];
        return dx * dx + dy * dy;
    };

    // candidate edges: complete graph (N<=600 -> <=180k edges)
    vector<tuple<long long,int,int>> edges;
    edges.reserve((size_t)N * (N - 1) / 2);
    for (int i = 0; i < N; i++)
        for (int j = i + 1; j < N; j++)
            edges.push_back({len2(i, j), i, j});
    sort(edges.begin(), edges.end());

    vector<int> deg(N, 0);
    vector<set<int>> adj(N);
    DSU dsu(N);

    // (1) degree-constrained Kruskal
    for (auto &e : edges) {
        int u = get<1>(e), v = get<2>(e);
        if (deg[u] >= cap[u] || deg[v] >= cap[v]) continue;
        if (dsu.find(u) == dsu.find(v)) continue;
        dsu.unite(u, v);
        adj[u].insert(v); adj[v].insert(u);
        deg[u]++; deg[v]++;
    }

    // (2) repair: merge remaining components via spare-capacity nodes
    // find a node with spare capacity in each component root; chain components together.
    auto spareInRoot = [&](int r) -> int {
        for (int i = 0; i < N; i++)
            if (dsu.find(i) == r && deg[i] < cap[i]) return i;
        return -1;
    };
    // collect distinct roots
    {
        int mainNode = 0;
        int mainRoot = dsu.find(mainNode);
        for (int i = 0; i < N; i++) {
            if (dsu.find(i) == mainRoot) continue;
            int r = dsu.find(i);
            int a = spareInRoot(mainRoot);
            int b = spareInRoot(r);
            if (a < 0 || b < 0) continue; // should not happen with cap>=2
            dsu.unite(a, b);
            adj[a].insert(b); adj[b].insert(a);
            deg[a]++; deg[b]++;
            mainRoot = dsu.find(mainNode);
        }
    }

    // (3) prune junction-cabinet leaves (t==0, degree 1)
    queue<int> q;
    for (int i = 0; i < N; i++)
        if (!term[i] && (int)adj[i].size() == 1) q.push(i);
    while (!q.empty()) {
        int u = q.front(); q.pop();
        if (term[u] || (int)adj[u].size() != 1) continue;
        int w = *adj[u].begin();
        adj[u].erase(w);
        adj[w].erase(u);
        if (!term[w] && (int)adj[w].size() == 1) q.push(w);
    }

    // emit remaining edges
    vector<pair<int,int>> out;
    for (int u = 0; u < N; u++)
        for (int w : adj[u])
            if (u < w) out.push_back({u + 1, w + 1});

    printf("%d\n", (int)out.size());
    for (auto &e : out) printf("%d %d\n", e.first, e.second);
    return 0;
}
