// TIER: strong
// Fee-aware degree-constrained Kruskal.  Candidate links = each inverter's k nearest
// neighbours (Manhattan).  Each candidate is weighted by trench(a,b) + w_a + w_b, i.e. its
// true marginal cost (one port lit at each endpoint), so short links are preferred AND
// expensive combiners are naturally pushed to be leaves.  Add a link if it joins two
// components and both endpoints still have a free port.  Any components the k-NN pass fails
// to merge are stitched together via spare-capacity nodes (a cap>=2 forest always keeps a
// spare port in every component), guaranteeing a feasible connected spanning tree.
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
    vector<long long> x(N), y(N), w(N);
    vector<int> cap(N);
    for (int i = 0; i < N; i++) {
        int c, ww;
        scanf("%lld %lld %d %d", &x[i], &y[i], &c, &ww);
        cap[i] = c; w[i] = ww;
    }
    if (N == 1) { printf("0\n"); return 0; }

    auto trench = [&](int a, int b) -> long long {
        return llabs(x[a] - x[b]) + llabs(y[a] - y[b]);
    };

    // ---- candidate links: k nearest neighbours of each inverter ----
    int k = min(N - 1, 10);
    struct Edge { long long wt; int a, b; };
    vector<Edge> edges;
    edges.reserve((size_t)N * k);
    vector<pair<long long,int>> dist(N);
    for (int i = 0; i < N; i++) {
        int m = 0;
        for (int j = 0; j < N; j++) {
            if (j == i) continue;
            dist[m++] = {trench(i, j), j};
        }
        int kk = min(k, m);
        partial_sort(dist.begin(), dist.begin() + kk, dist.begin() + m);
        for (int t = 0; t < kk; t++) {
            int j = dist[t].second;
            if (i < j) edges.push_back({dist[t].first + w[i] + w[j], i, j});
            else       edges.push_back({dist[t].first + w[i] + w[j], j, i});
        }
    }
    // dedup (i<j) & sort by marginal cost
    sort(edges.begin(), edges.end(), [](const Edge& p, const Edge& q) {
        if (p.a != q.a) return p.a < q.a;
        if (p.b != q.b) return p.b < q.b;
        return p.wt < q.wt;
    });
    edges.erase(unique(edges.begin(), edges.end(), [](const Edge& p, const Edge& q) {
        return p.a == q.a && p.b == q.b;
    }), edges.end());
    sort(edges.begin(), edges.end(), [](const Edge& p, const Edge& q) { return p.wt < q.wt; });

    DSU dsu(N);
    vector<int> deg(N, 0);
    vector<pair<int,int>> chosen;
    chosen.reserve(N - 1);
    int comps = N;

    for (const auto& e : edges) {
        if (comps == 1) break;
        if (deg[e.a] >= cap[e.a] || deg[e.b] >= cap[e.b]) continue;
        if (dsu.unite(e.a, e.b)) {
            deg[e.a]++; deg[e.b]++;
            chosen.push_back({e.a, e.b});
            comps--;
        }
    }

    // ---- repair: stitch leftover components via spare-capacity nodes ----
    while (comps > 1) {
        // find any node u with a free port
        int u = -1;
        for (int i = 0; i < N; i++) if (deg[i] < cap[i]) { u = i; break; }
        // find a spare node in a different component
        int v = -1;
        int ru = dsu.find(u);
        for (int i = 0; i < N; i++)
            if (deg[i] < cap[i] && dsu.find(i) != ru) { v = i; break; }
        // guaranteed to exist for a cap>=2 forest; add the connecting run
        dsu.unite(u, v);
        deg[u]++; deg[v]++;
        chosen.push_back({u, v});
        comps--;
    }

    printf("%d\n", (int)chosen.size());
    for (auto& e : chosen) printf("%d %d\n", e.first + 1, e.second + 1);
    return 0;
}
