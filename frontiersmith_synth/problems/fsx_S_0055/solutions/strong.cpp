// TIER: strong
// Degree-constrained Kruskal: sort all candidate lifts by elevation-weighted cost and add
// the cheapest whose endpoints are in different components and both still have a free tower
// slot. Then a repair phase merges any leftover components through their free-slot leaves
// (every tree component with cap>=2 has such a leaf), guaranteeing a connected feasible
// network. Produces a near-MST that respects the caps -> beats the nearest-neighbour path.
#include <bits/stdc++.h>
using namespace std;

static vector<long long> X, Y, H;
static inline long long len(int a, int b) {
    double dx = (double)(X[a] - X[b]);
    double dy = (double)(Y[a] - Y[b]);
    double dh = (double)(H[a] - H[b]);
    return (long long)llround(sqrt(dx * dx + dy * dy) + 3.0 * fabs(dh));
}

struct DSU {
    vector<int> p, sz;
    DSU(int n) : p(n), sz(n, 1) { for (int i = 0; i < n; i++) p[i] = i; }
    int find(int x) { while (p[x] != x) { p[x] = p[p[x]]; x = p[x]; } return x; }
    bool unite(int a, int b) {
        a = find(a); b = find(b);
        if (a == b) return false;
        if (sz[a] < sz[b]) swap(a, b);
        p[b] = a; sz[a] += sz[b];
        return true;
    }
};

int main() {
    int N;
    if (scanf("%d", &N) != 1) return 0;
    X.resize(N); Y.resize(N); H.resize(N);
    vector<int> cap(N);
    for (int i = 0; i < N; i++) scanf("%lld %lld %lld %d", &X[i], &Y[i], &H[i], &cap[i]);
    if (N <= 1) { printf("0\n"); return 0; }

    // candidate edges: all pairs (N <= 600 -> <= ~180k edges)
    struct E { long long w; int u, v; };
    vector<E> es;
    es.reserve((size_t)N * (N - 1) / 2);
    for (int i = 0; i < N; i++)
        for (int j = i + 1; j < N; j++)
            es.push_back({len(i, j), i, j});
    sort(es.begin(), es.end(), [](const E& a, const E& b){ return a.w < b.w; });

    vector<int> deg(N, 0);
    DSU dsu(N);
    vector<pair<int,int>> out;
    int added = 0;
    for (auto& e : es) {
        if (added == N - 1) break;
        if (deg[e.u] >= cap[e.u] || deg[e.v] >= cap[e.v]) continue;
        if (dsu.find(e.u) == dsu.find(e.v)) continue;
        dsu.unite(e.u, e.v);
        deg[e.u]++; deg[e.v]++;
        out.push_back({e.u + 1, e.v + 1});
        added++;
    }

    // repair: merge any remaining components via their free-slot nodes (cheapest pair)
    while (added < N - 1) {
        long long bd = LLONG_MAX; int bu = -1, bv = -1;
        for (int i = 0; i < N; i++) {
            if (deg[i] >= cap[i]) continue;
            for (int j = i + 1; j < N; j++) {
                if (deg[j] >= cap[j]) continue;
                if (dsu.find(i) == dsu.find(j)) continue;
                long long w = len(i, j);
                if (w < bd) { bd = w; bu = i; bv = j; }
            }
        }
        if (bu < 0) break;  // should not happen with cap>=2
        dsu.unite(bu, bv);
        deg[bu]++; deg[bv]++;
        out.push_back({bu + 1, bv + 1});
        added++;
    }

    // safety fallback: if somehow not spanning, emit the input-order chain instead
    if (added < N - 1) {
        printf("%d\n", N - 1);
        for (int i = 1; i < N; i++) printf("%d %d\n", i, i + 1);
        return 0;
    }

    printf("%d\n", (int)out.size());
    for (auto& e : out) printf("%d %d\n", e.first, e.second);
    return 0;
}
