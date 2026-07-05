// TIER: strong
// Degree-constrained Kruskal over the K ground stations (uses full caps 3/4 to branch, not
// just a degree-2 path), then component repair via spare-capacity endpoints, then a Steiner
// relay-insertion pass: splice an unused mast into a long station link a-b (replace a-b by
// a-mast-b) whenever it shortens total cable and ports allow. Beats the nearest-neighbour
// path and diverges from it per-test via the caps and relay usage.
#include <bits/stdc++.h>
using namespace std;

static vector<long long> X, Y;
static inline long long len(int a, int b) {
    double dx = (double)(X[a] - X[b]);
    double dy = (double)(Y[a] - Y[b]);
    return (long long)llround(sqrt(dx * dx + dy * dy));
}

struct DSU {
    vector<int> p;
    DSU(int n) : p(n) { for (int i = 0; i < n; i++) p[i] = i; }
    int find(int x) { while (p[x] != x) { p[x] = p[p[x]]; x = p[x]; } return x; }
    void unite(int a, int b) { p[find(a)] = find(b); }
    bool same(int a, int b) { return find(a) == find(b); }
};

int main() {
    int N, K;
    if (scanf("%d %d", &N, &K) != 2) return 0;
    X.resize(N); Y.resize(N);
    vector<int> cap(N);
    for (int i = 0; i < N; i++)
        scanf("%lld %lld %d", &X[i], &Y[i], &cap[i]);

    vector<int> deg(N, 0);
    set<pair<int,int>> edgeSet;
    vector<pair<int,int>> edges;

    // candidate edges among terminals (0..K-1), sorted by length
    struct E { long long w; int a, b; };
    vector<E> cand;
    cand.reserve((size_t)K * (K - 1) / 2);
    for (int i = 0; i < K; i++)
        for (int j = i + 1; j < K; j++)
            cand.push_back({len(i, j), i, j});
    sort(cand.begin(), cand.end(), [](const E& x, const E& y) { return x.w < y.w; });

    DSU dsu(N);
    auto addEdge = [&](int a, int b) {
        int u = min(a, b), v = max(a, b);
        edgeSet.insert({u, v});
        edges.push_back({u, v});
        deg[a]++; deg[b]++;
        dsu.unite(a, b);
    };

    // Phase 1: degree-constrained Kruskal over terminals
    for (auto& e : cand) {
        if (dsu.same(e.a, e.b)) continue;
        if (deg[e.a] < cap[e.a] && deg[e.b] < cap[e.b])
            addEdge(e.a, e.b);
    }

    // Phase 2: reconnect leftover terminal components using spare-capacity endpoints.
    // With every cap>=2, each component (a path/forest) has a free-port node, so repeated
    // passes over the sorted candidate edges always merge everything.
    auto terminalsConnected = [&]() {
        int r = dsu.find(0);
        for (int i = 1; i < K; i++) if (dsu.find(i) != r) return false;
        return true;
    };
    for (int pass = 0; pass < K + 2 && !terminalsConnected(); pass++) {
        for (auto& e : cand) {
            if (dsu.same(e.a, e.b)) continue;
            if (deg[e.a] < cap[e.a] && deg[e.b] < cap[e.b])
                addEdge(e.a, e.b);
        }
    }
    // Absolute fallback (should not trigger): brute chain along still-free terminals.
    if (!terminalsConnected()) {
        for (int i = 0; i + 1 < K; i++) {
            if (dsu.same(i, i + 1)) continue;
            int u = min(i, i + 1), v = max(i, i + 1);
            if (edgeSet.count({u, v})) continue;
            addEdge(i, i + 1);
        }
    }

    // Phase 3: Steiner relay insertion. For each unused mast r (cap>=2), find the existing
    // link (a,b) minimizing len(a,r)+len(b,r)-len(a,b); if that is a strict reduction,
    // replace a-b with a-r and b-r (a,b keep their degree; r takes 2 ports).
    for (int r = K; r < N; r++) {
        if (cap[r] < 2 || deg[r] != 0) continue;
        long long bestGain = 0; // require strictly positive reduction
        int bestIdx = -1;
        for (int idx = 0; idx < (int)edges.size(); idx++) {
            int a = edges[idx].first, b = edges[idx].second;
            long long gain = len(a, b) - (len(a, r) + len(b, r));
            if (gain > bestGain) { bestGain = gain; bestIdx = idx; }
        }
        if (bestIdx >= 0) {
            int a = edges[bestIdx].first, b = edges[bestIdx].second;
            // avoid creating a duplicate a-r or b-r (r is fresh, so safe)
            edgeSet.erase({min(a, b), max(a, b)});
            // remove edge idx by swapping with back
            edges[bestIdx] = edges.back();
            edges.pop_back();
            // add a-r and b-r
            edgeSet.insert({min(a, r), max(a, r)});
            edges.push_back({min(a, r), max(a, r)});
            edgeSet.insert({min(b, r), max(b, r)});
            edges.push_back({min(b, r), max(b, r)});
            deg[r] += 2;
        }
    }

    printf("%d\n", (int)edges.size());
    for (auto& e : edges)
        printf("%d %d\n", e.first + 1, e.second + 1);
    return 0;
}
