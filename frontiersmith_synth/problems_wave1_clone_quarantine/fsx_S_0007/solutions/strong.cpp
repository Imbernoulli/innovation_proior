// TIER: strong
// Degree-constrained Kruskal minimum spanning tree with a spare-capacity repair.
// Sort all candidate links by length; add each link if it joins two different components
// and both endpoints still have a free relay head. This yields a branching tree that
// hugs short links (approaching the true MST, well below any Hamiltonian path). If the
// greedy pass leaves several components, connect them via nodes that still have spare
// capacity -- every component of a cap>=2 forest keeps >= 2 spare slots, so this always
// completes with a feasible spanning tree.
#include <bits/stdc++.h>
using namespace std;

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
    vector<long long> X(N), Y(N);
    vector<int> cap(N), rem(N);
    for (int i = 0; i < N; i++) { scanf("%lld %lld %d", &X[i], &Y[i], &cap[i]); rem[i] = cap[i]; }

    auto lenr = [&](int a, int b) {
        double dx = (double)(X[a] - X[b]), dy = (double)(Y[a] - Y[b]);
        return llround(sqrt(dx * dx + dy * dy));
    };

    // all candidate links, sorted by length (deterministic tie-break by indices)
    struct E { long long w; int u, v; };
    vector<E> edges;
    edges.reserve((size_t)N * (N - 1) / 2);
    for (int i = 0; i < N; i++)
        for (int j = i + 1; j < N; j++)
            edges.push_back({lenr(i, j), i, j});
    sort(edges.begin(), edges.end(), [](const E& a, const E& b) {
        if (a.w != b.w) return a.w < b.w;
        if (a.u != b.u) return a.u < b.u;
        return a.v < b.v;
    });

    DSU dsu(N);
    vector<pair<int,int>> chosen;
    int comps = N;
    for (const auto& e : edges) {
        if (comps == 1) break;
        if (rem[e.u] <= 0 || rem[e.v] <= 0) continue;
        if (dsu.unite(e.u, e.v)) {
            rem[e.u]--; rem[e.v]--;
            chosen.push_back({e.u, e.v});
            comps--;
        }
    }

    // repair: connect any remaining components through spare-capacity nodes
    while (comps > 1) {
        // pick one spare node per component root
        unordered_map<int,int> repr;
        repr.reserve(comps * 2);
        for (int i = 0; i < N; i++) {
            if (rem[i] > 0) {
                int r = dsu.find(i);
                if (!repr.count(r)) repr[r] = i;
            }
        }
        // connect the first component to any other
        int firstRoot = -1, firstNode = -1;
        for (auto& kv : repr) { firstRoot = kv.first; firstNode = kv.second; break; }
        int otherNode = -1;
        for (auto& kv : repr) {
            if (kv.first != firstRoot) { otherNode = kv.second; break; }
        }
        if (firstNode < 0 || otherNode < 0) break;   // should not happen with cap>=2
        dsu.unite(firstNode, otherNode);
        rem[firstNode]--; rem[otherNode]--;
        chosen.push_back({firstNode, otherNode});
        comps--;
    }

    printf("%d\n", (int)chosen.size());
    for (auto& e : chosen) printf("%d %d\n", e.first + 1, e.second + 1);
    return 0;
}
