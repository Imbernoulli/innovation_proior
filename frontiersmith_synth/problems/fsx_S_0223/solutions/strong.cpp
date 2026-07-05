// TIER: strong
// Degree-constrained Kruskal over all pairs (cheapest capacity-feasible joining
// edges), then a spare-capacity repair that stitches any leftover components into a
// single connected spanning tree. Uses per-node branching (capacity 3/4) so it beats
// the degree-2 nearest-neighbour chain.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int n;
vector<ll> X, Y, C;

struct DSU {
    vector<int> p;
    DSU(int m) : p(m) { for (int i = 0; i < m; i++) p[i] = i; }
    int find(int x) { while (p[x] != x) x = p[x] = p[p[x]]; return x; }
    bool uni(int a, int b) { a = find(a); b = find(b); if (a == b) return false; p[a] = b; return true; }
};

int main() {
    if (scanf("%d", &n) != 1) return 0;
    X.assign(n + 1, 0); Y.assign(n + 1, 0); C.assign(n + 1, 0);
    for (int i = 1; i <= n; i++) scanf("%lld %lld %lld", &X[i], &Y[i], &C[i]);
    if (n == 1) { printf("0\n"); return 0; }

    struct E { ll w; int u, v; };
    vector<E> edges;
    edges.reserve((size_t)n * (n - 1) / 2);
    for (int i = 1; i <= n; i++)
        for (int j = i + 1; j <= n; j++) {
            ll dx = X[i] - X[j], dy = Y[i] - Y[j];
            edges.push_back({dx * dx + dy * dy, i, j});
        }
    sort(edges.begin(), edges.end(), [](const E& a, const E& b) { return a.w < b.w; });

    vector<int> deg(n + 1, 0);
    DSU dsu(n + 1);
    vector<pair<int,int>> out;
    out.reserve(n);

    for (const auto& e : edges) {
        if ((int)out.size() >= n - 1) break;
        if (deg[e.u] >= C[e.u] || deg[e.v] >= C[e.v]) continue;
        if (dsu.find(e.u) == dsu.find(e.v)) continue;
        dsu.uni(e.u, e.v);
        deg[e.u]++; deg[e.v]++;
        out.push_back({e.u, e.v});
    }

    // ---- repair: connect leftover components via spare-capacity nodes ----
    // Every forest component has total spare capacity >= 2 (since c_i >= 2), so a
    // spare node always exists on each side of a merge.
    if ((int)out.size() < n - 1) {
        unordered_map<int, vector<int>> comp;
        comp.reserve(n * 2);
        for (int i = 1; i <= n; i++) comp[dsu.find(i)].push_back(i);

        auto spareIn = [&](const vector<int>& nodes) -> int {
            for (int x : nodes) if (deg[x] < C[x]) return x;
            return -1;
        };

        int anchor = -1;
        for (auto& kv : comp) {
            const vector<int>& nodes = kv.second;
            if (anchor == -1) { anchor = spareIn(nodes); continue; }
            int cn = spareIn(nodes);
            // connect anchor (in the growing component) to cn (in this component)
            out.push_back({anchor, cn});
            deg[anchor]++; deg[cn]++;
            dsu.uni(anchor, cn);
            // pick a fresh spare node inside the just-merged component for next merge
            int na = spareIn(nodes);
            anchor = (na != -1) ? na : anchor;
        }
    }

    printf("%d\n", (int)out.size());
    for (auto& e : out) printf("%d %d\n", e.first, e.second);
    return 0;
}
