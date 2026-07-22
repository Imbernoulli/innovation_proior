// TIER: strong
// Insight: routing is downstream of GEOMETRY. Reformulate the door draw as a
// weighted linear-arrangement problem on the truck flow graph and solve it with the
// classic greedy-edge path-construction heuristic (process (i,j,f) pairs in
// descending flow order; connect i and j directly whenever both still have a free
// end and doing so would not close a cycle). This chains the heaviest flows into
// directly-adjacent doors first, so matched inbound/outbound pairs end up next to
// each other and their trips share almost no segments with unrelated flow -- the
// pallet flow field becomes close to laminar BEFORE any batching cleverness. Batching
// is then applied the same way greedy does it (maximal K-sized trips): the geometry
// fix is what actually moves the score.
#include <bits/stdc++.h>
using namespace std;

struct DSU {
    vector<int> p;
    DSU(int n) : p(n) { for (int i = 0; i < n; i++) p[i] = i; }
    int find(int x) { while (p[x] != x) { p[x] = p[p[x]]; x = p[x]; } return x; }
    bool unite(int a, int b) { a = find(a); b = find(b); if (a == b) return false; p[a] = b; return true; }
};

int main() {
    int Tin, Tout, M, K, cap;
    scanf("%d %d %d %d %d", &Tin, &Tout, &M, &K, &cap);
    vector<int> pi_(M), pj_(M), pf_(M);
    for (int e = 0; e < M; e++) scanf("%d %d %d", &pi_[e], &pj_[e], &pf_[e]);

    int D = Tin + Tout;

    // edges: (flow, truckA, truckB) -- truckA in 1..Tin, truckB in Tin+1..D
    vector<array<int,3>> edges(M);
    for (int e = 0; e < M; e++) edges[e] = {pf_[e], pi_[e], Tin + pj_[e]};
    // descending by flow; deterministic tie-break by (a,b) so re-runs are identical
    sort(edges.begin(), edges.end(), [](const array<int,3>& a, const array<int,3>& b) {
        if (a[0] != b[0]) return a[0] > b[0];
        if (a[1] != b[1]) return a[1] < b[1];
        return a[2] < b[2];
    });

    vector<vector<int>> adj(D + 1);
    vector<int> deg(D + 1, 0);
    DSU dsu(D + 1);
    for (auto& ed : edges) {
        int a = ed[1], b = ed[2];
        if (deg[a] >= 2 || deg[b] >= 2) continue;
        if (dsu.find(a) == dsu.find(b)) continue; // would close a cycle
        adj[a].push_back(b);
        adj[b].push_back(a);
        deg[a]++; deg[b]++;
        dsu.unite(a, b);
    }

    // walk each resulting path/isolated-node component, then concatenate all
    // components (in order of first appearance) into one full permutation of trucks.
    vector<bool> compVisited(D + 1, false);
    vector<int> order;
    order.reserve(D);
    for (int v = 1; v <= D; v++) {
        if (compVisited[v]) continue;
        // collect this component's nodes
        vector<int> compNodes;
        vector<int> st = {v};
        compVisited[v] = true;
        while (!st.empty()) {
            int cur = st.back(); st.pop_back();
            compNodes.push_back(cur);
            for (int nb : adj[cur]) if (!compVisited[nb]) { compVisited[nb] = true; st.push_back(nb); }
        }
        int start = compNodes[0];
        for (int u : compNodes) if (deg[u] <= 1) { start = u; break; }
        vector<bool> used(D + 1, false);
        int cur = start;
        while (true) {
            order.push_back(cur);
            used[cur] = true;
            int nxt = -1;
            for (int nb : adj[cur]) if (!used[nb]) { nxt = nb; break; }
            if (nxt == -1) break;
            cur = nxt;
        }
    }

    vector<int> door(D + 1, 0);
    for (int pos = 0; pos < (int)order.size(); pos++) door[order[pos]] = pos + 1;

    for (int k = 1; k <= D; k++) printf("%d%c", door[k], k == D ? '\n' : ' ');

    for (int e = 0; e < M; e++) {
        int f = pf_[e];
        int m = (f + K - 1) / K;
        printf("%d", m);
        int rem = f;
        for (int t = 0; t < m; t++) {
            int sz = min(K, rem);
            rem -= sz;
            printf(" %d", sz);
        }
        printf("\n");
    }
    return 0;
}
