#include <bits/stdc++.h>
using namespace std;

struct DSU {
    vector<int> p, r;
    DSU(int n) : p(n), r(n, 0) { iota(p.begin(), p.end(), 0); }
    int find(int x) { while (p[x] != x) { p[x] = p[p[x]]; x = p[x]; } return x; }
    bool unite(int a, int b) {
        a = find(a); b = find(b);
        if (a == b) return false;
        if (r[a] < r[b]) swap(a, b);
        p[b] = a;
        if (r[a] == r[b]) r[a]++;
        return true;
    }
};

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, m, q;
    if (!(cin >> n >> m >> q)) return 0;

    // edges stored as (capacity, u, v) so a plain sort orders by capacity first.
    vector<array<long long, 3>> edges(m);
    for (int i = 0; i < m; i++) {
        long long u, v, c;
        cin >> u >> v >> c;
        edges[i] = {c, u, v};
    }

    vector<int> S(q), T(q);
    for (int k = 0; k < q; k++) cin >> S[k] >> T[k];

    // Maximum spanning tree via Kruskal on DSU: add edges in DECREASING capacity.
    sort(edges.begin(), edges.end(),
         [](const array<long long, 3>& a, const array<long long, 3>& b) {
             return a[0] > b[0];
         });

    DSU dsu(n);
    vector<vector<pair<int, long long>>> tree(n);  // adjacency of the chosen tree
    for (auto& e : edges) {
        int u = (int)e[1], v = (int)e[2];
        long long c = e[0];
        if (dsu.unite(u, v)) {
            tree[u].push_back({v, c});
            tree[v].push_back({u, c});
        }
    }

    // Bottleneck(s, t) = min capacity on the unique s-t path of the max spanning tree.
    // The graph is connected, so the path always exists. n, q <= 2000 => O(q*n) BFS.
    const long long UNVISITED = LLONG_MIN;
    for (int k = 0; k < q; k++) {
        int s = S[k], t = T[k];
        vector<long long> best(n, UNVISITED);
        best[s] = LLONG_MAX;  // empty prefix: bottleneck is +infinity
        deque<int> bfs;
        bfs.push_back(s);
        while (!bfs.empty()) {
            int x = bfs.front();
            bfs.pop_front();
            for (auto& pr : tree[x]) {
                int y = pr.first;
                long long c = pr.second;
                if (best[y] == UNVISITED) {
                    best[y] = min(best[x], c);
                    bfs.push_back(y);
                }
            }
        }
        cout << best[t] << "\n";
    }
    return 0;
}
