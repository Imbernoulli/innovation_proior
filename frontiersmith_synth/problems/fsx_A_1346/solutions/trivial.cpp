// TIER: trivial
#include <bits/stdc++.h>
using namespace std;

// Fixed deepest-first BFS elimination order (root=vertex 0 last). Guard every eliminated
// vertex with ALL of its currently-live neighbors. This is always valid (a vertex's own
// live-neighbor set trivially dominates it) but never exploits any redundancy among
// those neighbors -- it is exactly the internal baseline the checker itself computes.
int main() {
    ios::sync_with_stdio(false); cin.tie(nullptr);
    int n; long long m;
    cin >> n >> m;
    vector<vector<int>> adj(n);
    for (long long i = 0; i < m; i++) {
        int u, v; cin >> u >> v;
        adj[u].push_back(v);
        adj[v].push_back(u);
    }

    vector<int> depth(n, -1);
    queue<int> q;
    depth[0] = 0; q.push(0);
    while (!q.empty()) {
        int u = q.front(); q.pop();
        for (int v : adj[u]) if (depth[v] == -1) { depth[v] = depth[u] + 1; q.push(v); }
    }

    vector<int> order(n);
    iota(order.begin(), order.end(), 0);
    sort(order.begin(), order.end(), [&](int a, int b) {
        if (depth[a] != depth[b]) return depth[a] > depth[b];
        return a < b;
    });
    order.pop_back(); // drop the root (unique minimum depth)

    vector<char> alive(n, 1);
    for (int v : order) {
        vector<int> guards;
        for (int u : adj[v]) if (alive[u]) guards.push_back(u);
        cout << v << ' ' << guards.size();
        for (int g : guards) cout << ' ' << g;
        cout << '\n';
        alive[v] = 0;
    }
    return 0;
}
