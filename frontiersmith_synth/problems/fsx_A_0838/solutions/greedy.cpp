// TIER: greedy
// Obvious approach: use the paddock-adjacency graph. BFS from each unvisited node and
// chunk shelves in visitation order -- classic graph-locality layout. This recovers the
// zone structure (good) but plants every hub wherever its single structural edge lands
// it -- inside its "home" zone, scattered away from the other hubs it is actually
// co-visited with in the tour.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n, m, pageCap, L, T;
    if (scanf("%d %d %d %d %d", &n, &m, &pageCap, &L, &T) != 5) return 0;
    vector<vector<int>> adj(n + 1);
    for (int i = 0; i < m; i++) {
        int u, v; scanf("%d %d", &u, &v);
        adj[u].push_back(v);
        adj[v].push_back(u);
    }
    for (int i = 0; i < T; i++) { int x; scanf("%d", &x); }

    vector<int> order;
    order.reserve(n);
    vector<char> vis(n + 1, 0);
    for (int s = 1; s <= n; s++) {
        if (vis[s]) continue;
        queue<int> q;
        q.push(s);
        vis[s] = 1;
        while (!q.empty()) {
            int u = q.front(); q.pop();
            order.push_back(u);
            for (int v : adj[u]) if (!vis[v]) { vis[v] = 1; q.push(v); }
        }
    }

    vector<int> page(n + 1);
    for (int k = 0; k < n; k++) page[order[k]] = 1 + k / pageCap;
    for (int i = 1; i <= n; i++) printf("%d\n", page[i]);
    return 0;
}
