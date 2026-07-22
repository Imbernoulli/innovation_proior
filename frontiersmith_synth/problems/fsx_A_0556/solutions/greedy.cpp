// TIER: greedy
// The obvious "add a backup for resilience" recipe: shortest-path primary plus a
// classic loop-free alternate -- a backup neighbor that is STRICTLY CLOSER to the
// destination (dist[w] < dist[u]), which is guaranteed loop-free by distance.
// This protects any node that happens to have a second strictly-closer neighbor,
// but on rung-only (same-distance) layers no such neighbor exists, so those nodes
// are left with no backup exactly like the trivial routing.
#include <bits/stdc++.h>
using namespace std;

int main(){
    int n, m;
    scanf("%d %d", &n, &m);
    vector<vector<int>> adj(n + 1);
    for (int i = 0; i < m; i++){
        int u, v; scanf("%d %d", &u, &v);
        adj[u].push_back(v); adj[v].push_back(u);
    }
    string out; out.reserve((size_t)n * n * 6);
    char buf[32];
    vector<int> dist(n + 1);
    for (int d = 1; d <= n; d++){
        fill(dist.begin(), dist.end(), -1);
        queue<int> q; dist[d] = 0; q.push(d);
        while (!q.empty()){
            int u = q.front(); q.pop();
            for (int v : adj[u]) if (dist[v] < 0){ dist[v] = dist[u] + 1; q.push(v); }
        }
        for (int u = 1; u <= n; u++){
            if (u == d) continue;
            int prim = -1;
            for (int v : adj[u])
                if (prim == -1 || dist[v] < dist[prim] || (dist[v] == dist[prim] && v < prim))
                    prim = v;
            // loop-free alternate: strictly-closer neighbor != prim, smallest id
            int bk = prim;
            for (int v : adj[u]){
                if (v == prim) continue;
                if (dist[v] < dist[u]){
                    if (bk == prim || v < bk) bk = v;
                }
            }
            int len = sprintf(buf, "%d %d\n", prim, bk);
            out.append(buf, len);
        }
    }
    fputs(out.c_str(), stdout);
    return 0;
}
