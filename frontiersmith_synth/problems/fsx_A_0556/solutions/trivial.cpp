// TIER: trivial
// Shortest-path forwarding, NO backup (backup = primary). This is exactly the
// checker's reference baseline B -> scores ~0.1. Any cut of a shortest-path-tree
// edge strands the whole subtree behind it.
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
            int best = -1;
            for (int v : adj[u])
                if (best == -1 || dist[v] < dist[best] || (dist[v] == dist[best] && v < best))
                    best = v;
            int len = sprintf(buf, "%d %d\n", best, best);
            out.append(buf, len);
        }
    }
    fputs(out.c_str(), stdout);
    return 0;
}
