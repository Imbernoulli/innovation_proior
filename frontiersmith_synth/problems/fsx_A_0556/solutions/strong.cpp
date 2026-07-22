// TIER: strong
// The insight: resilience is a property of the forwarding DAG, not of path length.
// Primary is still the shortest-path next hop, so the primaries form a tree rooted
// at d. A backup for u must let the packet escape when the link (u, prim[u]) dies.
// A neighbor w is a valid escape iff its OWN primary route to d does not pass back
// through u -- i.e. w is NOT in u's subtree of the primary tree. Such a w exists
// even when it is NOT strictly closer (a same-distance rung neighbor whose rail
// reaches d avoiding u), which the path-length recipe rejects. We take the closest
// valid escape. This protects every node that has an arc-disjoint second route,
// trading a longer detour for survivability.
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
    vector<int> dist(n + 1), prim(n + 1), tin(n + 1), tout(n + 1);
    vector<vector<int>> ch(n + 1);
    for (int d = 1; d <= n; d++){
        fill(dist.begin(), dist.end(), -1);
        queue<int> q; dist[d] = 0; q.push(d);
        while (!q.empty()){
            int u = q.front(); q.pop();
            for (int v : adj[u]) if (dist[v] < 0){ dist[v] = dist[u] + 1; q.push(v); }
        }
        for (int u = 1; u <= n; u++) ch[u].clear();
        for (int u = 1; u <= n; u++){
            if (u == d){ prim[u] = 0; continue; }
            int best = -1;
            for (int v : adj[u])
                if (best == -1 || dist[v] < dist[best] || (dist[v] == dist[best] && v < best))
                    best = v;
            prim[u] = best;
            ch[best].push_back(u);
        }
        // iterative DFS of the primary tree from d -> in/out times
        int timer = 0;
        vector<pair<int,int>> st;         // (node, child-index)
        st.push_back({d, 0});
        tin[d] = timer++;
        while (!st.empty()){
            auto &top = st.back();
            int u = top.first;
            if (top.second < (int)ch[u].size()){
                int c = ch[u][top.second++];
                tin[c] = timer++;
                st.push_back({c, 0});
            } else { tout[u] = timer++; st.pop_back(); }
        }
        auto isAncestor = [&](int a, int w){   // a is ancestor of w in primary tree
            return tin[a] <= tin[w] && tout[w] <= tout[a];
        };
        for (int u = 1; u <= n; u++){
            if (u == d) continue;
            int p = prim[u];
            int bk = p;                        // fall back to no backup
            int bestKey = INT_MAX, bestId = INT_MAX;
            for (int w : adj[u]){
                if (w == p) continue;
                if (isAncestor(u, w)) continue;   // w's route returns through u -> unsafe
                int key = dist[w];
                if (key < bestKey || (key == bestKey && w < bestId)){
                    bestKey = key; bestId = w; bk = w;
                }
            }
            int len = sprintf(buf, "%d %d\n", p, bk);
            out.append(buf, len);
        }
    }
    fputs(out.c_str(), stdout);
    return 0;
}
