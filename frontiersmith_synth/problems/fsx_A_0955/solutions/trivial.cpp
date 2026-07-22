// TIER: trivial
// "Do nothing" reference: route every room to the exit nearest by raw HOP COUNT
// (plain multi-source BFS, all corridors treated as unit-weight, no notion of
// transit length, capacity, or any fire scenario). This is exactly the checker's
// own internal baseline construction -> reproduces B, ratio ~0.1.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main(){
    int N, M, E, X, K;
    scanf("%d %d %d %d %d", &N, &M, &E, &X, &K);
    vector<int> exits(X);
    vector<char> isExit(M + 1, 0);
    for (int i = 0; i < X; i++){ scanf("%d", &exits[i]); isExit[exits[i]] = 1; }
    for (int i = 1; i <= N; i++){ ll p; scanf("%lld", &p); } // populations, unused

    vector<vector<pair<int,int>>> adj(M + 1);
    for (int e = 1; e <= E; e++){
        int u, v; ll cap, len;
        scanf("%d %d %lld %lld", &u, &v, &cap, &len);
        adj[u].push_back({v, e});
        adj[v].push_back({u, e});
    }
    // scenarios are irrelevant to this construction; do not need to be read.

    vector<int> dist(M + 1, -1), par(M + 1, -1), parEdge(M + 1, -1);
    deque<int> q;
    for (int x : exits){ dist[x] = 0; q.push_back(x); }
    while (!q.empty()){
        int u = q.front(); q.pop_front();
        for (auto &pr : adj[u]){
            int v = pr.first, eid = pr.second;
            if (dist[v] == -1){ dist[v] = dist[u] + 1; par[v] = u; parEdge[v] = eid; q.push_back(v); }
        }
    }
    for (int i = 1; i <= N; i++){
        vector<int> path;
        int cur = i;
        path.push_back(cur);
        while (!isExit[cur]){ cur = par[cur]; path.push_back(cur); }
        printf("%d", (int)path.size());
        for (int v : path) printf(" %d", v);
        printf("\n");
    }
    return 0;
}
