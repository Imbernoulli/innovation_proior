// TIER: trivial
// Fewest-hops route per field (BFS), flat min(D,CAP)/F split for every field --
// this exactly mirrors the checker's own internal baseline B, so it scores ~0.1.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main(){
    int N, M, F; ll D, CAP, L;
    scanf("%d %d %d %lld %lld %lld", &N, &M, &F, &D, &CAP, &L);
    vector<int> fieldId(F);
    for (int i = 0; i < F; i++) scanf("%d", &fieldId[i]);
    vector<int> eu(M), ev(M); vector<ll> elen(M); vector<int> eret(M);
    vector<vector<pair<int,int>>> adj(N);
    for (int i = 0; i < M; i++){
        scanf("%d %d %lld %d", &eu[i], &ev[i], &elen[i], &eret[i]);
        adj[eu[i]].push_back({ev[i], i});
        adj[ev[i]].push_back({eu[i], i});
    }
    vector<int> parentEdge(N, -1), dist(N, -1);
    queue<int> q; q.push(0); dist[0] = 0;
    while (!q.empty()){
        int u = q.front(); q.pop();
        for (auto &pr : adj[u]){
            int v = pr.first, eid = pr.second;
            if (dist[v] == -1){ dist[v] = dist[u] + 1; parentEdge[v] = eid; q.push(v); }
        }
    }
    double X = (double)min(D, CAP) / (double)max(1, F);
    printf("%d\n", F);
    for (int i = 0; i < F; i++){
        int v = fieldId[i];
        vector<int> path;
        path.push_back(v);
        while (v != 0){
            int eid = parentEdge[v];
            v = (eu[eid] == v) ? ev[eid] : eu[eid];
            path.push_back(v);
        }
        reverse(path.begin(), path.end());
        printf("%d %.9f %d", fieldId[i], X, (int)path.size());
        for (int x : path) printf(" %d", x);
        printf("\n");
    }
    return 0;
}
