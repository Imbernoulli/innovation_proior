// TIER: greedy
// The obvious approach: send every room to its ACTUAL nearest exit by real
// transit weight (Dijkstra by corridor length), completely ignoring capacity and
// every fire scenario. Reasonable in isolation, but scenario-blind: in a shared
// corridor system this sends everyone through whichever corridor is globally
// fastest, concentrating load precisely where the planted fire ensemble strikes.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
const ll INF64 = (ll)4e18;

int main(){
    int N, M, E, X, K;
    scanf("%d %d %d %d %d", &N, &M, &E, &X, &K);
    vector<int> exits(X);
    vector<char> isExit(M + 1, 0);
    for (int i = 0; i < X; i++){ scanf("%d", &exits[i]); isExit[exits[i]] = 1; }
    vector<ll> pop_(N + 1, 0);
    for (int i = 1; i <= N; i++) scanf("%lld", &pop_[i]);

    vector<vector<pair<int,ll>>> adjW(M + 1); // (neighbor, len) -- for weighted Dijkstra
    vector<vector<pair<int,int>>> adjE(M + 1); // (neighbor, edgeId)
    vector<ll> elen(E + 1);
    for (int e = 1; e <= E; e++){
        int u, v; ll cap, len;
        scanf("%d %d %lld %lld", &u, &v, &cap, &len);
        elen[e] = len;
        adjW[u].push_back({v, len}); adjW[v].push_back({u, len});
        adjE[u].push_back({v, e});   adjE[v].push_back({u, e});
    }
    // scenarios not needed by this (scenario-blind) construction.

    // multi-source Dijkstra from all exits simultaneously (by transit length)
    vector<ll> dist(M + 1, INF64);
    vector<int> par(M + 1, -1), parEdge(M + 1, -1);
    priority_queue<pair<ll,int>, vector<pair<ll,int>>, greater<pair<ll,int>>> pq;
    for (int x : exits){ dist[x] = 0; pq.push({0, x}); }
    while (!pq.empty()){
        auto [d, u] = pq.top(); pq.pop();
        if (d > dist[u]) continue;
        for (size_t k = 0; k < adjW[u].size(); k++){
            int v = adjW[u][k].first; ll w = adjW[u][k].second;
            int eid = adjE[u][k].second;
            if (dist[u] + w < dist[v]){
                dist[v] = dist[u] + w; par[v] = u; parEdge[v] = eid;
                pq.push({dist[v], v});
            }
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
