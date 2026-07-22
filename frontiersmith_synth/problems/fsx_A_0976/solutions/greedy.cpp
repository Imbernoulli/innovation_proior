// TIER: greedy
// Natural first attempt: proximity/coverage siting on the RAW road graph, ignoring
// capacities and completely ignoring the published blockage scenarios. For each
// candidate site, BFS its hop-distance to the nearest population node on the
// unblocked graph; rank sites by (distance ascending, home-district population
// descending) and build the S closest-to-population sites. This is exactly the
// "obvious" k-median-flavored heuristic a strong coder writes first.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main(){
    int N,M,C,S,K;
    scanf("%d %d %d %d %d", &N,&M,&C,&S,&K);
    vector<ll> pop(N+1,0);
    for (int i=1;i<=N;i++) scanf("%lld", &pop[i]);
    vector<vector<int>> adj(N+1);
    for (int e=0;e<M;e++){
        int u,v; ll c; scanf("%d %d %lld", &u,&v,&c);
        adj[u].push_back(v); adj[v].push_back(u);
    }
    vector<int> candNode(C); vector<ll> candCap(C);
    for (int i=0;i<C;i++) scanf("%d %lld", &candNode[i], &candCap[i]);
    for (int s=0;s<K;s++){
        int b; scanf("%d", &b);
        for (int j=0;j<b;j++){ int x; scanf("%d",&x); }
    }

    // BFS distance from every node to nearest populated node (multi-source BFS)
    vector<int> dist(N+1, INT_MAX);
    vector<ll> nearestPop(N+1, 0);
    queue<int> q;
    for (int i=1;i<=N;i++) if (pop[i] > 0){ dist[i]=0; nearestPop[i]=pop[i]; q.push(i); }
    while(!q.empty()){
        int u=q.front(); q.pop();
        for (int v: adj[u]){
            if (dist[v] > dist[u]+1){
                dist[v]=dist[u]+1; nearestPop[v]=nearestPop[u]; q.push(v);
            }
        }
    }

    vector<int> order(C);
    for (int i=0;i<C;i++) order[i]=i;
    sort(order.begin(), order.end(), [&](int a, int b){
        int da = dist[candNode[a]], db = dist[candNode[b]];
        if (da != db) return da < db;
        if (nearestPop[candNode[a]] != nearestPop[candNode[b]]) return nearestPop[candNode[a]] > nearestPop[candNode[b]];
        return a < b;
    });

    for (int i=0;i<S;i++) printf("%d ", order[i]);
    printf("\n");
    return 0;
}
