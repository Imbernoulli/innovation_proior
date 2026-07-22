// TIER: trivial
// Reproduces the checker's own minimal-effort baseline B: pool all demand pairs
// from the three scenarios, sort by weight descending, and try a SINGLE line for
// the heaviest pair using only 3% of BUDGET -- stop the instant it doesn't fit.
// The remaining L-1 lines are printed as unused (degenerate, k=1).
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int N;
vector<vector<pair<int,ll>>> adjv;

vector<array<ll,3>> shortestPathEdges(int s,int t){
    vector<ll> dist(N+1, LLONG_MAX/4);
    vector<int> par(N+1, -1);
    vector<char> done(N+1, 0);
    priority_queue<pair<ll,int>, vector<pair<ll,int>>, greater<>> pq;
    dist[s]=0; pq.push({0,s});
    while(!pq.empty()){
        auto [d,u] = pq.top(); pq.pop();
        if(done[u]) continue; done[u]=1;
        if(u==t) break;
        for(auto&e: adjv[u]){
            int v=e.first; ll w=e.second;
            if(dist[u]+w < dist[v]){ dist[v]=dist[u]+w; par[v]=u; pq.push({dist[v],v}); }
        }
    }
    vector<array<ll,3>> path;
    if(dist[t] > LLONG_MAX/8) return path;
    int cur=t;
    while(cur!=s){
        int p = par[cur];
        ll w=-1;
        for(auto&e: adjv[p]) if(e.first==cur){ w=e.second; break; }
        path.push_back({(ll)p,(ll)cur,w});
        cur=p;
    }
    reverse(path.begin(), path.end());
    return path;
}

int main(){
    int M; scanf("%d %d", &N, &M);
    adjv.assign(N+1, {});
    for(int i=0;i<M;i++){
        int u,v; ll w; scanf("%d %d %lld", &u,&v,&w);
        adjv[u].push_back({v,w}); adjv[v].push_back({u,w});
    }
    int L; ll BUDGET; scanf("%d %lld", &L, &BUDGET);
    vector<array<ll,3>> pooled;
    for(int s=0;s<3;s++){
        int D; ll oracle; scanf("%d %lld", &D, &oracle);
        for(int i=0;i<D;i++){
            ll u,v,w; scanf("%lld %lld %lld", &u,&v,&w);
            pooled.push_back({u,v,w});
        }
    }
    std::sort(pooled.begin(), pooled.end(), [](const array<ll,3>&a, const array<ll,3>&b){ return a[2] > b[2]; });

    ll budgetTriv = max((ll)1, (ll)llround(BUDGET*0.03));
    vector<int> line1;
    if(!pooled.empty()){
        int u=(int)pooled[0][0], v=(int)pooled[0][1];
        auto path = shortestPathEdges(u,v);
        ll length=0; for(auto&e:path) length+=e[2];
        if(length<=budgetTriv){
            line1.push_back(u);
            for(auto&e:path) line1.push_back((int)e[1]);
        }
    }
    if(line1.empty()) line1.push_back(1);

    printf("%d", (int)line1.size());
    for(int x: line1) printf(" %d", x);
    printf("\n");
    for(int i=1;i<L;i++) printf("1 1\n");
    return 0;
}
