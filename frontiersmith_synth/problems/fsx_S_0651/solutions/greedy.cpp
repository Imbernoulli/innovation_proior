// TIER: greedy
// The obvious approach: pool all three scenarios' demand into one list, sort by
// weight descending, and greedily build lines for the heaviest pairs first
// (stop the instant the next pair doesn't fit). This "fits lines to the summed
// OD matrix": it nails weekday/weekend (their pairs have the largest weights)
// but structurally never reaches the low-weight stadium-night pairs, so it
// misses the spike corridor and pays the full walk-penalty there.
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

    vector<vector<int>> lines;
    ll remaining = BUDGET;
    for(auto&p: pooled){
        if((int)lines.size() >= L) break;
        int u=(int)p[0], v=(int)p[1];
        auto path = shortestPathEdges(u,v);
        ll length=0; for(auto&e:path) length += e[2];
        if(length <= remaining){
            remaining -= length;
            vector<int> ln; ln.push_back(u);
            for(auto&e:path) ln.push_back((int)e[1]);
            lines.push_back(ln);
        } else break;
    }
    while((int)lines.size() < L) lines.push_back({1});

    for(auto&ln: lines){
        printf("%d", (int)ln.size());
        for(int x: ln) printf(" %d", x);
        printf("\n");
    }
    return 0;
}
