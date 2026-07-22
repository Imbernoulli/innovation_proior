// TIER: strong
// The insight: read each scenario's shipped oracle cost, and estimate each
// scenario's DO-NOTHING regret (what it would cost with zero service, divided by
// its own oracle). The scenario with the largest do-nothing regret is the
// regret-critical one -- its oracle is cheap relative to what an empty network
// costs it, so ignoring it is disproportionately punished by the max-regret
// objective even though it may carry little total demand. Buy that scenario's
// pairs FIRST (by weight, within its own priority group), THEN spend whatever
// budget remains on the pooled highest-weight remainder -- so the network still
// covers as much of weekday/weekend as it can afford.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

static const ll WALK_MULT = 5;

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

// plain shortest-path DISTANCE (real weights) from src, used for the do-nothing estimate
vector<ll> plainDist(int src){
    vector<ll> dist(N+1, LLONG_MAX/4);
    vector<char> done(N+1,0);
    priority_queue<pair<ll,int>, vector<pair<ll,int>>, greater<>> pq;
    dist[src]=0; pq.push({0,src});
    while(!pq.empty()){
        auto [d,u]=pq.top(); pq.pop();
        if(done[u]) continue; done[u]=1;
        for(auto&e: adjv[u]){
            int v=e.first; ll w=e.second;
            if(dist[u]+w < dist[v]){ dist[v]=dist[u]+w; pq.push({dist[v],v}); }
        }
    }
    return dist;
}

int main(){
    int M; scanf("%d %d", &N, &M);
    adjv.assign(N+1, {});
    for(int i=0;i<M;i++){
        int u,v; ll w; scanf("%d %d %lld", &u,&v,&w);
        adjv[u].push_back({v,w}); adjv[v].push_back({u,w});
    }
    int L; ll BUDGET; scanf("%d %lld", &L, &BUDGET);
    vector<vector<array<ll,3>>> scen(3);
    vector<ll> oracle(3);
    for(int s=0;s<3;s++){
        int D; scanf("%d %lld", &D, &oracle[s]);
        if(oracle[s] < 1) oracle[s]=1;
        for(int i=0;i<D;i++){
            ll u,v,w; scanf("%lld %lld %lld", &u,&v,&w);
            scen[s].push_back({u,v,w});
        }
    }

    // vulnerability_s = do-nothing cost / oracle_s
    double vuln[3];
    for(int s=0;s<3;s++){
        ll doNothing = 0;
        for(auto&p: scen[s]){
            int u=(int)p[0], v=(int)p[1]; ll w=p[2];
            auto d = plainDist(u);
            doNothing += w * (WALK_MULT * d[v]);
        }
        vuln[s] = (double)doNothing / (double)oracle[s];
    }
    int smax = 0;
    for(int s=1;s<3;s++) if(vuln[s] > vuln[smax]) smax = s;

    // boosted order: smax's own pairs (by weight desc) first, then the pooled
    // remainder (by weight desc)
    vector<array<ll,3>> order;
    vector<array<ll,3>> critical = scen[smax];
    std::sort(critical.begin(), critical.end(), [](const array<ll,3>&a, const array<ll,3>&b){ return a[2] > b[2]; });
    for(auto&p: critical) order.push_back(p);
    vector<array<ll,3>> rest;
    for(int s=0;s<3;s++) if(s != smax) for(auto&p: scen[s]) rest.push_back(p);
    std::sort(rest.begin(), rest.end(), [](const array<ll,3>&a, const array<ll,3>&b){ return a[2] > b[2]; });
    for(auto&p: rest) order.push_back(p);

    vector<vector<int>> lines;
    ll remaining = BUDGET;
    for(auto&p: order){
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
