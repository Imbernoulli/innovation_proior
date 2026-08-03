// TIER: greedy
// Cheapest-first path interdiction: repeatedly take a current shortest tour and
// rope off the cheapest affordable doorway on it that keeps the museum connected.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
const ll INF=LLONG_MAX;

int n,m,s,t; ll K;
struct AdjE{int to; ll w; int idx;};
vector<vector<AdjE>> g;
vector<ll> cost;
vector<char> removed;

ll dist_only(){
    vector<ll> d(n+1,INF);
    priority_queue<pair<ll,int>,vector<pair<ll,int>>,greater<>> pq;
    d[s]=0; pq.push({0,s});
    while(!pq.empty()){auto[dd,u]=pq.top();pq.pop(); if(dd>d[u])continue; if(u==t)return dd;
        for(auto&e:g[u]) if(!removed[e.idx]){ll nd=dd+e.w; if(nd<d[e.to]){d[e.to]=nd;pq.push({nd,e.to});}}}
    return d[t];
}
// returns shortest dist and fills pathEdges with edge indices on one shortest tour
ll shortest_path(vector<int>&pathEdges){
    vector<ll> d(n+1,INF); vector<int> pe(n+1,-1), pn(n+1,-1);
    priority_queue<pair<ll,int>,vector<pair<ll,int>>,greater<>> pq;
    d[s]=0; pq.push({0,s});
    while(!pq.empty()){auto[dd,u]=pq.top();pq.pop(); if(dd>d[u])continue;
        for(auto&e:g[u]) if(!removed[e.idx]){ll nd=dd+e.w; if(nd<d[e.to]){d[e.to]=nd;pe[e.to]=e.idx;pn[e.to]=u;pq.push({nd,e.to});}}}
    pathEdges.clear();
    if(d[t]==INF) return INF;
    int cur=t; while(cur!=s){pathEdges.push_back(pe[cur]); cur=pn[cur];}
    return d[t];
}

int main(){
    if(!(cin>>n>>m>>s>>t>>K)) return 0;
    g.assign(n+1,{}); cost.assign(m+1,0); removed.assign(m+1,0);
    for(int i=1;i<=m;i++){int u,v; ll w,c; cin>>u>>v>>w>>c; cost[i]=c;
        g[u].push_back({v,w,i}); g[v].push_back({u,w,i});}
    ll remaining=K;
    vector<int> closed;
    while(true){
        vector<int> pe;
        ll d=shortest_path(pe);
        if(d==INF) break;
        int best=-1; ll bestCost=INF;
        for(int idx:pe){
            if(removed[idx]||cost[idx]>remaining) continue;
            removed[idx]=1; ll d2=dist_only(); removed[idx]=0;
            if(d2==INF) continue;                 // would disconnect
            if(cost[idx]<bestCost){bestCost=cost[idx]; best=idx;}
        }
        if(best==-1) break;
        removed[best]=1; remaining-=cost[best]; closed.push_back(best);
    }
    printf("%d\n",(int)closed.size());
    for(int x:closed) printf("%d\n",x);
    return 0;
}
