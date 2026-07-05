// TIER: strong
// Best-improvement path interdiction: at each step evaluate every affordable
// doorway on the current shortest tour, and rope off the one that lengthens the
// shortest tour the MOST (ties -> cheaper), while keeping the museum connected.
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
        // best strict improver (by improvement, tie -> cheaper)
        int bestImp=-1; ll bestDelay=-1, bestImpCost=INF;
        // fallback: cheapest affordable path edge that keeps s-t connected
        int bestCheap=-1; ll bestCheapCost=INF;
        for(int idx:pe){
            if(removed[idx]||cost[idx]>remaining) continue;
            removed[idx]=1; ll d2=dist_only(); removed[idx]=0;
            if(d2==INF) continue;                 // would disconnect
            if(cost[idx]<bestCheapCost){bestCheapCost=cost[idx]; bestCheap=idx;}
            if(d2>d && (d2>bestDelay || (d2==bestDelay && cost[idx]<bestImpCost))){
                bestDelay=d2; bestImpCost=cost[idx]; bestImp=idx;
            }
        }
        int pick = (bestImp!=-1) ? bestImp : bestCheap;   // prefer big jumps, else keep spending
        if(pick==-1) break;
        removed[pick]=1; remaining-=cost[pick]; closed.push_back(pick);
    }
    printf("%d\n",(int)closed.size());
    for(int x:closed) printf("%d\n",x);
    return 0;
}
