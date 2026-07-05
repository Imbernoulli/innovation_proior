// TIER: strong
// Per-subtree uniform inflation. Root fixed at 3. Each child-of-root subtree is
// independent (they only share the root). For each subtree try a common internal
// level u in {2..V}; with that u, tune every flower's own level to bloom; keep the
// u that maximizes the subtree's blooming weight. Captures shallow/planted/needle
// flowers and resolves trap brooms by value -- always at least as good as the
// leaf-only tuner (which is the u=2 case).
#include <bits/stdc++.h>
using namespace std;
static long long gcdll(long long a,long long b){ while(b){ long long t=a%b; a=b; b=t; } return a<0?-a:a; }
int main(){
    int N,V; long long L,U;
    if(scanf("%d %d %lld %lld",&N,&V,&L,&U)!=4) return 0;
    vector<vector<int>> adj(N+1);
    for(int i=0;i<N-1;i++){ int a,b; scanf("%d %d",&a,&b); adj[a].push_back(b); adj[b].push_back(a); }
    vector<long long> w(N+1);
    for(int i=1;i<=N;i++) scanf("%lld",&w[i]);

    vector<int> par(N+1,0), depth(N+1,0), order; order.reserve(N);
    vector<char> vis(N+1,0);
    { queue<int> q; q.push(1); vis[1]=1;
      while(!q.empty()){ int u=q.front(); q.pop(); order.push_back(u);
        for(int v:adj[u]) if(!vis[v]){ vis[v]=1; par[v]=u; depth[v]=depth[u]+1; q.push(v); } } }

    vector<char> isLeaf(N+1,0);
    for(int i=1;i<=N;i++) if(i!=1 && (int)adj[i].size()==1) isLeaf[i]=1;

    // subtree id = the child-of-root ancestor of each node
    vector<int> sub(N+1,0);
    for(int u: order){
        if(u==1) sub[u]=0;
        else sub[u] = (par[u]==1)? u : sub[par[u]];
    }
    unordered_map<int,vector<int>> leavesOf; leavesOf.reserve(1024);
    for(int i=1;i<=N;i++) if(isLeaf[i]) leavesOf[sub[i]].push_back(i);

    // best own level for a flower at depth d given internal level u (0 if none blooms)
    auto bestLeafLevel=[&](int d,int u)->int{
        for(int x=2;x<=V;x++){
            long long s = 3 + (long long)u*(d-1) + x;
            long long g=3; if(d-1>0) g=gcdll(g,u); g=gcdll(g,x);
            if(g==1 && s>=L && s<=U) return x;
        }
        return 0;
    };

    // pick best uniform internal level per subtree
    unordered_map<int,int> bestU; bestU.reserve(leavesOf.size()*2+8);
    for(auto &kv: leavesOf){
        long long bw=-1; int bu=2;
        for(int u=2;u<=V;u++){
            long long tot=0;
            for(int f: kv.second) if(bestLeafLevel(depth[f],u)) tot+=w[f];
            if(tot>bw){ bw=tot; bu=u; }
        }
        bestU[kv.first]=bu;
    }

    vector<int> lab(N+1,2); lab[1]=3;
    // internal nodes -> their subtree's chosen u
    for(int i=2;i<=N;i++){
        if(isLeaf[i]) continue;
        auto it=bestU.find(sub[i]);
        lab[i] = (it==bestU.end())? 2 : it->second;
    }
    // leaves -> best own level under that u (else 2)
    for(auto &kv: leavesOf){
        int u=bestU[kv.first];
        for(int f: kv.second){
            int x=bestLeafLevel(depth[f],u);
            lab[f] = x? x : 2;
        }
    }

    for(int i=1;i<=N;i++) printf("%d%c", lab[i], i==N?'\n':' ');
    return 0;
}
