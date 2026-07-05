// TIER: greedy
// Leaf-only tuner: root=3, all internal nodes=2; for each flower choose its OWN
// level in {2..V} to slide its path-sum into the band while keeping the path coprime.
// Only touches leaf nodes, so it never disturbs another flower's path.
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

    // BFS depths from root 1
    vector<int> par(N+1,0), depth(N+1,0); vector<char> vis(N+1,0);
    queue<int> q; q.push(1); vis[1]=1;
    while(!q.empty()){ int u=q.front(); q.pop(); for(int v:adj[u]) if(!vis[v]){ vis[v]=1; par[v]=u; depth[v]=depth[u]+1; q.push(v); } }

    vector<int> lab(N+1,2); lab[1]=3;
    for(int i=1;i<=N;i++){
        bool leaf = (i!=1 && (int)adj[i].size()==1);
        if(!leaf) continue;
        int d=depth[i];                     // edges from root
        long long basePref = 3 + 2LL*(d-1); // root(3) + (d-1) internal nodes at level 2
        int best=2;
        for(int x=2;x<=V;x++){
            long long s = basePref + x;
            // gcd of {3, (d-1) twos, x}
            long long g=3; if(d-1>0) g=gcdll(g,2); g=gcdll(g,x);
            if(g==1 && s>=L && s<=U){ best=x; break; }
        }
        lab[i]=best;
    }
    for(int i=1;i<=N;i++) printf("%d%c", lab[i], i==N?'\n':' ');
    return 0;
}
