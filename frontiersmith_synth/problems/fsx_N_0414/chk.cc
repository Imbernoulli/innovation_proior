#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

static long long gcdll(long long a,long long b){ while(b){ long long t=a%b; a=b; b=t; } return a<0?-a:a; }

int main(int argc,char*argv[]){
    registerTestlibCmd(argc,argv);

    int N = inf.readInt();
    int V = inf.readInt();
    long long L = inf.readInt();
    long long U = inf.readInt();

    vector<vector<int>> adj(N+1);
    for(int i=0;i<N-1;i++){
        int a=inf.readInt(1,N), b=inf.readInt(1,N);
        adj[a].push_back(b); adj[b].push_back(a);
    }
    vector<long long> w(N+1);
    for(int i=1;i<=N;i++) w[i]=inf.readInt(0LL,(long long)2000000000LL);

    // BFS from root=1 -> parent-before-child order
    vector<int> par(N+1,0), order; order.reserve(N);
    vector<char> vis(N+1,0);
    queue<int> q; q.push(1); vis[1]=1; par[1]=0;
    while(!q.empty()){
        int u=q.front(); q.pop(); order.push_back(u);
        for(int v: adj[u]) if(!vis[v]){ vis[v]=1; par[v]=u; q.push(v); }
    }
    // leaf iff not root and degree 1
    vector<char> isLeaf(N+1,0);
    for(int i=1;i<=N;i++) if(i!=1 && (int)adj[i].size()==1) isLeaf[i]=1;

    // ---- internal reference baseline B: root=3, everything else=2 ----
    vector<long long> accS(N+1,0), accG(N+1,0);
    for(int u: order){
        long long lab = (u==1)?3:2;
        if(u==1){ accS[u]=lab; accG[u]=lab; }
        else { accS[u]=accS[par[u]]+lab; accG[u]=gcdll(accG[par[u]],lab); }
    }
    long long B=0;
    for(int i=1;i<=N;i++) if(isLeaf[i] && accG[i]==1 && accS[i]>=L && accS[i]<=U) B+=w[i];
    if(B<=0) B=1; // guard; generator guarantees B>0

    // ---- participant dosing ----
    vector<long long> lab(N+1,0);
    for(int i=1;i<=N;i++) lab[i]=ouf.readInt((long long)2,(long long)V,"level");
    if(!ouf.seekEof()) quitf(_wa,"trailing tokens after the N nutrient levels");

    for(int u: order){
        if(u==1){ accS[u]=lab[u]; accG[u]=lab[u]; }
        else { accS[u]=accS[par[u]]+lab[u]; accG[u]=gcdll(accG[par[u]],lab[u]); }
    }
    long long F=0;
    for(int i=1;i<=N;i++) if(isLeaf[i] && accG[i]==1 && accS[i]>=L && accS[i]<=U) F+=w[i];

    double sc = min(1000.0, 100.0*(double)F/(double)max(1LL,B));
    quitp(sc/1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc/1000.0);
    return 0;
}
