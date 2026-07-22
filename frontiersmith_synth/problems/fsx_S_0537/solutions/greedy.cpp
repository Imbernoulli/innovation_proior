// TIER: greedy
// Smooth congestion-proportional pricing: tax every road in proportion to its
// congestion sensitivity (the coefficient a_e of its congestion term), spending
// the whole budget. This is the obvious first heuristic -- price the congestible
// roads, ignore the zero-latency shortcuts (a_e = 0). It smears the scarce budget
// uniformly, cannot concentrate enough to delete a paradox edge, and wastes/mis-
// spends budget on congestible roads that should NOT be tolled (the decoy
// districts), so it lands far short of surgical intervention.
#include <bits/stdc++.h>
using namespace std;

struct E { int u, v; long long a; int k; long long b; };
static const long long CAP = (long long)4e15;
static long long latency(const E& e, long long x){
    long long xp = 1;
    for (int i = 0; i < e.k; i++){ xp *= x; if (xp > CAP){ xp = CAP; break; } }
    long long r = e.a * xp + e.b; if (r > CAP) r = CAP; return r;
}
static long long settle(int N, const vector<E>& edg, const vector<vector<int>>& adj,
                        const vector<int>& cs, const vector<int>& ct, int R,
                        const vector<long long>& toll, vector<int>* flowOut){
    int M = edg.size(), D = cs.size();
    vector<int> flow(M, 0); vector<vector<int>> path(D);
    auto ec = [&](int e){ long long c = latency(edg[e], (long long)flow[e]+1)+toll[e]; return c>CAP?CAP:c; };
    auto shortest = [&](int s, int t){
        vector<long long> dist(N, LLONG_MAX); vector<int> pe(N, -1);
        priority_queue<pair<long long,int>, vector<pair<long long,int>>, greater<pair<long long,int>>> pq;
        dist[s]=0; pq.push({0,s});
        while(!pq.empty()){ auto tp=pq.top(); pq.pop(); long long d=tp.first; int u=tp.second;
            if(d>dist[u]) continue;
            for(int e: adj[u]){ int v=edg[e].v; long long nd=d+ec(e);
                if(nd<dist[v]){ dist[v]=nd; pe[v]=e; pq.push({nd,v}); } } }
        vector<int> p; int cur=t; while(cur!=s){ int e=pe[cur]; if(e<0){p.clear();return p;} p.push_back(e); cur=edg[e].u; }
        reverse(p.begin(),p.end()); return p;
    };
    auto assign=[&](int i){ auto p=shortest(cs[i],ct[i]); path[i]=p; for(int e:p) flow[e]++; };
    auto rem=[&](int i){ for(int e:path[i]) flow[e]--; path[i].clear(); };
    for(int i=0;i<D;i++) assign(i);
    for(int r=0;r<R;r++) for(int i=0;i<D;i++){ rem(i); assign(i); }
    long long F=0; for(int e=0;e<M;e++) F+=(long long)flow[e]*latency(edg[e],(long long)flow[e]);
    if(flowOut)*flowOut=flow; return F;
}

int main(){
    int N,M,D,R; long long T;
    if(!(cin>>N>>M>>D>>T>>R)) return 0;
    vector<E> edg(M); vector<vector<int>> adj(N);
    for(int e=0;e<M;e++){ int u,v,k; long long a,b; cin>>u>>v>>a>>k>>b; u--; v--; edg[e]={u,v,a,k,b}; adj[u].push_back(e); }
    vector<int> cs(D), ct(D);
    for(int i=0;i<D;i++){ cin>>cs[i]>>ct[i]; cs[i]--; ct[i]--; }

    vector<long long> toll(M, 0);

    // congestion weight = road's congestion sensitivity (coefficient a_e)
    vector<long long> w(M, 0); long long sw = 0;
    for(int e=0;e<M;e++){ w[e] = edg[e].a; sw += w[e]; }
    if(sw > 0 && T > 0){
        long long spent = 0;
        for(int e=0;e<M;e++){
            // proportional allocation, floored so the sum never exceeds T
            long long give = (long long)((__int128)T * w[e] / sw);
            if(give > T - spent) give = T - spent;
            toll[e] = give; spent += give;
        }
    }
    for(int e=0;e<M;e++){ if(e) putchar(' '); printf("%lld", toll[e]); }
    printf("\n");
    return 0;
}
