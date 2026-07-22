// TIER: strong
// Insight: the load ceiling is a MIN-CUT -- it equals the capacity of the WEAKEST rock
// stratum the force paths must cross, not the total tonnage of steel.  So start from the
// same full triangulated sheet, then repeatedly add the single X-brace that raises the
// actual max-flow the most (Delta-flow per steel), which automatically water-fills spare
// steel into exactly the bottleneck stratum.  Reinforcing a strong level buys nothing.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int H,W,N; ll MAXC; vector<ll> C;
inline int rowOf(int id){return id/(W+1);} inline int colOf(int id){return id%(W+1);}
inline ll d2of(int a,int b){ ll dx=colOf(a)-colOf(b),dy=rowOf(a)-rowOf(b); return dx*dx+dy*dy; }
inline ll capOf(int a,int b){
    int ia=rowOf(a),ib=rowOf(b),lo=min(ia,ib),hi=max(ia,ib); ll ceil=MAXC;
    if(hi>lo){ ceil=LLONG_MAX; for(int i=lo;i<hi;i++) ceil=min(ceil,C[i]); }
    ll c=ceil-d2of(a,b); return c>0?c:0;
}
struct Dinic{
    struct E{int to;ll cap;int rev;}; vector<vector<E>> g; vector<int> lvl,it; int n;
    void init(int n_){n=n_; g.assign(n,{});}
    void add(int u,int v,ll c){ g[u].push_back({v,c,(int)g[v].size()}); g[v].push_back({u,c,(int)g[u].size()-1}); }
    void addDir(int u,int v,ll c){ g[u].push_back({v,c,(int)g[v].size()}); g[v].push_back({u,0,(int)g[u].size()-1}); }
    bool bfs(int s,int t){ lvl.assign(n,-1); queue<int>q; lvl[s]=0; q.push(s);
        while(!q.empty()){int u=q.front();q.pop(); for(auto&e:g[u]) if(e.cap>0&&lvl[e.to]<0){lvl[e.to]=lvl[u]+1;q.push(e.to);} } return lvl[t]>=0; }
    ll dfs(int u,int t,ll f){ if(u==t)return f;
        for(int&i=it[u];i<(int)g[u].size();i++){ E&e=g[u][i];
            if(e.cap>0&&lvl[e.to]==lvl[u]+1){ ll d=dfs(e.to,t,min(f,e.cap)); if(d>0){e.cap-=d; g[e.to][e.rev].cap+=d; return d;} } }
        return 0; }
    ll maxflow(int s,int t){ ll fl=0; while(bfs(s,t)){ it.assign(n,0); ll f; while((f=dfs(s,t,LLONG_MAX))>0) fl+=f; } return fl; }
};
ll loadCeiling(const vector<pair<int,int>>& mem){
    vector<vector<char>> A(N, vector<char>(N,0));
    for(auto&m:mem){ A[m.first][m.second]=1; A[m.second][m.first]=1; }
    int S=N,T=N+1; Dinic D; D.init(N+2); ll INF=(ll)4e15;
    for(auto&m:mem){ int a=m.first,b=m.second; bool act=false;
        int xa=colOf(a),ya=rowOf(a),xb=colOf(b),yb=rowOf(b);
        for(int c=0;c<N;c++){ if(c==a||c==b)continue; if(A[a][c]&&A[b][c]){
            int xc=colOf(c),yc=rowOf(c); ll cr=(ll)(xb-xa)*(yc-ya)-(ll)(yb-ya)*(xc-xa); if(cr!=0){act=true;break;} } }
        if(!act) continue; ll cp=capOf(a,b); if(cp<=0) continue; D.add(a,b,cp);
    }
    for(int j=0;j<=W;j++) D.addDir(S, j, INF);
    for(int j=0;j<=W;j++) D.addDir(H*(W+1)+j, T, INF);
    return D.maxflow(S,T);
}
int main(){
    scanf("%d %d",&H,&W); ll Budget; scanf("%lld",&Budget);
    N=(H+1)*(W+1); C.resize(H); MAXC=0;
    for(int i=0;i<H;i++){ scanf("%lld",&C[i]); MAXC=max(MAXC,C[i]); }
    auto nid=[&](int i,int j){ return i*(W+1)+j; };
    vector<pair<int,int>> mem; ll steel=0;
    auto add=[&](int a,int b){ mem.push_back({a,b}); steel+=d2of(a,b); };
    // base sheet
    for(int j=0;j<=W;j++) for(int i=0;i<H;i++) add(nid(i,j),nid(i+1,j));
    for(int i=0;i<=H;i++) for(int j=0;j<W;j++) add(nid(i,j),nid(i,j+1));
    for(int i=0;i<H;i++)  for(int j=0;j<W;j++) add(nid(i,j),nid(i+1,j+1));
    // candidate X-braces
    vector<pair<int,int>> cand;
    for(int i=0;i<H;i++) for(int j=0;j<W;j++) cand.push_back({nid(i,j+1),nid(i+1,j)});
    vector<char> used(cand.size(),0);
    // max-flow guided water-filling
    while(true){
        ll curF=loadCeiling(mem);
        int best=-1; ll bestGain=0;
        for(size_t k=0;k<cand.size();k++){
            if(used[k]) continue;
            ll cost=d2of(cand[k].first,cand[k].second);
            if(steel+cost>Budget) continue;
            mem.push_back(cand[k]);
            ll f=loadCeiling(mem);
            mem.pop_back();
            ll gain=f-curF;
            if(gain>bestGain){ bestGain=gain; best=(int)k; }
        }
        if(best<0) break;                 // no affordable improving brace
        used[best]=1; add(cand[best].first,cand[best].second);
    }
    printf("%d\n",(int)mem.size());
    for(auto&m:mem) printf("%d %d\n",m.first,m.second);
    return 0;
}
