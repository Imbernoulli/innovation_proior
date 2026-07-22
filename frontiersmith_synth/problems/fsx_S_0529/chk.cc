// Checker/scorer for "Gorge Bridge from Rationed Steel" (frame-truss-load-ceiling).
//
// A participant prints a set of straight steel MEMBERS on an integer lattice.
// Feasibility: distinct in-range members, no duplicate, total steel (sum of squared
// lengths) within Budget.
// Physics/scoring (all deterministic, integer):
//   * A member is ACTIVE (can carry force) iff it belongs to a non-degenerate triangle
//     of members. A member in no triangle "hinges" and carries zero force.
//   * Each active member crossing rock levels has an integer capacity cap = max(0, C - d2)
//     where C = min per-level ceiling over the levels it spans, d2 = squared length.
//   * The load ceiling F = max load routable from the deck (top row) down to the bedrock
//     anchors (bottom row) = min-cut of the active truss (integer max-flow).
// Baseline B = load ceiling of a single canonical triangulated ladder in the two leftmost
// columns.  Score = min(1.0, 0.1 * F / B).
#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int H, W, N;               // rows 0..H, cols 0..W ; N=(H+1)*(W+1)
ll Budget;
vector<ll> C;              // C[0..H-1] per-level ceilings
ll MAXC;

inline int rowOf(int id){ return id/(W+1); }
inline int colOf(int id){ return id%(W+1); }

struct Member{ int a,b; };

// per-member squared length
inline ll d2of(int a,int b){
    ll dx=colOf(a)-colOf(b), dy=rowOf(a)-rowOf(b);
    return dx*dx+dy*dy;
}
// capacity of a member from level ceilings
inline ll capOf(int a,int b){
    int ia=rowOf(a), ib=rowOf(b);
    int lo=min(ia,ib), hi=max(ia,ib);
    ll ceil=MAXC;
    if(hi>lo){ ceil=LLONG_MAX; for(int i=lo;i<hi;i++) ceil=min(ceil,C[i]); }
    ll d2=d2of(a,b);
    ll c=ceil-d2;
    return c>0?c:0;
}

// ---------- Dinic max-flow ----------
struct Dinic{
    struct E{int to; ll cap; int rev;};
    vector<vector<E>> g; vector<int> lvl, it; int n;
    void init(int n_){ n=n_; g.assign(n,{}); }
    void add(int u,int v,ll c){ // undirected member: cap c each direction
        g[u].push_back({v,c,(int)g[v].size()});
        g[v].push_back({u,c,(int)g[u].size()-1});
    }
    void addDir(int u,int v,ll c){
        g[u].push_back({v,c,(int)g[v].size()});
        g[v].push_back({u,0,(int)g[u].size()-1});
    }
    bool bfs(int s,int t){
        lvl.assign(n,-1); queue<int>q; lvl[s]=0; q.push(s);
        while(!q.empty()){int u=q.front();q.pop();
            for(auto&e:g[u]) if(e.cap>0&&lvl[e.to]<0){lvl[e.to]=lvl[u]+1;q.push(e.to);} }
        return lvl[t]>=0;
    }
    ll dfs(int u,int t,ll f){
        if(u==t) return f;
        for(int&i=it[u]; i<(int)g[u].size(); i++){
            E&e=g[u][i];
            if(e.cap>0&&lvl[e.to]==lvl[u]+1){
                ll d=dfs(e.to,t,min(f,e.cap));
                if(d>0){ e.cap-=d; g[e.to][e.rev].cap+=d; return d; }
            }
        }
        return 0;
    }
    ll maxflow(int s,int t){
        ll fl=0;
        while(bfs(s,t)){ it.assign(n,0); ll f; while((f=dfs(s,t,LLONG_MAX))>0) fl+=f; }
        return fl;
    }
};

// active-member detection + max-flow load ceiling for a given member list
ll loadCeiling(const vector<Member>& mem){
    // adjacency for triangle test
    vector<vector<char>> A(N, vector<char>(N,0));
    for(auto&m:mem){ A[m.a][m.b]=1; A[m.b][m.a]=1; }
    // build flow graph over active members
    int S=N, T=N+1;
    Dinic D; D.init(N+2);
    ll INF=(ll)4e15;
    for(auto&m:mem){
        int a=m.a,b=m.b;
        // active iff exists c with A[a][c]&&A[b][c] and non-degenerate triangle
        bool active=false;
        int xa=colOf(a),ya=rowOf(a),xb=colOf(b),yb=rowOf(b);
        for(int c=0;c<N;c++){
            if(c==a||c==b) continue;
            if(A[a][c]&&A[b][c]){
                int xc=colOf(c),yc=rowOf(c);
                ll cross=(ll)(xb-xa)*(yc-ya)-(ll)(yb-ya)*(xc-xa);
                if(cross!=0){ active=true; break; }
            }
        }
        if(!active) continue;
        ll cp=capOf(a,b);
        if(cp<=0) continue;
        D.add(a,b,cp);
    }
    for(int j=0;j<=W;j++){ D.addDir(S, 0*(W+1)+j, INF); }           // deck sources (row 0)
    for(int j=0;j<=W;j++){ D.addDir(H*(W+1)+j, T, INF); }           // bedrock sinks (row H)
    return D.maxflow(S,T);
}

int main(int argc,char**argv){
    registerTestlibCmd(argc,argv);
    H=inf.readInt(); W=inf.readInt(); Budget=inf.readLong();
    N=(H+1)*(W+1);
    C.resize(H); MAXC=0;
    for(int i=0;i<H;i++){ C[i]=inf.readLong(); MAXC=max(MAXC,C[i]); }

    // ---- internal baseline: canonical single triangulated ladder (cols 0,1) ----
    vector<Member> canon;
    auto nid=[&](int i,int j){ return i*(W+1)+j; };
    for(int i=0;i<H;i++){
        canon.push_back({nid(i,0),nid(i+1,0)});   // left vertical
        canon.push_back({nid(i,1),nid(i+1,1)});   // right vertical
        canon.push_back({nid(i,0),nid(i,1)});     // rung (top of panel)
        canon.push_back({nid(i,0),nid(i+1,1)});   // diagonal
    }
    canon.push_back({nid(H,0),nid(H,1)});         // bottom rung
    ll B=loadCeiling(canon);
    if(B<=0) quitf(_fail,"bad instance: baseline B=%lld",B);

    // ---- read participant members ----
    ll MAXM=(ll)N*(N-1)/2;
    int M=ouf.readInt(0,(int)min<ll>(MAXM, 200000),"M");
    vector<Member> mem;
    set<pair<int,int>> seen;
    ll steel=0;
    for(int k=0;k<M;k++){
        int a=ouf.readInt(0,N-1,"a");
        int b=ouf.readInt(0,N-1,"b");
        if(a==b) quitf(_wa,"member %d is a self-loop (node %d)",k+1,a);
        pair<int,int> key=minmax(a,b);
        if(seen.count(key)) quitf(_wa,"duplicate member %d-%d",a,b);
        seen.insert(key);
        steel+=d2of(a,b);
        mem.push_back({a,b});
    }
    if(!ouf.seekEof()) quitf(_wa,"trailing output tokens");
    if(steel>Budget) quitf(_wa,"steel budget exceeded: used %lld > %lld",steel,Budget);

    ll F=loadCeiling(mem);
    double sc=min(1000.0, 100.0*(double)F/(double)max((ll)1,B));
    quitp(sc/1000.0,"OK F=%lld B=%lld steel=%lld/%lld Ratio: %.6f",F,B,steel,Budget,sc/1000.0);
    return 0;
}
