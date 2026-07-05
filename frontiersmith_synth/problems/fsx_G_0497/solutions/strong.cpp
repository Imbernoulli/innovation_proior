// TIER: strong
// Multi-strategy floorplanner: try many block ORDERINGS (index, BFS from several seeds, DFS,
// power-sorted) shelf-packed at several effective WIDTHS (narrow bands make clusters compact ->
// lower HPWL), keep the best feasible layout; then local search by swapping pairs in the winning
// order. Every candidate is a shelf packing, hence always non-overlapping and (if it fits H) feasible.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int N, M, W, H, R;
vector<ll> w, h, p;
vector<vector<int>> nets, adj;

ll objective(const vector<ll>& x, const vector<ll>& y) {
    vector<ll> cx(N), cy(N);
    for (int i = 0; i < N; i++) { cx[i]=2*x[i]+w[i]; cy[i]=2*y[i]+h[i]; }
    ll wire = 0;
    for (auto& s : nets) {
        ll a=LLONG_MAX,b=LLONG_MIN,c=LLONG_MAX,d=LLONG_MIN;
        for (int u : s){a=min(a,cx[u]);b=max(b,cx[u]);c=min(c,cy[u]);d=max(d,cy[u]);}
        wire += (b-a)+(d-c);
    }
    ll therm = 0;
    for (int i=0;i<N;i++){ if(!p[i])continue; for(int j=i+1;j<N;j++){ if(!p[j])continue;
        ll dd=llabs(cx[i]-cx[j])+llabs(cy[i]-cy[j]); if(dd<R) therm += p[i]*p[j]*(ll)(R-dd);} }
    return wire+therm;
}

// shelf pack `order` at effective width Weff; returns fits (all inside H).
bool shelf(const vector<int>& order, int Weff, vector<ll>& x, vector<ll>& y) {
    x.assign(N,0); y.assign(N,0);
    ll cx=0, cy=0, rowh=0, maxY=0;
    for (int idx=0; idx<N; idx++){ int i=order[idx];
        if (cx + w[i] > Weff){ cy += rowh; cx=0; rowh=0; }
        x[i]=cx; y[i]=cy; cx+=w[i]; rowh=max(rowh,h[i]);
        maxY = max(maxY, y[i]+h[i]);
    }
    return maxY <= H && Weff >= 1;
}

vector<int> bfsOrder(int start){
    vector<int> order; order.reserve(N);
    vector<char> vis(N,0);
    // start from `start`, then continue with remaining unvisited by index
    for(int s0=0;s0<N;s0++){
        int st = (s0==0)? start : s0;
        if(vis[st])continue;
        queue<int>q;q.push(st);vis[st]=1;
        while(!q.empty()){int u=q.front();q.pop();order.push_back(u);
            for(int v:adj[u]) if(!vis[v]){vis[v]=1;q.push(v);} }
    }
    return order;
}
vector<int> dfsOrder(int start){
    vector<int> order; order.reserve(N);
    vector<char> vis(N,0);
    for(int s0=0;s0<N;s0++){
        int st=(s0==0)?start:s0; if(vis[st])continue;
        stack<int>stk; stk.push(st);
        while(!stk.empty()){int u=stk.top();stk.pop(); if(vis[u])continue; vis[u]=1; order.push_back(u);
            for(int v:adj[u]) if(!vis[v]) stk.push(v);}
    }
    return order;
}

int main(){
    scanf("%d %d %d %d %d",&W,&H,&N,&M,&R);
    w.resize(N);h.resize(N);p.resize(N);adj.assign(N,{});
    int maxw=1;
    for(int i=0;i<N;i++){scanf("%lld %lld %lld",&w[i],&h[i],&p[i]); maxw=max(maxw,(int)w[i]);}
    nets.resize(M);
    for(int e=0;e<M;e++){int k;scanf("%d",&k);nets[e].resize(k);for(int j=0;j<k;j++){int a;scanf("%d",&a);nets[e][j]=a-1;}}
    for(auto&s:nets) for(int a=0;a<(int)s.size();a++) for(int b=a+1;b<(int)s.size();b++){adj[s[a]].push_back(s[b]);adj[s[b]].push_back(s[a]);}

    mt19937 rng(987654321u);

    // seed pool for BFS/DFS: highest-degree vertices + a few random
    vector<int> deg(N,0); for(int i=0;i<N;i++) deg[i]=adj[i].size();
    vector<int> byDeg(N); for(int i=0;i<N;i++) byDeg[i]=i;
    sort(byDeg.begin(),byDeg.end(),[&](int a,int b){return deg[a]>deg[b];});

    vector<vector<int>> orders;
    { vector<int> id(N); for(int i=0;i<N;i++) id[i]=i; orders.push_back(id); }
    for(int s=0;s<min(N,4);s++){ orders.push_back(bfsOrder(byDeg[s])); orders.push_back(dfsOrder(byDeg[s])); }
    for(int r=0;r<4;r++){ orders.push_back(bfsOrder(rng()%N)); }
    // power-sorted (low power first): mild thermal help
    { vector<int> ps(N); for(int i=0;i<N;i++) ps[i]=i; sort(ps.begin(),ps.end(),[&](int a,int b){return p[a]<p[b];}); orders.push_back(ps); }

    // effective widths to try
    vector<int> widths;
    for(double f : {1.0,0.8,0.65,0.55,0.45,0.38,0.32}){ int Wf=max(maxw,(int)(W*f)); widths.push_back(Wf);}
    sort(widths.begin(),widths.end()); widths.erase(unique(widths.begin(),widths.end()),widths.end());

    vector<ll> bestX, bestY; ll bestF=LLONG_MAX; vector<int> bestOrder; int bestWidth=W;
    vector<ll> x,y;
    for(auto& ord : orders) for(int Wf : widths){
        if(shelf(ord,Wf,x,y)){ ll f=objective(x,y); if(f<bestF){bestF=f;bestX=x;bestY=y;bestOrder=ord;bestWidth=Wf;} }
    }
    // fallback (should not happen): index order at full width always fits
    if(bestF==LLONG_MAX){ vector<int> id(N); for(int i=0;i<N;i++) id[i]=i; shelf(id,W,bestX,bestY); bestF=objective(bestX,bestY); bestOrder=id; bestWidth=W; }

    // local search: swap pairs in the best order, re-shelf at bestWidth, accept improvements.
    int iters = min(4000, 20*N + 500);
    vector<int> cur = bestOrder;
    for(int it=0; it<iters; it++){
        int a=rng()%N, b=rng()%N; if(a==b) continue;
        swap(cur[a],cur[b]);
        if(shelf(cur,bestWidth,x,y)){ ll f=objective(x,y);
            if(f<=bestF){ bestF=f; bestX=x; bestY=y; }
            else swap(cur[a],cur[b]);
        } else swap(cur[a],cur[b]);
    }

    for(int i=0;i<N;i++) printf("%lld %lld\n",bestX[i],bestY[i]);
    return 0;
}
