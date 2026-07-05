// TIER: greedy
// Connectivity-ordered shelf packing: order blocks by BFS over the net-adjacency graph so that
// connected blocks land next to each other in the shelf (lower HPWL). Return the better of the
// index-order shelf and the connectivity-order shelf (so it never loses to trivial).
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int N, M, W, H, R;
vector<ll> w, h, p;
vector<vector<int>> nets, adj;

ll objective(const vector<ll>& x, const vector<ll>& y) {
    vector<ll> cx(N), cy(N);
    for (int i = 0; i < N; i++) { cx[i] = 2*x[i]+w[i]; cy[i] = 2*y[i]+h[i]; }
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

void shelf(const vector<int>& order, vector<ll>& x, vector<ll>& y) {
    x.assign(N,0); y.assign(N,0);
    ll cx=0, cy=0, rowh=0;
    for (int idx=0; idx<N; idx++){ int i=order[idx];
        if (cx + w[i] > W){ cy += rowh; cx=0; rowh=0; }
        x[i]=cx; y[i]=cy; cx+=w[i]; rowh=max(rowh,h[i]);
    }
}

int main(){
    scanf("%d %d %d %d %d",&W,&H,&N,&M,&R);
    w.resize(N);h.resize(N);p.resize(N);adj.assign(N,{});
    for(int i=0;i<N;i++) scanf("%lld %lld %lld",&w[i],&h[i],&p[i]);
    nets.resize(M);
    for(int e=0;e<M;e++){int k;scanf("%d",&k);nets[e].resize(k);for(int j=0;j<k;j++){int a;scanf("%d",&a);nets[e][j]=a-1;}}
    // adjacency
    for(auto&s:nets) for(int a=0;a<(int)s.size();a++) for(int b=a+1;b<(int)s.size();b++){adj[s[a]].push_back(s[b]);adj[s[b]].push_back(s[a]);}

    // BFS order
    vector<int> order; order.reserve(N);
    vector<char> vis(N,0);
    for(int st=0; st<N; st++) if(!vis[st]){
        queue<int>q; q.push(st); vis[st]=1;
        while(!q.empty()){int u=q.front();q.pop();order.push_back(u);
            for(int v:adj[u]) if(!vis[v]){vis[v]=1;q.push(v);} }
    }

    vector<int> idxOrder(N); for(int i=0;i<N;i++) idxOrder[i]=i;

    vector<ll> x1,y1,x2,y2;
    shelf(idxOrder,x1,y1);
    shelf(order,x2,y2);
    ll f1=objective(x1,y1), f2=objective(x2,y2);
    vector<ll>&X = (f2<f1)?x2:x1; vector<ll>&Y=(f2<f1)?y2:y1;
    for(int i=0;i<N;i++) printf("%lld %lld\n",X[i],Y[i]);
    return 0;
}
