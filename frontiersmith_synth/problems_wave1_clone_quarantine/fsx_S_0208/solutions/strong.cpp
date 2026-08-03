// TIER: strong
// Profitable-set selection + capacity/precedence nearest-neighbor interleaving.
// Selects the same "solo-profitable" tasks as greedy, but routes them with a
// nearest-available-event heuristic that batches probes on the rack, cutting the
// Chebyshev travel well below the serialize-each greedy tour. A light or-opt-style
// re-scan of the served set (drop any task whose realized marginal exceeds its
// penalty) is applied afterwards.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
static inline ll cheb(ll ax, ll ay, ll bx, ll by){ return max(llabs(ax-bx), llabs(ay-by)); }

int P; ll Q, X0, Y0;
vector<ll> px,py,dx,dy,q,c,w;

// build a NN interleaved tour over a chosen set; returns events and total F contribution
struct Plan { vector<pair<int,int>> ev; ll dist; };

Plan route(const vector<int>& set){
    Plan pl; pl.dist=0;
    int n = set.size();
    vector<char> mounted(n,0), read(n,0);
    ll cx=X0, cy=Y0, load=0;
    int done=0;
    while (done < n){
        int best=-1, bestType=-1; ll bestCost=LLONG_MAX;
        for (int k=0;k<n;k++){
            int i=set[k];
            if (!mounted[k]){
                if (load + q[i] <= Q){
                    ll cost = cheb(cx,cy,px[i],py[i]);
                    if (cost < bestCost){ bestCost=cost; best=k; bestType=0; }
                }
            } else if (!read[k]){
                ll cost = cheb(cx,cy,dx[i],dy[i]);
                if (cost < bestCost){ bestCost=cost; best=k; bestType=1; }
            }
        }
        if (best<0) break; // should not happen (a read is always available)
        int i=set[best];
        if (bestType==0){
            mounted[best]=1; load+=q[i];
            pl.dist += cheb(cx,cy,px[i],py[i]); cx=px[i]; cy=py[i];
            pl.ev.push_back({0,i});
        } else {
            read[best]=1; load-=q[i]; done++;
            pl.dist += cheb(cx,cy,dx[i],dy[i]); cx=dx[i]; cy=dy[i];
            pl.ev.push_back({1,i});
        }
    }
    if (!pl.ev.empty()) pl.dist += cheb(cx,cy,X0,Y0);
    return pl;
}

ll objective(const Plan& pl, const vector<int>& set){
    // F = dist + sum served c_i + sum omitted w_i
    ll totW=0; for (int i=0;i<P;i++) totW += w[i];
    ll instr=0, servedPen=0;
    vector<char> in(P,0);
    for (int i: set){ in[i]=1; instr += c[i]; }
    ll pen=0; for (int i=0;i<P;i++) if(!in[i]) pen += w[i];
    return pl.dist + instr + pen;
}

int main(){
    scanf("%d %lld",&P,&Q);
    scanf("%lld %lld",&X0,&Y0);
    px.resize(P);py.resize(P);dx.resize(P);dy.resize(P);q.resize(P);c.resize(P);w.resize(P);
    for(int i=0;i<P;i++)
        scanf("%lld %lld %lld %lld %lld %lld %lld",&px[i],&py[i],&dx[i],&dy[i],&q[i],&c[i],&w[i]);

    // initial profitable set (same conservative test as greedy)
    vector<int> set;
    for(int i=0;i<P;i++){
        ll solo = cheb(X0,Y0,px[i],py[i]) + cheb(px[i],py[i],dx[i],dy[i])
                + cheb(dx[i],dy[i],X0,Y0) + c[i];
        if (w[i] > solo) set.push_back(i);
    }

    Plan pl = route(set);
    ll bestF = objective(pl, set);

    // one single-pass or-opt prune: drop any task whose removal improves F.
    // (single pass -> O(n) reroutes, safe on time; still refines selection.)
    {
        vector<int> keep;
        for (size_t idx=0; idx<set.size(); idx++){
            vector<int> cand;
            for (size_t k=0;k<set.size();k++) if(k!=idx) cand.push_back(set[k]);
            Plan p2 = route(cand);
            ll f2 = objective(p2, cand);
            if (f2 >= bestF) keep.push_back(set[idx]);
        }
        if (keep.size() != set.size()){
            set = keep;
            pl = route(set);
            bestF = objective(pl, set);
        }
    }

    printf("%d\n", (int)pl.ev.size());
    for (auto& e: pl.ev) printf("%d %d\n", e.first, e.second+1);
    return 0;
}
