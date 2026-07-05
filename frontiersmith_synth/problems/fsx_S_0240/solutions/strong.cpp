// TIER: strong
// Local-search consolidation. Build the EDF baseline and the cost-aware greedy, take the
// cheaper as start, then repeatedly move single tasks to a cheaper in-window step until no
// improving move remains, over several fixed-seed restart orders. Never worse than start,
// so never worse than the EDF baseline B.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int T, N, M; ll G;
vector<ll> f, s; vector<int> cap;
vector<int> r, dl;

static inline ll stepCost(int t, int y) {
    if (y <= 0) return 0;
    ll c = f[t];
    int sp = min(y, cap[t]);
    c += s[t] * (ll)sp + G * (ll)(y - sp);
    return c;
}
static ll totalCost(const vector<int>& y) {
    ll tot = 0; for (int t = 1; t <= T; t++) tot += stepCost(t, y[t]); return tot;
}

// build EDF baseline assignment
vector<int> edf() {
    vector<int> order(N); iota(order.begin(), order.end(), 0);
    sort(order.begin(), order.end(), [&](int a, int b){
        if (dl[a]!=dl[b]) return dl[a]<dl[b];
        if (r[a]!=r[b]) return r[a]<r[b];
        return a<b; });
    vector<int> y(T+1,0), asg(N,-1);
    for (int idx: order){
        for (int t=r[idx]; t<=dl[idx]; t++) if (y[t]<M){asg[idx]=t;y[t]++;break;}
        if (asg[idx]<0) asg[idx]=r[idx];
    }
    return asg;
}
// cost-aware greedy assignment
vector<int> greedy() {
    vector<int> order(N); iota(order.begin(), order.end(), 0);
    sort(order.begin(), order.end(), [&](int a, int b){
        if (dl[a]!=dl[b]) return dl[a]<dl[b];
        if (r[a]!=r[b]) return r[a]<r[b];
        return a<b; });
    vector<int> y(T+1,0), asg(N,-1);
    for (int idx: order){
        int best=-1; ll bd=LLONG_MAX;
        for (int t=r[idx]; t<=dl[idx]; t++){
            if (y[t]>=M) continue;
            ll d=stepCost(t,y[t]+1)-stepCost(t,y[t]);
            if (d<bd){bd=d;best=t;}
        }
        if (best<0){ for(int t=r[idx];t<=dl[idx];t++) if(y[t]<M){best=t;break;} if(best<0)best=r[idx]; }
        asg[idx]=best; if(best>=1&&best<=T) y[best]++;
    }
    return asg;
}

int main() {
    if (scanf("%d %d %d", &T, &N, &M) != 3) return 0;
    scanf("%lld", &G);
    f.assign(T+1,0); s.assign(T+1,0); cap.assign(T+1,0);
    for (int t=1;t<=T;t++) scanf("%lld %lld %d",&f[t],&s[t],&cap[t]);
    r.assign(N,0); dl.assign(N,0);
    for (int i=0;i<N;i++) scanf("%d %d",&r[i],&dl[i]);

    vector<int> a1=edf(), a2=greedy();
    vector<int> yb(T+1,0); for(int i=0;i<N;i++) yb[a1[i]]++;
    vector<int> yg(T+1,0); for(int i=0;i<N;i++) yg[a2[i]]++;
    vector<int> asg = (totalCost(yg) < totalCost(yb)) ? a2 : a1;

    vector<int> y(T+1,0); for(int i=0;i<N;i++) y[asg[i]]++;

    // local search with fixed-seed restart orders
    mt19937 rng(12345);
    vector<int> idxOrder(N); iota(idxOrder.begin(), idxOrder.end(), 0);
    for (int restart=0; restart<4; restart++){
        if (restart>0) shuffle(idxOrder.begin(), idxOrder.end(), rng);
        bool improved=true; int guard=0;
        while (improved && guard<60){
            improved=false; guard++;
            for (int oi=0; oi<N; oi++){
                int i=idxOrder[oi];
                int cur=asg[i];
                ll curPair = stepCost(cur, y[cur]); // cost with i present
                ll removeGain = curPair - stepCost(cur, y[cur]-1); // cost saved removing i
                int bestT=cur; ll bestDelta=0;
                for (int t=r[i]; t<=dl[i]; t++){
                    if (t==cur) continue;
                    if (y[t]>=M) continue;
                    ll addCost = stepCost(t, y[t]+1) - stepCost(t, y[t]);
                    ll delta = addCost - removeGain; // change in total if we move i to t
                    if (delta < bestDelta){ bestDelta=delta; bestT=t; }
                }
                if (bestT!=cur){
                    y[cur]--; y[bestT]++; asg[i]=bestT; improved=true;
                }
            }
        }
    }
    for (int i=0;i<N;i++) printf("%d\n", asg[i]);
    return 0;
}
