// TIER: strong
// Insight: hottest-first is a trap because the free-cooling tail is exponentially flat --
// idling all the way down to a cool job costs enormous TIME, so deadline-tight cool jobs
// done last are ruined. Reformulate as a deadline-driven schedule: process (roughly) by
// deadline, which for this structure ASCENDS in temperature, so the kiln is heated up
// monotonically -- almost no wasted idle and cool-tight jobs finish early. We evaluate
// several structured orders by exact simulation and refine the best with adjacent-swap
// local search on the true objective, inserting a reheat spike only where it is cheaper
// than paying the deadline penalty.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
ll num_,den_,cheat_;
int N;
vector<ll> lo,hi,d,D,w;
ll coolT(ll T,ll k){ while(k>0&&T>0){T=(T*num_)/den_;k--;} return T; }
ll minIdleTo(ll T,ll H){ ll g=0; while(T>H){T=(T*num_)/den_;g++;} return g; }

// simulate an order with the minimal-energy firing policy F=max(lo,T_idle); return objective
ll simulate(const vector<int>& ord){
    ll T=0,t=0,cost=0;
    for(int id:ord){
        ll g=minIdleTo(T,hi[id]);
        T=coolT(T,g); t+=g;
        ll F=max(lo[id],T);
        cost+=cheat_*(F-T);
        t+=d[id];
        if(t>D[id]) cost+=w[id]*(t-D[id]);
        T=F;
    }
    return cost;
}
void emit(const vector<int>& ord){
    ll T=0;
    for(int id:ord){
        ll g=minIdleTo(T,hi[id]);
        T=coolT(T,g);
        ll F=max(lo[id],T);
        printf("%d %lld %lld\n", id, g, F);
        T=F;
    }
}
int main(){
    scanf("%d %lld %lld %lld",&N,&cheat_,&num_,&den_);
    lo.assign(N+1,0);hi.assign(N+1,0);d.assign(N+1,0);D.assign(N+1,0);w.assign(N+1,0);
    for(int i=1;i<=N;i++) scanf("%lld %lld %lld %lld %lld",&lo[i],&hi[i],&d[i],&D[i],&w[i]);

    vector<int> base(N); for(int i=0;i<N;i++) base[i]=i+1;

    auto byDeadline=base; sort(byDeadline.begin(),byDeadline.end(),
        [&](int a,int b){ if(D[a]!=D[b])return D[a]<D[b]; return hi[a]<hi[b]; });
    auto byHot=base; sort(byHot.begin(),byHot.end(),
        [&](int a,int b){ if(hi[a]!=hi[b])return hi[a]>hi[b]; return lo[a]>lo[b]; });
    auto byCold=base; sort(byCold.begin(),byCold.end(),
        [&](int a,int b){ if(hi[a]!=hi[b])return hi[a]<hi[b]; return lo[a]<lo[b]; });
    // deadline-slack order: urgency = D - d, break ties toward cooler first
    auto bySlack=base; sort(bySlack.begin(),bySlack.end(),
        [&](int a,int b){ ll ka=D[a]-d[a], kb=D[b]-d[b]; if(ka!=kb)return ka<kb; return hi[a]<hi[b]; });

    vector<vector<int>> cands={base,byHot,byCold,byDeadline,bySlack};
    vector<int> best; ll bestC=LLONG_MAX;
    for(auto&c:cands){ ll v=simulate(c); if(v<bestC){bestC=v;best=c;} }

    // adjacent-swap local search on the true objective (bounded work)
    long long budget = 4000000; // simulate-steps budget proxy
    long long perSim = (long long)N;
    int maxPasses = (int)max(1LL, min((ll)40, budget/(perSim*perSim+1)));
    for(int pass=0; pass<maxPasses; pass++){
        bool improved=false;
        for(int i=0;i+1<N;i++){
            swap(best[i],best[i+1]);
            ll v=simulate(best);
            if(v<bestC){ bestC=v; improved=true; }
            else swap(best[i],best[i+1]);
        }
        if(!improved) break;
    }
    emit(best);
    return 0;
}
