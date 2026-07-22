// TIER: strong
// Insight: accuracy is a currency the tour spends. Keep the near-monotone sweep
// (so each slew step is short and a fresh calibration buys near-zero drift), but
// place the K recalibrations RIGHT BEFORE the high-value clusters that gain the most
// -- co-designing routing and reset placement instead of spreading resets uniformly.
// Candidate reset points = the starts of high-value runs; a greedy simulation picks
// the K resets that maximize total yield.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int N; ll P,k; int K; ll R,Tb,SCALE,Dmax;
vector<ll> px, pv, acc;
vector<int> ord;

static inline ll accAt(ll d){ if(d<0)d=0; if(d>Dmax)d=Dmax; return acc[d]; }

ll simulate(const vector<char>& reset){
    ll cur=0, drift=0, F=0;
    for(int j=0;j<N;j++){
        if(reset[j]) drift=0;
        int id=ord[j];
        ll dist = llabs(cur - px[id]);
        drift += k*dist; cur=px[id];
        F += pv[id]*accAt(drift);
    }
    return F;
}

int main(){
    scanf("%d %lld %lld %d %lld %lld %lld %lld",&N,&P,&k,&K,&R,&Tb,&SCALE,&Dmax);
    px.assign(N+1,0); pv.assign(N+1,0);
    for(int i=1;i<=N;i++) scanf("%lld %lld",&px[i],&pv[i]);
    acc.assign(Dmax+1,0);
    for(ll d=0;d<=Dmax;d++) scanf("%lld",&acc[d]);

    ord.resize(N); for(int i=0;i<N;i++) ord[i]=i+1;
    sort(ord.begin(),ord.end(),[&](int a,int b){ if(px[a]!=px[b]) return px[a]<px[b]; return a<b;});

    // candidate reset boundaries = starts of high-value runs (value >= 20 = a cluster)
    const ll HIV = 20;
    vector<int> cand;
    for(int j=0;j<N;j++){
        ll v = pv[ord[j]];
        ll vp = (j>0)? pv[ord[j-1]] : 0;
        if(v>=HIV && vp<HIV) cand.push_back(j);
    }
    // fallback: if too few high-value run-starts, add uniform positions
    if((int)cand.size() < K){
        for(int v=1; v<=K; v++){
            int pos=(int)llround((double)v*(double)N/(double)(K+1));
            if(pos>=1 && pos<N) cand.push_back(pos);
        }
        sort(cand.begin(),cand.end()); cand.erase(unique(cand.begin(),cand.end()),cand.end());
    }

    vector<char> reset(N,0);
    int used=0;
    vector<char> taken(cand.size(),0);
    while(used<K){
        ll best=-1; int bestIdx=-1;
        for(size_t c=0;c<cand.size();c++){
            if(taken[c]) continue;
            int pos=cand[c]; if(reset[pos]) { taken[c]=1; continue; }
            reset[pos]=1;
            ll F=simulate(reset);
            reset[pos]=0;
            if(F>best){ best=F; bestIdx=(int)c; }
        }
        if(bestIdx<0) break;
        reset[cand[bestIdx]]=1; taken[bestIdx]=1; used++;
    }

    // emit
    vector<pair<int,int>> acts;
    for(int j=0;j<N;j++){
        if(reset[j]) acts.push_back({0,0});
        acts.push_back({1, ord[j]});
    }
    printf("%d\n",(int)acts.size());
    for(auto&a: acts){ if(a.first==0) printf("0\n"); else printf("1 %d\n", a.second); }
    return 0;
}
