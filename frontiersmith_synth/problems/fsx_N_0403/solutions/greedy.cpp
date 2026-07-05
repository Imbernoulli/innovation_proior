// TIER: greedy
// Weight-greedy grouping: sort rigs by weight desc, pour into loops up to the coolant
// budget, then repair the residue lock by dropping the single lightest rig that fixes it.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

static ll powmod(ll a, ll e, ll m){ a%=m; if(a<0)a+=m; ll r=1%m; while(e){ if(e&1) r=(__int128)r*a%m; a=(__int128)a*a%m; e>>=1;} return r; }
static bool isNZQR(ll s, ll p){ s%=p; if(s<0)s+=p; if(s==0) return false; return powmod(s,(p-1)/2,p)==1; }

int main(){
    ll N,M,C,p;
    if(scanf("%lld %lld %lld %lld",&N,&M,&C,&p)!=4) return 0;
    vector<ll> w(N+1),c(N+1),k(N+1);
    for(ll i=1;i<=N;i++) scanf("%lld %lld %lld",&w[i],&c[i],&k[i]);

    vector<ll> ord(N);
    for(ll i=0;i<N;i++) ord[i]=i+1;
    sort(ord.begin(),ord.end(),[&](ll a,ll b){ return w[a]>w[b]; });

    vector<vector<ll>> loops;
    ll ptr=0;
    while((ll)loops.size()<M && ptr<N){
        // fill one loop up to the coolant budget
        vector<ll> cur;
        ll U=0, ksum=0;
        while(ptr<N){
            ll id=ord[ptr];
            if(U + c[id] > C){ ptr++; if(!cur.empty()) break; else continue; }
            cur.push_back(id); U+=c[id]; ksum=(ksum+k[id])%p; ptr++;
            if((ll)cur.size()>=6) break;
        }
        if(cur.empty()) continue;
        // repair residue lock: if unstable, drop the lightest rig whose removal makes it stable
        if(!isNZQR(ksum,p)){
            ll bestDrop=-1; ll bestW=LLONG_MAX;
            for(ll id : cur){
                ll ns=((ksum-k[id])%p+p)%p;
                if(isNZQR(ns,p) && w[id]<bestW){ bestW=w[id]; bestDrop=id; }
            }
            if(bestDrop!=-1){
                vector<ll> nc; for(ll id:cur) if(id!=bestDrop) nc.push_back(id);
                cur.swap(nc);
            } else {
                // fall back to the single heaviest stable rig in the group, if any
                ll pick=-1;
                for(ll id:cur) if(isNZQR(k[id],p) && (pick==-1 || w[id]>w[pick])) pick=id;
                if(pick==-1){ continue; }
                cur.clear(); cur.push_back(pick);
            }
        }
        if(!cur.empty()) loops.push_back(cur);
    }

    printf("%lld\n",(ll)loops.size());
    for(auto& L:loops){
        printf("%lld",(ll)L.size());
        for(ll id:L) printf(" %lld",id);
        printf("\n");
    }
    return 0;
}
