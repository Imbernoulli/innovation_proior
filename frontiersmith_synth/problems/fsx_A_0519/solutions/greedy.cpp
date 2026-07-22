// TIER: greedy
// The obvious approach: sort hottest-first and ride the free cooling curve down,
// idling the minimum into each window and heating only when overshot below lo.
// Pays almost no energy -- but blows the deadlines of cool jobs done last (the trap).
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
ll num_,den_;
ll coolT(ll T,ll k){ while(k>0&&T>0){T=(T*num_)/den_;k--;} return T; }
ll minIdleTo(ll T,ll H){ ll g=0; while(T>H){T=(T*num_)/den_;g++;} return g; }
int main(){
    int N; ll cheat;
    scanf("%d %lld %lld %lld",&N,&cheat,&num_,&den_);
    vector<ll> lo(N+1),hi(N+1),d(N+1),D(N+1),w(N+1);
    for(int i=1;i<=N;i++) scanf("%lld %lld %lld %lld %lld",&lo[i],&hi[i],&d[i],&D[i],&w[i]);
    vector<int> ord(N);
    for(int i=0;i<N;i++) ord[i]=i+1;
    sort(ord.begin(),ord.end(),[&](int a,int b){
        if(hi[a]!=hi[b]) return hi[a]>hi[b];
        return lo[a]>lo[b];
    });
    ll T=0;
    for(int id:ord){
        ll g=minIdleTo(T,hi[id]);
        T=coolT(T,g);
        ll F=max(lo[id],T);
        printf("%d %lld %lld\n", id, g, F);
        T=F;
    }
    return 0;
}
