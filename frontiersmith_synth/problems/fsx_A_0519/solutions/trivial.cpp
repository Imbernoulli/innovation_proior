// TIER: trivial
// input order, minimal idle into window, heat only up to lo_i  (== checker baseline B)
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
    ll T=0;
    for(int i=1;i<=N;i++){
        ll g=minIdleTo(T,hi[i]);
        T=coolT(T,g);
        ll F=max(lo[i],T);
        printf("%d %lld %lld\n", i, g, F);
        T=F;
    }
    return 0;
}
