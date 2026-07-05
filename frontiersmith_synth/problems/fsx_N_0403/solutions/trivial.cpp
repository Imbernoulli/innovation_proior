// TIER: trivial
// Reference baseline: keep the M heaviest individually-stable rigs as singleton loops.
// Reproduces the checker's B exactly -> ratio ~= 0.1.
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

    auto yof=[&](ll W,ll U){ ll b=(ll)((__int128)W*U/(__int128)(2*C)); return W+b; };

    vector<pair<ll,ll>> cand; // (objective, index)
    for(ll i=1;i<=N;i++)
        if(isNZQR(k[i],p)) cand.push_back({yof(w[i],c[i]), i});
    sort(cand.begin(),cand.end(),greater<pair<ll,ll>>());

    ll G = min((ll)cand.size(), M);
    printf("%lld\n", G);
    for(ll i=0;i<G;i++) printf("1 %lld\n", cand[i].second);
    return 0;
}
