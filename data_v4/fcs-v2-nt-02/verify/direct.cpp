// Independent reference: directly apply Burnside over all 2n group elements,
// iterating every rotation d=0..n-1 (k^gcd(d,n)) and every reflection.
// O(n) per query, no divisor enumeration -- cross-checks the divisor-sum code
// for larger n (up to ~1e7) where the brute orbit-count is infeasible.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
const ll MOD = 1000000007LL;

ll power_mod(ll b, ll e, ll m){ b%=m; if(b<0)b+=m; ll r=1; while(e>0){ if(e&1) r=(__int128)r*b%m; b=(__int128)b*b%m; e>>=1;} return r; }
ll gcd_ll(ll a, ll b){ while(b){ll t=a%b;a=b;b=t;} return a; }

int main(){
    ll n,k; if(!(cin>>n>>k)) return 0;
    ll kk=k%MOD;
    ll rot=0;
    for(ll d=0; d<n; d++){
        ll g=gcd_ll(d,n);
        rot=(rot+power_mod(kk,g,MOD))%MOD;
    }
    ll refl=0;
    if(n%2==1){
        for(ll a=0;a<n;a++) refl=(refl+power_mod(kk,(n+1)/2,MOD))%MOD;
    }else{
        ll half=n/2;
        // n/2 axes vertex-vertex: k^(half+1); n/2 axes edge-edge: k^half
        for(ll a=0;a<half;a++){
            refl=(refl+power_mod(kk,half+1,MOD))%MOD;
            refl=(refl+power_mod(kk,half,MOD))%MOD;
        }
    }
    ll total=(rot+refl)%MOD;
    ll denom=(2*(n%MOD))%MOD;
    ll ans=total%MOD*power_mod(denom,MOD-2,MOD)%MOD;
    cout<<ans<<"\n";
    return 0;
}
