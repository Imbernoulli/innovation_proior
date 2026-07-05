// TIER: trivial
// Baseline: output the single fragment maximizing b_i - Lam*max(0,w_i-Wmax).
// This equals the checker's B, so ratio = 0.1 exactly.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
int main(){
    ll M,C,Wmax,Lam,Rb;
    if(!(cin>>M>>C>>Wmax>>Lam>>Rb)) return 0;
    vector<ll> w(M+1),b(M+1),val(M+1),p(M+1);
    for(ll i=1;i<=M;i++) cin>>w[i]>>b[i]>>val[i]>>p[i];
    vector<vector<ll>> q(C, vector<ll>(C));
    for(ll a=0;a<C;a++) for(ll c=0;c<C;c++) cin>>q[a][c];
    ll best=LLONG_MIN, bi=1;
    for(ll i=1;i<=M;i++){
        ll over=max((ll)0,w[i]-Wmax);
        ll v=b[i]-Lam*over;
        if(v>best){best=v;bi=i;}
    }
    printf("1\n%lld\n0\n", bi);
    return 0;
}
