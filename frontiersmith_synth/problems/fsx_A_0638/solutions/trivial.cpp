// TIER: trivial
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
int main(){
    ll W,H,Wb,Pmax,Ymin,Ymax,K;
    cin>>W>>H>>Wb>>Pmax>>Ymin>>Ymax>>K;
    vector<ll> inflow(H);
    for (auto &v: inflow) cin>>v;
    ll dn,dd; cin>>dn>>dd;
    vector<ll> cost(Pmax+1),drag(Pmax+1),jet(Pmax+1);
    for (ll p=1;p<=Pmax;p++) cin>>cost[p];
    for (ll p=1;p<=Pmax;p++) cin>>drag[p];
    for (ll p=1;p<=Pmax;p++) cin>>jet[p];
    int M; cin>>M;
    for (int i=0;i<M;i++){ ll x,y; cin>>x>>y; }
    // plant nothing at all -- exactly the checker's own baseline construction.
    for (ll x=0;x<Wb;x++){
        for (ll y=0;y<H;y++) cout<<0<<(y+1==H?'\n':' ');
    }
    return 0;
}
