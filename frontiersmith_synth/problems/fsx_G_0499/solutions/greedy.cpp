// TIER: greedy
// Affinity-first growth: seed with the best single fragment, then add fragments
// in decreasing b_i, each connected by one bond to the already-selected fragment
// with the most synergistic interaction and a free valence; accept only if the
// marginal change (including the weight-overflow penalty) is positive. Finish by
// adding every ring-closing bond whose q + Rb > 0.
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

    auto over=[&](ll W){ return max((ll)0, W-Wmax); };

    // seed = best single fragment
    ll seed=1, best=LLONG_MIN;
    for(ll i=1;i<=M;i++){ ll v=b[i]-Lam*over(w[i]); if(v>best){best=v;seed=i;} }

    vector<char> chosen(M+1,0);
    vector<ll> rem(M+1); for(ll i=1;i<=M;i++) rem[i]=val[i];
    vector<ll> selList;
    ll W=0;
    auto addFrag=[&](ll j){ chosen[j]=1; selList.push_back(j); W+=w[j]; };
    addFrag(seed);

    // candidate order: b descending
    vector<ll> order;
    for(ll i=1;i<=M;i++) if(i!=seed) order.push_back(i);
    sort(order.begin(),order.end(),[&](ll x,ll y){ return b[x]>b[y]; });

    vector<pair<ll,ll>> bonds;
    set<pair<ll,ll>> bset;
    auto addBond=[&](ll a,ll c){
        ll lo=min(a,c),hi=max(a,c);
        bonds.push_back({a,c}); bset.insert({lo,hi});
        rem[a]--; rem[c]--;
    };

    for(ll j: order){
        if(chosen[j]) continue;
        ll bi=-1, bq=LLONG_MIN;
        for(ll i: selList) if(rem[i]>0){ ll v=q[p[i]][p[j]]; if(v>bq){bq=v;bi=i;} }
        if(bi<0) continue;
        ll delta = b[j] + bq - Lam*( over(W+w[j]) - over(W) );
        if(delta>0){ addFrag(j); addBond(bi,j); }
    }

    // ring-closing bonds: any spare-valence pair with q + Rb > 0
    ll ns=(ll)selList.size();
    for(ll x=0;x<ns;x++) for(ll y=x+1;y<ns;y++){
        ll a=selList[x], c=selList[y];
        if(rem[a]>0 && rem[c]>0){
            ll lo=min(a,c),hi=max(a,c);
            if(!bset.count({lo,hi}) && q[p[a]][p[c]] + Rb > 0) addBond(a,c);
        }
    }

    // output
    printf("%lld\n",(ll)selList.size());
    for(size_t i=0;i<selList.size();i++) printf("%lld%c", selList[i], i+1==selList.size()?'\n':' ');
    printf("%lld\n",(ll)bonds.size());
    for(auto&e:bonds) printf("%lld %lld\n", e.first, e.second);
    return 0;
}
