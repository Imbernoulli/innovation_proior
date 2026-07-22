// TIER: greedy
// The "obvious" first idea: wind is strongest in some lanes, so build the densest
// (most solid) windbreak you can afford right at the front, prioritizing the
// windiest lanes first. Ignores crop locations, drift and jet/tip effects entirely.
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

    vector<vector<int>> grid(Wb, vector<int>(H, 0));
    vector<ll> lanes;
    for (ll y=Ymin;y<=Ymax;y++) lanes.push_back(y);
    sort(lanes.begin(), lanes.end(), [&](ll a, ll b){ return inflow[a] > inflow[b]; });

    ll budget = K;
    for (ll y : lanes){
        if (budget >= cost[Pmax]){
            grid[0][y] = (int)Pmax;    // solid, front row only
            budget -= cost[Pmax];
        }
    }
    for (ll x=0;x<Wb;x++){
        for (ll y=0;y<H;y++) cout<<grid[x][y]<<(y+1==H?'\n':' ');
    }
    return 0;
}
