// TIER: greedy
// The obvious recipe: fill every target well directly from stock+diluent,
// chunked at Vmax. Near-perfect accuracy and volume, but every well touches
// the stock reservoir directly, so D = W -- this is the trap the problem sets.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main(){
    ll W, M, Vmax, Vcap, stepCost, stockAccessCost, maxOps;
    cin >> W >> M >> Vmax >> Vcap >> stepCost >> stockAccessCost >> maxOps;
    vector<ll> c(W + 1), Vreq(W + 1);
    for (ll i = 1; i <= W; i++) cin >> c[i] >> Vreq[i];

    vector<array<ll,3>> ops; // src dst vol

    for (ll i = 1; i <= W; i++){
        ll stockVol = (Vreq[i] * c[i]) / 1000; // integer rounding, tiny deviation
        ll dilVol = Vreq[i] - stockVol;
        ll rem = stockVol;
        while (rem > 0){
            ll chunk = min(Vmax, rem);
            ops.push_back({-1, i, chunk});
            rem -= chunk;
        }
        rem = dilVol;
        while (rem > 0){
            ll chunk = min(Vmax, rem);
            ops.push_back({-2, i, chunk});
            rem -= chunk;
        }
    }

    cout << ops.size() << "\n";
    for (auto &o : ops) cout << o[0] << " " << o[1] << " " << o[2] << "\n";
    return 0;
}
