// TIER: trivial
// Naive baseline: install no depots. Each farm (in id order) sends its full
// supply along its FIRST-LISTED direct farm->market edge, filling whatever
// capacity remains at that market and leaving any excess unrouted. This is
// exactly the checker's internal baseline construction B -> ratio ~0.1.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main(){
    int S, J, T; ll BUD;
    scanf("%d %d %d %lld", &S, &J, &T, &BUD);
    vector<ll> supply(S + 1);
    for (int i = 1; i <= S; i++) scanf("%lld", &supply[i]);
    for (int j = 1; j <= J; j++){ ll a, b; scanf("%lld %lld", &a, &b); }
    vector<ll> cap(T + 1, 0);
    for (int k = 1; k <= T; k++){
        ll c; int m; scanf("%lld %d", &c, &m);
        cap[k] = c;
        for (int t = 0; t < m; t++){ int th, pr; scanf("%d %d", &th, &pr); }
    }
    int mkt0 = S + J;
    int E; scanf("%d", &E);
    vector<int> firstMkt(S + 1, 0);
    for (int e = 0; e < E; e++){
        int u, v, r; scanf("%d %d %d", &u, &v, &r);
        if (u >= 1 && u <= S && v > mkt0 && firstMkt[u] == 0) firstMkt[u] = v - mkt0;
    }

    vector<ll> remCap(cap);
    vector<pair<int,ll>> blocks; // (farm, amt) with implicit market = firstMkt[farm]
    for (int i = 1; i <= S; i++){
        if (firstMkt[i] == 0) continue;
        int k = firstMkt[i];
        ll amt = min(supply[i], remCap[k]);
        if (amt > 0){ blocks.push_back({i, amt}); remCap[k] -= amt; }
    }

    printf("0\n");
    printf("%d\n", (int)blocks.size());
    for (auto &pr : blocks){
        int i = pr.first; ll amt = pr.second;
        int k = firstMkt[i];
        printf("%d %d %lld 2\n", i, mkt0 + k, amt);
        printf("%d %d\n", i, mkt0 + k);
    }
    return 0;
}
