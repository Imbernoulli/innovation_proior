// TIER: greedy
// The obvious one-pass heuristic: install no depots (routing never touches a
// depot, so a station would sit idle -- no point paying for it). Each farm, in
// id order, looks at ALL of its direct farm->market edges and sends its supply
// to whichever reachable market (with capacity left) currently pays the best
// price for the quality that single hop delivers. Never reconsiders that
// rerouting via a depot (and installing a station there) could beat every
// direct haul once enough farms are funnelled through it.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int tierPrice(vector<pair<int,int>> &tv, ll q){
    int price = 0;
    for (auto &tp : tv) if (tp.first <= q) price = tp.second; else break;
    return price;
}

int main(){
    int S, J, T; ll BUD;
    scanf("%d %d %d %lld", &S, &J, &T, &BUD);
    vector<ll> supply(S + 1);
    for (int i = 1; i <= S; i++) scanf("%lld", &supply[i]);
    for (int j = 1; j <= J; j++){ ll a, b; scanf("%lld %lld", &a, &b); }
    vector<ll> cap(T + 1, 0);
    vector<vector<pair<int,int>>> tiers(T + 1);
    for (int k = 1; k <= T; k++){
        ll c; int m; scanf("%lld %d", &c, &m);
        cap[k] = c;
        for (int t = 0; t < m; t++){ int th, pr; scanf("%d %d", &th, &pr); tiers[k].push_back({th, pr}); }
    }
    int mkt0 = S + J;
    int E; scanf("%d", &E);
    vector<vector<pair<int,int>>> directs(S + 1); // (market idx, r)
    for (int e = 0; e < E; e++){
        int u, v, r; scanf("%d %d %d", &u, &v, &r);
        if (u >= 1 && u <= S && v > mkt0) directs[u].push_back({v - mkt0, r});
    }

    vector<ll> remCap(cap);
    vector<pair<int,pair<int,ll>>> blocks; // (farm, (market, amt))
    for (int i = 1; i <= S; i++){
        int bestK = -1, bestPrice = -1;
        for (auto &pr : directs[i]){
            int k = pr.first, r = pr.second;
            if (remCap[k] <= 0) continue;
            ll q = r; // starting quality 1000, one hop -> quality == r
            int price = tierPrice(tiers[k], q);
            if (price > bestPrice || (price == bestPrice && (bestK == -1 || k < bestK))){
                bestPrice = price; bestK = k;
            }
        }
        if (bestK == -1) continue;
        ll amt = min(supply[i], remCap[bestK]);
        if (amt > 0){ blocks.push_back({i, {bestK, amt}}); remCap[bestK] -= amt; }
    }

    printf("0\n");
    printf("%d\n", (int)blocks.size());
    for (auto &b : blocks){
        int i = b.first, k = b.second.first; ll amt = b.second.second;
        printf("%d %d %lld 2\n", i, mkt0 + k, amt);
        printf("%d %d\n", i, mkt0 + k);
    }
    return 0;
}
