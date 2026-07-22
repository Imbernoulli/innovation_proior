// TIER: strong
// Insight: an installed depot's payoff is SUMMED over every unit of flow routed
// through it, so choosing where to install is a facility-location problem over
// the traffic you are willing to CREATE, not a property of any single path.
// For each depot, estimate the net value of deliberately funnelling its
// reachable farms' supply through it into its reachable markets (only markets
// whose post-reset price beats the depot's energy cost), rank depots by that
// net value (a value-density knapsack against the shared budget), install and
// commit the best one(s) that fit, then mop up whatever supply is left over
// with the same direct-haul fallback greedy.cpp uses.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int S, J, T, mkt0, depot0;
vector<ll> instcost, energy, cap0;
vector<vector<pair<int,int>>> tiers;          // per market: (threshold, price)
vector<vector<pair<int,int>>> depotFarms;      // per depot j: (farm, r) -- r unused (reset)
vector<vector<pair<int,int>>> depotMarkets;    // per depot j: (market, r)
vector<vector<pair<int,int>>> directs;         // per farm: (market, r)

int tierPrice(int mIdx, ll q){
    int price = 0;
    for (auto &tp : tiers[mIdx]) if (tp.first <= q) price = tp.second; else break;
    return price;
}

struct OutBlock { int farm, mkt; ll amt; int depotNode; }; // depotNode==0 -> direct

// Matches depot j's reachable farms to its reachable (profitable) markets,
// best price first. If commit==true, mutates remSupply/remCap and appends
// OutBlocks; otherwise works on private copies and only returns totals.
void runDepot(int j, vector<ll> &remSupply, vector<ll> &remCap, bool commit,
              vector<OutBlock> &blocks, ll &outRevenue, ll &outFlow){
    vector<ll> S_ = commit ? vector<ll>() : remSupply;
    vector<ll> C_ = commit ? vector<ll>() : remCap;
    vector<ll> &sup = commit ? remSupply : S_;
    vector<ll> &cp  = commit ? remCap    : C_;

    vector<pair<int,int>> mkts = depotMarkets[j]; // (market idx, r)
    sort(mkts.begin(), mkts.end(), [&](const pair<int,int>&a, const pair<int,int>&b){
        int pa = tierPrice(a.first, a.second), pb = tierPrice(b.first, b.second);
        return pa > pb;
    });
    outRevenue = 0; outFlow = 0;
    int depotNode = depot0 + j;
    for (auto &mp : mkts){
        int mIdx = mp.first, r = mp.second;
        int price = tierPrice(mIdx, r);
        if (price <= energy[j]) continue; // not worth funnelling into this market
        ll remaining = cp[mIdx];
        if (remaining <= 0) continue;
        for (auto &fp : depotFarms[j]){
            if (remaining <= 0) break;
            int farm = fp.first;
            if (sup[farm] <= 0) continue;
            ll amt = min(sup[farm], remaining);
            if (amt <= 0) continue;
            sup[farm] -= amt; cp[mIdx] -= amt; remaining -= amt;
            outRevenue += amt * (ll)price; outFlow += amt;
            if (commit) blocks.push_back({farm, mIdx, amt, depotNode});
        }
    }
}

int main(){
    ll BUD;
    scanf("%d %d %d %lld", &S, &J, &T, &BUD);
    vector<ll> supply(S + 1);
    for (int i = 1; i <= S; i++) scanf("%lld", &supply[i]);
    instcost.assign(J + 1, 0); energy.assign(J + 1, 0);
    for (int j = 1; j <= J; j++) scanf("%lld %lld", &instcost[j], &energy[j]);
    cap0.assign(T + 1, 0); tiers.assign(T + 1, {});
    for (int k = 1; k <= T; k++){
        ll c; int m; scanf("%lld %d", &c, &m);
        cap0[k] = c;
        for (int t = 0; t < m; t++){ int th, pr; scanf("%d %d", &th, &pr); tiers[k].push_back({th, pr}); }
    }
    depot0 = S; mkt0 = S + J;
    int E; scanf("%d", &E);
    depotFarms.assign(J + 1, {}); depotMarkets.assign(J + 1, {}); directs.assign(S + 1, {});
    for (int e = 0; e < E; e++){
        int u, v, r; scanf("%d %d %d", &u, &v, &r);
        if (u >= 1 && u <= S && v > depot0 && v <= mkt0) depotFarms[v - depot0].push_back({u, r});
        else if (u > depot0 && u <= mkt0 && v > mkt0) depotMarkets[u - depot0].push_back({v - mkt0, r});
        else if (u >= 1 && u <= S && v > mkt0) directs[u].push_back({v - mkt0, r});
    }

    vector<ll> remSupply(supply), remCap(cap0);
    vector<char> chosen(J + 1, 0);
    vector<int> chosenList;
    ll budgetLeft = BUD;
    vector<OutBlock> blocks;

    for (int round = 0; round < J; round++){
        int bestJ = -1; ll bestVal = 0;
        for (int j = 1; j <= J; j++){
            if (chosen[j]) continue;
            if (instcost[j] > budgetLeft) continue;
            vector<OutBlock> dummy; ll rev = 0, flow = 0;
            runDepot(j, remSupply, remCap, false, dummy, rev, flow);
            ll netVal = rev - flow * energy[j] - instcost[j];
            if (netVal > bestVal){ bestVal = netVal; bestJ = j; }
        }
        if (bestJ == -1) break;
        ll rev = 0, flow = 0;
        runDepot(bestJ, remSupply, remCap, true, blocks, rev, flow);
        chosen[bestJ] = 1; chosenList.push_back(bestJ);
        budgetLeft -= instcost[bestJ];
    }

    // fallback: direct-haul greedy for whatever supply remains
    for (int i = 1; i <= S; i++){
        if (remSupply[i] <= 0) continue;
        int bestK = -1, bestPrice = -1;
        for (auto &pr : directs[i]){
            int k = pr.first, r = pr.second;
            if (remCap[k] <= 0) continue;
            int price = tierPrice(k, r);
            if (price > bestPrice || (price == bestPrice && (bestK == -1 || k < bestK))){
                bestPrice = price; bestK = k;
            }
        }
        if (bestK == -1) continue;
        ll amt = min(remSupply[i], remCap[bestK]);
        if (amt > 0){
            blocks.push_back({i, bestK, amt, 0});
            remSupply[i] -= amt; remCap[bestK] -= amt;
        }
    }

    printf("%d\n", (int)chosenList.size());
    for (size_t t = 0; t < chosenList.size(); t++) printf("%d%c", depot0 + chosenList[t], t + 1 == chosenList.size() ? '\n' : ' ');
    if (chosenList.empty()) printf("\n");
    printf("%d\n", (int)blocks.size());
    for (auto &b : blocks){
        int mktNode = mkt0 + b.mkt;
        if (b.depotNode == 0){
            printf("%d %d %lld 2\n", b.farm, mktNode, b.amt);
            printf("%d %d\n", b.farm, mktNode);
        } else {
            printf("%d %d %lld 3\n", b.farm, mktNode, b.amt);
            printf("%d %d %d\n", b.farm, b.depotNode, mktNode);
        }
    }
    return 0;
}
