#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// -----------------------------------------------------------------------------
// "Cold Chain with Re-Chill Depots"  (generator)  family: attrition-routing-upgraders
//
// Nodes: S farms (1..S), J depot SITES (S+1..S+J), T markets (S+J+1..S+J+T).
// Edges: farm->depot, depot->market (both directions always present for a
// designated "hub" depot, sparse for "decoy" depots), and a sparse set of
// farm->market direct-haul edges.
//
// PLANTED STRUCTURE (checker never sees these labels):
//  - direct-haul decay in [780,860]  -> quality ~800-860, mid/upper tier, no cost.
//  - farm->depot decay in [600,760]  -> IRRELEVANT if the depot gets installed
//    (quality resets to 1000 on arrival), but if NOT installed, using a depot as
//    a bare pass-through is worse than a direct haul (600-760 * 930-990 ~= 560-750,
//    typically a LOWER tier than the direct haul's ~800-860).
//  - depot->market decay in [930,990] -> after a reset, quality lands ~930-990,
//    the TOP tier at any market that offers one.
//  So: routing THROUGH an uninstalled depot is a trap (worse than direct haul);
//  routing through an INSTALLED depot dominates (better than direct haul). One
//  depot (the "hub") is wired to almost every farm and market; the rest ("decoys")
//  are wired to only a small subset and are individually cheaper to install --
//  a budget that exactly affords the hub alone, but also affords a few decoys,
//  is a genuine facility-location trade-off (installing decoys wastes the budget
//  on low-traffic sites; the hub's fixed cost is amortized over everyone).
// -----------------------------------------------------------------------------

struct Edge { int u, v, r; };

int main(int argc, char* argv[]){
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);
    double f = (testId - 1) / 9.0;

    int S = 8 + (int)llround(f * 42.0);     // 8..50
    int J = 2 + (testId % 5) + (int)llround(f * 3.0); // 2..~10, keep small
    if (J > 10) J = 10;
    int T = 8 + (int)llround(f * 46.0);     // 8..54

    int depot0 = S;              // depot ids: depot0+1 .. depot0+J
    int mkt0   = S + J;          // market ids: mkt0+1 .. mkt0+T

    // ---- supplies / capacities ----
    vector<ll> supply(S + 1, 0);
    ll totalSupply = 0;
    for (int i = 1; i <= S; i++){ supply[i] = 18 + rnd.next(0, 55); totalSupply += supply[i]; }

    vector<ll> cap(T + 1, 0);
    for (int k = 1; k <= T; k++) cap[k] = 22 + rnd.next(0, 70);

    // ---- market tier schedules ----
    vector<vector<pair<int,int>>> tiers(T + 1); // (threshold, price) ascending
    for (int k = 1; k <= T; k++){
        int m = rnd.next(0, 2); // 0: 2 tiers, 1,2: 3 tiers -> most markets have a premium tier
        int p500 = 2 + rnd.next(0, 2);
        int p750 = p500 + 3 + rnd.next(0, 3);
        tiers[k].push_back({500, p500});
        tiers[k].push_back({750, p750});
        if (m != 0){
            int p900 = p750 + 4 + rnd.next(0, 4);
            tiers[k].push_back({900, p900});
        }
    }

    // ---- depot roles: one hub, rest decoys ----
    int hub = 1 + rnd.next(0, J - 1); // 1-indexed among depots
    vector<ll> instcost(J + 1, 0), energy(J + 1, 0);
    ll instHub = 8LL * S + 40 + rnd.next(0, 30);
    for (int j = 1; j <= J; j++){
        if (j == hub){
            instcost[j] = instHub;
            energy[j] = 1 + rnd.next(0, 2);
        } else {
            instcost[j] = (ll)llround(instHub * (0.25 + 0.05 * rnd.next(0, 8))); // 0.25..0.65x
            if (instcost[j] < 5) instcost[j] = 5;
            energy[j] = 2 + rnd.next(0, 3);
        }
    }
    ll BUD = instHub + 5 + rnd.next(0, 15); // affords the hub alone (+ tiny slack)

    // ---- edges ----
    vector<Edge> edges;

    // farm -> depot
    for (int j = 1; j <= J; j++){
        int dNode = depot0 + j;
        if (j == hub){
            for (int i = 1; i <= S; i++){
                if (rnd.next(0, 99) < 92){
                    int r = 600 + rnd.next(0, 160);
                    edges.push_back({i, dNode, r});
                }
            }
        } else {
            int cnt = max(1, (int)llround(S * (0.10 + 0.15 * rnd.next(0, 5) / 5.0)));
            vector<int> ids(S); iota(ids.begin(), ids.end(), 1);
            for (int t = 0; t < S - 1; t++) swap(ids[t], ids[t + rnd.next(0, S - 1 - t)]);
            cnt = min(cnt, S);
            for (int c = 0; c < cnt; c++){
                int r = 600 + rnd.next(0, 160);
                edges.push_back({ids[c], dNode, r});
            }
        }
    }
    // depot -> market
    for (int j = 1; j <= J; j++){
        int dNode = depot0 + j;
        if (j == hub){
            for (int k = 1; k <= T; k++){
                if (rnd.next(0, 99) < 92){
                    int r = 930 + rnd.next(0, 60);
                    edges.push_back({dNode, mkt0 + k, r});
                }
            }
        } else {
            int cnt = max(1, (int)llround(T * (0.08 + 0.12 * rnd.next(0, 5) / 5.0)));
            vector<int> ids(T); iota(ids.begin(), ids.end(), 1);
            for (int t = 0; t < T - 1; t++) swap(ids[t], ids[t + rnd.next(0, T - 1 - t)]);
            cnt = min(cnt, T);
            for (int c = 0; c < cnt; c++){
                int r = 930 + rnd.next(0, 60);
                edges.push_back({dNode, mkt0 + ids[c], r});
            }
        }
    }
    // farm -> market direct haul: a deliberately WEAK "decoy" edge per farm
    // (placed first in the final edge order below, so the checker's naive
    // trivial baseline always latches onto it) plus a sparser set of GENUINE
    // direct-haul edges with meaningfully better, price-varying quality that
    // an alert one-pass greedy will actually search out and prefer.
    double directFrac = 0.30 + 0.15 * (rnd.next(0, 100) / 100.0);
    vector<set<int>> usedMkt(S + 1);
    for (int i = 1; i <= S; i++){
        for (int k = 1; k <= T; k++){
            if (rnd.next(0, 999) < (int)(directFrac * 1000)){
                int r = 700 + rnd.next(0, 220); // 700..920: real price variance
                edges.push_back({i, mkt0 + k, r});
                usedMkt[i].insert(k);
            }
        }
    }
    // decoy: one deliberately WEAK direct edge per farm, on a market that farm
    // does NOT already reach via a genuine edge (no duplicate (farm,market)
    // pairs, so the checker's edge-decay map is unambiguous).
    vector<Edge> decoys;
    for (int i = 1; i <= S; i++){
        int k = -1;
        for (int tries = 0; tries < 30; tries++){
            int cand = 1 + rnd.next(0, T - 1);
            if (!usedMkt[i].count(cand)){ k = cand; break; }
        }
        if (k == -1){ for (int cand = 1; cand <= T; cand++) if (!usedMkt[i].count(cand)){ k = cand; break; } }
        if (k == -1) continue; // farm connects to every market already -- no room for a decoy
        int r = (i == 1) ? (640 + rnd.next(0, 60)) : (600 + rnd.next(0, 160)); // farm1 stays B>0-safe
        decoys.push_back({i, mkt0 + k, r});
        usedMkt[i].insert(k);
    }

    // shuffle the non-decoy edges among themselves, then PREPEND the decoys
    // (in farm order) so each farm's checker/trivial "first-listed direct
    // edge" is deterministically its decoy, never one of the better options.
    for (int t = (int)edges.size() - 1; t > 0; t--) swap(edges[t], edges[rnd.next(0, t)]);
    vector<Edge> allEdges;
    allEdges.reserve(decoys.size() + edges.size());
    for (auto &e : decoys) allEdges.push_back(e);
    for (auto &e : edges) allEdges.push_back(e);
    edges.swap(allEdges);

    printf("%d %d %d %lld\n", S, J, T, BUD);
    for (int i = 1; i <= S; i++) printf("%lld\n", supply[i]);
    for (int j = 1; j <= J; j++) printf("%lld %lld\n", instcost[j], energy[j]);
    for (int k = 1; k <= T; k++){
        printf("%lld %d", cap[k], (int)tiers[k].size());
        for (auto &tp : tiers[k]) printf(" %d %d", tp.first, tp.second);
        printf("\n");
    }
    printf("%d\n", (int)edges.size());
    for (auto &e : edges) printf("%d %d %d\n", e.u, e.v, e.r);
    return 0;
}
