#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// -----------------------------------------------------------------------------
// Checker / scorer for "Cold Chain with Re-Chill Depots".
//
// Nodes: farms 1..S, depot SITES S+1..S+J, markets S+J+1..S+J+T.
// Input: S J T BUD ; supply[1..S] ; (instcost,energy)[depot 1..J] ;
//        (cap,m,tiers...)[market 1..T] ; E ; E edges (u v r), u<v, r in [1,999]
//        meaning quality *= r/1000 (floor) when crossed.
// Output: Kp installed depot ids (subset of depot sites, sum(instcost)<=BUD),
//   then M flow blocks "s snk amt L" + explicit path n_1..n_L (n_1=s, n_L=snk,
//   every consecutive pair a given edge). Quality starts at 1000, decays per
//   edge, RESETS to 1000 on arrival at an installed depot (and charges
//   amt*energy[depot]). Market pays amt*price(finalQuality) via its tier table.
//
// Objective (MAX): F = revenue - sum(instcost installed) - sum(energy charges).
// Baseline B (checker-computed, no depots, no cleverness): each farm sends its
//   FULL supply to whatever market its FIRST-LISTED (input edge order) direct
//   farm->market edge reaches, filling remaining capacity and wasting the rest;
//   this is exactly what solutions/trivial.cpp reproduces -> ratio 0.1.
// Score (max): sc = min(1000, max(0, 100*F/max(1,B))); ratio = sc/1000.
// -----------------------------------------------------------------------------

int S, J, T;
ll BUD;
int depot0, mkt0, V;

int tierPrice(vector<pair<int,int>> &tv, ll q){
    int price = 0;
    for (auto &tp : tv) if (tp.first <= q) price = tp.second; else break;
    return price;
}

int main(int argc, char* argv[]){
    registerTestlibCmd(argc, argv);

    S = inf.readInt(); J = inf.readInt(); T = inf.readInt(); BUD = inf.readLong();
    depot0 = S; mkt0 = S + J; V = S + J + T;

    vector<ll> supply(S + 1, 0);
    for (int i = 1; i <= S; i++) supply[i] = inf.readLong();

    vector<ll> instcost(J + 1, 0), energy(J + 1, 0);
    for (int j = 1; j <= J; j++){ instcost[j] = inf.readLong(); energy[j] = inf.readLong(); }

    vector<ll> cap(T + 1, 0);
    vector<vector<pair<int,int>>> tiers(T + 1);
    for (int k = 1; k <= T; k++){
        cap[k] = inf.readLong();
        int m = inf.readInt();
        for (int t = 0; t < m; t++){
            int th = inf.readInt(), pr = inf.readInt();
            tiers[k].push_back({th, pr});
        }
    }

    int E = inf.readInt();
    map<pair<int,int>, int> edgeR;
    vector<int> firstDirectMkt(S + 1, 0);
    vector<int> firstDirectR(S + 1, 0);
    for (int e = 0; e < E; e++){
        int u = inf.readInt(), v = inf.readInt(), r = inf.readInt();
        if (edgeR.find({u, v}) == edgeR.end()) edgeR[{u, v}] = r; // first occurrence wins
        if (u >= 1 && u <= S && v > mkt0 && v <= V && firstDirectMkt[u] == 0){
            firstDirectMkt[u] = v - mkt0;
            firstDirectR[u] = r;
        }
    }

    // ---- internal baseline B: naive, no depots, first-listed direct edge only ----
    vector<ll> remCap(cap);
    ll B = 0;
    for (int i = 1; i <= S; i++){
        if (firstDirectMkt[i] == 0) continue;
        int k = firstDirectMkt[i];
        ll q = 1000LL * firstDirectR[i] / 1000; // starting quality 1000 -> exactly r
        ll amt = min(supply[i], remCap[k]);
        if (amt > 0){
            int price = tierPrice(tiers[k], q);
            B += amt * (ll)price;
            remCap[k] -= amt;
        }
    }
    if (B <= 0) B = 1;

    // ---- participant output ----
    int Kp = ouf.readInt(0, J, "Kp");
    vector<char> installed(J + 1, 0);
    ll sumInst = 0;
    for (int t = 0; t < Kp; t++){
        int id = ouf.readInt(depot0 + 1, depot0 + J, "depot_id");
        int jj = id - depot0;
        if (installed[jj]) quitf(_wa, "depot %d installed more than once", id);
        installed[jj] = 1;
        sumInst += instcost[jj];
    }
    if (sumInst > BUD) quitf(_wa, "install budget exceeded: %lld > BUD=%lld", sumInst, BUD);

    const int MCAP = 300000;
    const ll NODECAP = 4000000;
    int M = ouf.readInt(0, MCAP, "M");
    vector<ll> usedSupply(S + 1, 0), usedCap(T + 1, 0);
    ll revenue = 0, energyCost = 0, totalNodes = 0;

    for (int b = 0; b < M; b++){
        int s = ouf.readInt(1, S, "farm_s");
        int snk = ouf.readInt(mkt0 + 1, mkt0 + T, "market_snk");
        ll amt = ouf.readLong(1, (ll)2e9, "amt");
        int L = ouf.readInt(2, 200, "path_len");
        totalNodes += L;
        if (totalNodes > NODECAP) quitf(_wa, "total path nodes exceed %lld", NODECAP);
        vector<int> path(L);
        for (int t = 0; t < L; t++) path[t] = ouf.readInt(1, V, "path_node");
        if (path[0] != s) quitf(_wa, "path does not start at farm %d", s);
        if (path[L - 1] != snk) quitf(_wa, "path does not end at market %d", snk);

        ll Q = 1000;
        for (int t = 0; t + 1 < L; t++){
            int u = path[t], v = path[t + 1];
            auto it = edgeR.find({u, v});
            if (it == edgeR.end()) quitf(_wa, "edge (%d,%d) not present in input", u, v);
            Q = Q * (ll)it->second / 1000;
            if (v > depot0 && v <= depot0 + J && installed[v - depot0]){
                Q = 1000;
                energyCost += amt * energy[v - depot0];
            }
        }
        int price = tierPrice(tiers[snk - mkt0], Q);
        revenue += amt * (ll)price;

        usedSupply[s] += amt;
        if (usedSupply[s] > supply[s]) quitf(_wa, "farm %d supply exceeded: %lld > %lld", s, usedSupply[s], supply[s]);
        usedCap[snk - mkt0] += amt;
        if (usedCap[snk - mkt0] > cap[snk - mkt0]) quitf(_wa, "market %d capacity exceeded: %lld > %lld", snk, usedCap[snk - mkt0], cap[snk - mkt0]);
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output tokens");

    ll F = revenue - sumInst - energyCost;
    double raw = 100.0 * (double)F / (double)max((ll)1, B);
    if (!isfinite(raw)) quitf(_wa, "non-finite score");
    double sc = min(1000.0, max(0.0, raw));
    quitp(sc / 1000.0, "OK F=%lld B=%lld revenue=%lld inst=%lld energy=%lld Ratio: %.6f",
          F, B, revenue, sumInst, energyCost, sc / 1000.0);
    return 0;
}
