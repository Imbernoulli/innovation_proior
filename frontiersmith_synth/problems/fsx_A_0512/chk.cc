#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// -----------------------------------------------------------------------------
// Checker / scorer for "Firebreak Grid".
//
// Input:  L M S B ; L lines (w v) ; M reroute edges (u j num) ; S scenario lines.
// Output: L integer capacities cap_i, each in [0,B], with sum <= B.
//
// Objective (maximize): F = min over the S scenarios of served(t), where a cascade
// is simulated (see runScenario) and served = totalValue - lost value of tripped.
// Baseline Bsl = same worst-case served for the UNIFORM grid cap_i = floor(B/L).
// Score (max): sc = min(1000, 100 * F / max(1,Bsl)); ratio = sc/1000.
//   uniform -> F=Bsl -> 0.1 ; better hardening -> higher, capped at 1.0 (10x).
// -----------------------------------------------------------------------------

static const ll DEN = 1000;

int L, M, S;
ll B;
vector<ll> W, Vv;
vector<vector<pair<int,int>>> out;   // per line: (target, num)
vector<int> scen;
ll totalValue;

// Deterministic, order-independent round-based cascade.
ll runScenario(const vector<ll>& cap, int trip){
    static vector<ll> load;
    static vector<char> tripped, dist;
    load.assign(L, 0);
    tripped.assign(L, 0);
    dist.assign(L, 0);
    for (int i = 0; i < L; i++) load[i] = W[i];
    vector<int> pend;
    tripped[trip] = 1; pend.push_back(trip);
    for (int i = 0; i < L; i++)
        if (!tripped[i] && load[i] > cap[i]){ tripped[i] = 1; pend.push_back(i); }
    while (!pend.empty()){
        // distribute everyone tripped in the previous round exactly once
        vector<int> affected;
        for (int u : pend){
            if (dist[u]) continue;
            dist[u] = 1;
            for (auto& e : out[u]){
                int j = e.first;
                if (tripped[j]) continue;                 // shed onto dead line
                load[j] += load[u] * (ll)e.second / DEN;   // integer reroute
                affected.push_back(j);
            }
        }
        // threshold: newly overloaded lines trip (guard against double-push)
        pend.clear();
        for (int j : affected)
            if (!tripped[j] && load[j] > cap[j]){ tripped[j] = 1; pend.push_back(j); }
    }
    ll lost = 0;
    for (int i = 0; i < L; i++) if (tripped[i]) lost += Vv[i];
    return totalValue - lost;
}

ll worstServed(const vector<ll>& cap){
    ll best = LLONG_MAX;
    for (int t : scen) best = min(best, runScenario(cap, t));
    return best;
}

int main(int argc, char* argv[]){
    registerTestlibCmd(argc, argv);

    L = inf.readInt();
    M = inf.readInt();
    S = inf.readInt();
    B = inf.readLong();
    W.assign(L, 0); Vv.assign(L, 0); out.assign(L, {});
    totalValue = 0;
    for (int i = 0; i < L; i++){
        W[i]  = inf.readLong();
        Vv[i] = inf.readLong();
        totalValue += Vv[i];
    }
    for (int e = 0; e < M; e++){
        int u = inf.readInt();
        int j = inf.readInt();
        int num = inf.readInt();
        out[u].push_back({j, num});
    }
    scen.assign(S, 0);
    for (int i = 0; i < S; i++) scen[i] = inf.readInt();

    // ---- read participant capacities ----
    vector<ll> cap(L, 0);
    ll sum = 0;
    for (int i = 0; i < L; i++){
        ll c = ouf.readLong(0, B, "cap");   // rejects negative / >B / nan / garbage
        cap[i] = c;
        sum += c;
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output tokens");
    if (sum > B) quitf(_wa, "capacity sum %lld exceeds budget %lld", sum, B);

    // ---- objective and baseline ----
    ll F = worstServed(cap);
    ll U = B / (ll)L;
    vector<ll> unif(L, U);
    ll Bsl = worstServed(unif);
    if (Bsl <= 0) Bsl = 1;

    double sc = min(1000.0, 100.0 * (double)F / (double)max((ll)1, Bsl));
    quitp(sc / 1000.0, "OK F=%lld Bsl=%lld Ratio: %.6f", F, Bsl, sc / 1000.0);
    return 0;
}
