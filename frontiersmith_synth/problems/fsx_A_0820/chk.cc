#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// -----------------------------------------------------------------------------
// Checker / scorer for "Delay-Line Cascade: Synchronized Toppling Network".
// family: domino-cascade-timing-logic   objective: MAX
//
// Node 1 pulses at tick 0. Only COMMITTED edges (the participant's chosen
// subset, cost-bounded by K) conduct. Edges always go low-id -> high-id, so
// processing nodes 1..N in order is a valid propagation order.
//   plain node v  : arrival[v] = min(arrival[u]+w) over committed (u,v,w,*)
//                   with arrival[u] defined; undefined if no committed inbound
//                   edge has an active tail.
//   AND-merge v   : arrival[v] = smallest tick t that is hit by >=2 DISTINCT
//                   committed inbound edges (arrival[u]+w == t); undefined if
//                   no tick is hit twice (a single/mistimed pulse is wasted).
// F = sum of value[v] over v with arrival[v] defined.
//
// Baseline B = a one-hop-only knapsack: value/cost-greedy fill of budget K
// using ONLY direct edges out of node 1 into positive-value nodes (ignores
// every AND-merge entirely). B is always > 0 by generator construction.
// Score sc = min(1000, 100*F/max(1,B)); ratio = sc/1000, so matching the
// baseline scores 0.1 and each real improvement scores higher (cap 1.0).
//
// Feasibility: 0<=c<=M; indices in [1,M]; distinct; total committed cost<=K;
// no trailing tokens.
// -----------------------------------------------------------------------------

int main(int argc, char* argv[]) {
    registerTestlibCmd(argc, argv);

    int N = inf.readInt();
    int M = inf.readInt();
    ll K = inf.readLong();

    vector<int> reqSync(N + 1, 0);
    vector<ll> val(N + 1, 0);
    for (int i = 2; i <= N; i++) {
        reqSync[i] = inf.readInt(0, 1, "req_sync");
        val[i] = inf.readLong((ll)0, (ll)2000000, "value");
    }

    vector<int> eu(M + 1), ev(M + 1);
    vector<ll> ew(M + 1), ec(M + 1);
    for (int i = 1; i <= M; i++) {
        eu[i] = inf.readInt(1, N, "u");
        ev[i] = inf.readInt(1, N, "v");
        ew[i] = inf.readLong(1, (ll)2e9, "w");
        ec[i] = inf.readLong(1, (ll)2e9, "cost");
        if (eu[i] >= ev[i]) quitf(_fail, "malformed test data: u>=v on edge %d", i);
    }

    // ---- baseline B: direct one-hop knapsack from node 1 ----
    vector<ll> bestCost(N + 1, -1);
    for (int i = 1; i <= M; i++) {
        if (eu[i] == 1 && val[ev[i]] > 0 && !reqSync[ev[i]]) {
            int v = ev[i];
            if (bestCost[v] < 0 || ec[i] < bestCost[v]) bestCost[v] = ec[i];
        }
    }
    struct Cand { int v; ll val, cost; };
    vector<Cand> cands;
    for (int v = 2; v <= N; v++) if (bestCost[v] >= 0) cands.push_back({v, val[v], bestCost[v]});
    sort(cands.begin(), cands.end(), [](const Cand& a, const Cand& b) {
        __int128 lhs = (__int128)a.val * b.cost;
        __int128 rhs = (__int128)b.val * a.cost;
        if (lhs != rhs) return lhs > rhs;
        if (a.cost != b.cost) return a.cost < b.cost;
        return a.v < b.v;
    });
    ll remaining = K, B = 0;
    for (auto& cd : cands) if (cd.cost <= remaining) { remaining -= cd.cost; B += cd.val; }
    if (B <= 0) B = 1;

    // ---- read participant output ----
    int c = ouf.readInt(0, M, "count");
    vector<int> chosen(c);
    vector<char> seen(M + 1, 0);
    ll totalCost = 0;
    for (int i = 0; i < c; i++) {
        int idx = ouf.readInt(1, M, "edge index");
        if (seen[idx]) quitf(_wa, "edge %d committed more than once", idx);
        seen[idx] = 1;
        chosen[i] = idx;
        totalCost += ec[idx];
        if (totalCost > K) quitf(_wa, "committed cost %lld exceeds budget %lld", totalCost, K);
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing tokens after the committed-edge list");

    // ---- simulate the cascade over committed edges only ----
    vector<vector<pair<int, ll>>> inbound(N + 1);
    for (int idx : chosen) inbound[ev[idx]].push_back({eu[idx], ew[idx]});

    vector<ll> arrival(N + 1, -1);
    arrival[1] = 0;
    for (int v = 2; v <= N; v++) {
        vector<ll> cand;
        cand.reserve(inbound[v].size());
        for (auto& pr : inbound[v]) {
            int u = pr.first; ll w = pr.second;
            if (arrival[u] >= 0) cand.push_back(arrival[u] + w);
        }
        if (cand.empty()) continue;
        if (!reqSync[v]) {
            arrival[v] = *min_element(cand.begin(), cand.end());
        } else {
            sort(cand.begin(), cand.end());
            ll best = -1;
            for (size_t i = 0; i + 1 < cand.size(); i++) {
                if (cand[i] == cand[i + 1]) { best = cand[i]; break; }
            }
            arrival[v] = best;
        }
    }

    ll F = 0;
    for (int v = 2; v <= N; v++) if (arrival[v] >= 0) F += val[v];

    double sc = min(1000.0, 100.0 * (double)F / (double)max((ll)1, B));
    if (sc < 0.0) sc = 0.0;
    quitp(sc / 1000.0, "OK F=%lld B=%lld c=%d Ratio: %.6f", F, B, c, sc / 1000.0);
    return 0;
}
