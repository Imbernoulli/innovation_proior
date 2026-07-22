#include "testlib.h"
#include "routing_lib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// -----------------------------------------------------------------------------
// Checker / scorer for "Tunnel Relay".
//
// Input:  N M K T ; M edges (u v dur) ; K packets (s t r d v).
// Output: K lines, each "-1" (drop) or "L t_1 e_1 ... t_L e_L" (a timed route).
//
// Feasibility per delivered packet: path continuity (e_1 starts at s, e_L
// ends at t, consecutive edges connect), t_1 >= r, t_{j+1} >= t_j + dur(e_j)
// (may wait, may not overlap), final arrival t_L + dur(e_L) <= d. GLOBALLY,
// no two delivered packets (in the whole output) may share the same (edge,
// departure-time) slot -- each slot is consumed by at most one packet.
//
// Objective (max): F = sum of v over delivered packets.
// Baseline B (checker-internal, always positive): simulate the naive
// "always take the durationwise-shortest path, process packets in input
// order, wait for the same edge to free up" construction -- this is exactly
// what solutions/trivial.cpp also computes, so F_trivial == B by
// construction (ratio 0.1 on every test).
// Score: sc = min(1000, 100*F/max(1,B)); ratio = sc/1000.
// -----------------------------------------------------------------------------

static const int LMAX = 20;

int main(int argc, char* argv[]){
    registerTestlibCmd(argc, argv);

    int N = inf.readInt();
    int M = inf.readInt();
    int K = inf.readInt();
    ll  T = inf.readLong();

    vector<Edge> edges(M);
    for (int i = 0; i < M; i++){
        int u = inf.readInt();
        int v = inf.readInt();
        ll dur = inf.readLong();
        edges[i] = {u, v, dur};
    }
    vector<int> S(K), Tt(K);
    vector<ll> R(K), D(K), V(K);
    for (int k = 0; k < K; k++){
        S[k]  = inf.readInt();
        Tt[k] = inf.readInt();
        R[k]  = inf.readLong();
        D[k]  = inf.readLong();
        V[k]  = inf.readLong();
    }

    vector<vector<int>> adj = buildAdj(N, edges);

    // ---- internal baseline B: naive shortest-path-first, input order ----
    ll B = 0;
    {
        unordered_set<ll> used;
        for (int k = 0; k < K; k++){
            vector<int> path = shortestPath(N, edges, adj, S[k], Tt[k]);
            if (path.empty()) continue;
            if (tryPath(path, edges, used, R[k], D[k], true)) B += V[k];
        }
    }
    ll Bsl = max((ll)1, B);

    // ---- read & validate participant output ----
    unordered_set<ll> usedOut;
    ll F = 0;
    for (int k = 0; k < K; k++){
        ll first = ouf.readLong((ll)-1, (ll)LMAX, "route_len_or_drop");
        if (first == -1) continue; // packet dropped, contributes 0
        int L = (int)first;
        if (L < 1) quitf(_wa, "packet %d: route length must be >=1 or -1", k);
        int curNode = S[k];
        ll lastArrival = -1;
        vector<pair<int,ll>> slots(L);
        ll t = R[k];
        for (int j = 0; j < L; j++){
            ll tj = ouf.readLong(0, T, "depart_time");
            int ej = ouf.readInt(0, M - 1, "edge_id");
            if (edges[ej].u != curNode)
                quitf(_wa, "packet %d hop %d: edge %d does not start at node %d", k, j, ej, curNode);
            if (tj < t)
                quitf(_wa, "packet %d hop %d: depart time %lld earlier than allowed %lld", k, j, tj, t);
            slots[j] = {ej, tj};
            curNode = edges[ej].v;
            t = tj + edges[ej].dur;
            lastArrival = t;
        }
        if (curNode != Tt[k])
            quitf(_wa, "packet %d: route ends at node %d, expected sink %d", k, curNode, Tt[k]);
        if (lastArrival > D[k])
            quitf(_wa, "packet %d: arrival %lld exceeds deadline %lld", k, lastArrival, D[k]);
        for (auto& pr : slots){
            ll key = slotKey(pr.first, pr.second);
            if (usedOut.count(key))
                quitf(_wa, "slot (edge=%d, t=%lld) used by more than one packet", pr.first, pr.second);
            usedOut.insert(key);
        }
        F += V[k];
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output tokens");

    double sc = min(1000.0, 100.0 * (double)F / (double)Bsl);
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}
