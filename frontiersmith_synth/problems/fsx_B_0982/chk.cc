#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// -----------------------------------------------------------------------------
// Checker/scorer for "The Oasis Cooperative's Fair Ditch Web".
//
// Input:  N M F D CAP L ; F field ids ; M lines "u v len ret" (candidate ditch
//         segments, ret = retention in parts-per-thousand).
// Output: K commitments "f X L p_0 ... p_{L-1}" (see statement.txt).
//
// A commitment (f, X, path) is walked edge by edge from the spring: the amount
// ENTERING the first edge is X; the amount entering each later edge is what
// SURVIVED every earlier edge on that same path (multiplicative retention).
// Every edge that appears in some commitment accumulates, across ALL commitments
// using it, the amount entering it there; that sum must not exceed CAP. The sum
// of every commitment's X must not exceed D. The total length of the distinct
// edges referenced by any commitment must not exceed L.
//
// Objective (MAX): F_obj = min over fields of (sum over that field's
//   commitments of what finally survives to it).
//
// Baseline B (checker-computed): give every field its fewest-HOPS route from the
// spring (BFS, deterministic tie-break by input edge order) and split
// min(D,CAP) FLATLY across all F fields (no compensation for the worse-off) --
// this is exactly what solutions/trivial.cpp does, so B reproduces trivial's own
// delivered minimum -> ratio 0.1 by construction.
// -----------------------------------------------------------------------------

static const int MAXN = 260;
static const int MAXM = 520;
static const int MAXK = 600;
static const int MAXPATH = 260;

int main(int argc, char* argv[]){
    registerTestlibCmd(argc, argv);

    int N = inf.readInt();
    int M = inf.readInt();
    int F = inf.readInt();
    ll D = inf.readLong();
    ll CAP = inf.readLong();
    ll L = inf.readLong();

    vector<int> fieldId(F);
    vector<char> isField(N, 0);
    for (int i = 0; i < F; i++){ fieldId[i] = inf.readInt(0, N - 1); isField[fieldId[i]] = 1; }

    vector<int> eu(M), ev(M); vector<ll> elen(M); vector<int> eret(M);
    map<pair<int,int>, int> edgeId;
    vector<vector<pair<int,int>>> adj(N); // (neighbor, edgeIdx), input order
    for (int i = 0; i < M; i++){
        eu[i] = inf.readInt(0, N - 1);
        ev[i] = inf.readInt(0, N - 1);
        elen[i] = inf.readLong(1, (ll)1e9);
        eret[i] = inf.readInt(1, 999);
        int a = min(eu[i], ev[i]), b = max(eu[i], ev[i]);
        edgeId[{a, b}] = i;
        adj[eu[i]].push_back({ev[i], i});
        adj[ev[i]].push_back({eu[i], i});
    }

    // ---- internal baseline B: BFS fewest-hops route per field, flat split ----
    vector<int> parentEdge(N, -1), dist(N, -1);
    {
        queue<int> q; q.push(0); dist[0] = 0;
        while (!q.empty()){
            int u = q.front(); q.pop();
            for (auto &pr : adj[u]){
                int v = pr.first, eid = pr.second;
                if (dist[v] == -1){ dist[v] = dist[u] + 1; parentEdge[v] = eid; q.push(v); }
            }
        }
    }
    double flatX = (double)min(D, CAP) / (double)max(1, F);
    double B = 1e18;
    for (int i = 0; i < F; i++){
        int v = fieldId[i];
        double retProd = 1.0;
        while (v != 0){
            int eid = parentEdge[v];
            retProd *= eret[eid] / 1000.0;
            v = (eu[eid] == v) ? ev[eid] : eu[eid];
        }
        double deliv = flatX * retProd;
        B = min(B, deliv);
    }
    if (!(B > 0)) B = 1e-9;

    // ---- replay participant's commitments ----
    int K = ouf.readInt(0, MAXK, "K");
    vector<double> fieldInflow(N, 0.0);
    vector<double> edgeFocus(M, 0.0);
    vector<char> edgeUsed(M, 0);
    ll totalLen = 0;
    double springTotal = 0.0;
    vector<int> stamp(N, -1);

    for (int c = 0; c < K; c++){
        int f = ouf.readInt(0, N - 1, "field_id");
        if (!isField[f]) quitf(_wa, "commitment %d: node %d is not a field", c, f);
        double X = ouf.readDouble(0.0, 1e15, "flow");
        if (!isfinite(X)) quitf(_wa, "commitment %d: non-finite flow", c);
        int Lp = ouf.readInt(2, MAXPATH, "path_len");
        vector<int> path(Lp);
        for (int i = 0; i < Lp; i++) path[i] = ouf.readInt(0, N - 1, "path_node");
        if (path[0] != 0) quitf(_wa, "commitment %d: path does not start at the spring", c);
        if (path[Lp - 1] != f) quitf(_wa, "commitment %d: path does not end at field %d", c, f);
        for (int i = 0; i < Lp; i++){
            if (stamp[path[i]] == c) quitf(_wa, "commitment %d: repeated node %d in path", c, path[i]);
            stamp[path[i]] = c;
        }
        double cur = X;
        springTotal += X;
        for (int i = 0; i + 1 < Lp; i++){
            int a = min(path[i], path[i + 1]), b = max(path[i], path[i + 1]);
            auto it = edgeId.find({a, b});
            if (it == edgeId.end()) quitf(_wa, "commitment %d: no ditch segment between %d and %d", c, path[i], path[i + 1]);
            int eid = it->second;
            edgeFocus[eid] += cur;
            if (!edgeUsed[eid]){ edgeUsed[eid] = 1; totalLen += elen[eid]; }
            cur *= eret[eid] / 1000.0;
        }
        fieldInflow[f] += cur;
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output tokens");

    if (springTotal > (double)D + 1e-6)
        quitf(_wa, "spring discharge exceeded: used %.6f > D=%lld", springTotal, D);
    for (int i = 0; i < M; i++)
        if (edgeFocus[i] > (double)CAP + 1e-6)
            quitf(_wa, "segment %d capacity exceeded: used %.6f > CAP=%lld", i, edgeFocus[i], CAP);
    if (totalLen > L)
        quitf(_wa, "ditch length budget exceeded: used %lld > L=%lld", totalLen, L);

    double Fobj = 1e18;
    for (int i = 0; i < F; i++) Fobj = min(Fobj, fieldInflow[fieldId[i]]);
    if (Fobj < 0) Fobj = 0;

    double sc = min(1000.0, 100.0 * Fobj / max(1e-9, B));
    quitp(sc / 1000.0, "OK F=%.6f B=%.6f Ratio: %.6f", Fobj, B, sc / 1000.0);
    return 0;
}
