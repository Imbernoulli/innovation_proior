#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// -----------------------------------------------------------------------------
// Checker / scorer for "Compressed-Air City Grid: Booster Siting Under Quadratic
// Line Loss". family: superlinear-flow-splitting.
//
// Input:  V E S T K ; S lines (source node, amt) ; T lines (sink node, amt) ;
//         E lines  u v r cand gain   (pipe e, 1-indexed by order).
// Output: M b_1..b_M f_1..f_E   (boosted pipe indices, then a signed flow per
//         pipe; positive f_e means direction u_e -> v_e).
//
// Feasibility: boosted indices distinct, in [1,E], cand=1; every f_e finite,
//   |f_e|<=1e6; net flow at every source/sink/junction matches its requirement
//   (source: net outflow = amt; sink: net inflow = amt; else: 0), tol 1e-3.
//
// Objective (MIN): F = sum_e r'_e * f_e^2, r'_e = r_e-gain_e if e boosted else r_e.
//
// Baseline B (checker-computed reference, NEVER boosts, NEVER splits): process
//   sources in ascending node id; repeatedly send the source's still-unmet
//   amount to the nearest (fewest-pipes) sink with unmet demand along ONE fixed
//   BFS shortest-hop path (deterministic tie-break: BFS explores each node's
//   neighbours in ascending neighbour-id order). This is exactly what the
//   "trivial" reference solution reproduces -> ratio 0.1.
// Score (min): sc = min(1000, 100*B/max(1e-9,F)); ratio = sc/1000.
// -----------------------------------------------------------------------------

static int V, E, S, T, K;
static vector<int> eu, ev;
static vector<ll> er, egain;
static vector<int> ecand;
static vector<vector<pair<int,int>>> adj; // node -> (neighbor, edgeIdx), sorted by neighbor id

// BFS shortest-hop path from s to t (fewest pipes), deterministic tie-break via
// sorted adjacency + FIFO BFS. Returns true and fills outPath if t is reachable;
// returns false (outPath untouched) if not -- callers MUST check the result
// instead of assuming reachability (the harness feeds adversarial/edge-case
// graphs, and a naive reconstruction loop would spin forever on an unreachable
// target since parentNode[] stays -1).
static bool bfsPath(int s, int t, vector<int>& outPath) {
    vector<int> parentEdge(V + 1, -1), parentNode(V + 1, -1);
    vector<char> vis(V + 1, 0);
    queue<int> q;
    vis[s] = 1; q.push(s);
    while (!q.empty()) {
        int u = q.front(); q.pop();
        if (u == t) break;
        for (auto &pr : adj[u]) {
            int v = pr.first, ei = pr.second;
            if (!vis[v]) { vis[v] = 1; parentNode[v] = u; parentEdge[v] = ei; q.push(v); }
        }
    }
    if (!vis[t]) return false;
    vector<int> path;
    int cur = t;
    while (cur != s) {
        int ei = parentEdge[cur];
        if (ei < 0) return false; // defensive: should be unreachable given vis[t]
        path.push_back(ei);
        cur = parentNode[cur];
    }
    reverse(path.begin(), path.end());
    outPath = path;
    return true;
}

// Reference construction: single unsplit shortest-hop path per augmentation, no
// boosters. Returns per-edge accumulated signed flow and its total dissipation.
static double referenceBaseline(vector<pair<int,ll>> srcs, vector<pair<int,ll>> sinks) {
    sort(srcs.begin(), srcs.end());
    vector<double> flow(E, 0.0);
    vector<ll> remSrc(srcs.size()), remSink(sinks.size());
    for (size_t i = 0; i < srcs.size(); i++) remSrc[i] = srcs[i].second;
    for (size_t j = 0; j < sinks.size(); j++) remSink[j] = sinks[j].second;

    int guard = 0;
    for (size_t i = 0; i < srcs.size(); i++) {
        while (remSrc[i] > 0) {
            if (++guard > 10000) break; // safety; sums always balance by construction
            int bestJ = -1, bestDist = INT_MAX;
            for (size_t j = 0; j < sinks.size(); j++) {
                if (remSink[j] <= 0) continue;
                vector<int> p;
                if (!bfsPath(srcs[i].first, sinks[j].first, p)) continue; // unreachable
                int dist = (int)p.size();
                if (dist < bestDist || (dist == bestDist && (bestJ == -1 || sinks[j].first < sinks[bestJ].first))) {
                    bestDist = dist; bestJ = (int)j;
                }
            }
            if (bestJ < 0) break;
            vector<int> path;
            if (!bfsPath(srcs[i].first, sinks[bestJ].first, path)) break; // unreachable (shouldn't happen)
            ll amt = min(remSrc[i], remSink[bestJ]);
            int cur = srcs[i].first;
            for (int ei : path) {
                if (eu[ei] == cur) { flow[ei] += (double)amt; cur = ev[ei]; }
                else                { flow[ei] -= (double)amt; cur = eu[ei]; }
            }
            remSrc[i] -= amt; remSink[bestJ] -= amt;
        }
    }
    double F = 0.0;
    for (int e = 0; e < E; e++) F += (double)er[e] * flow[e] * flow[e];
    return F;
}

int main(int argc, char* argv[]) {
    registerTestlibCmd(argc, argv);

    V = inf.readInt(); E = inf.readInt(); S = inf.readInt(); T = inf.readInt(); K = inf.readInt();
    vector<pair<int,ll>> srcs(S), sinks(T);
    for (int i = 0; i < S; i++) { srcs[i].first = inf.readInt(); srcs[i].second = inf.readLong(); }
    for (int i = 0; i < T; i++) { sinks[i].first = inf.readInt(); sinks[i].second = inf.readLong(); }

    eu.assign(E, 0); ev.assign(E, 0); er.assign(E, 0); ecand.assign(E, 0); egain.assign(E, 0);
    adj.assign(V + 1, {});
    for (int e = 0; e < E; e++) {
        eu[e] = inf.readInt(); ev[e] = inf.readInt(); er[e] = inf.readLong();
        ecand[e] = inf.readInt(); egain[e] = inf.readLong();
        adj[eu[e]].push_back({ev[e], e});
        adj[ev[e]].push_back({eu[e], e});
    }
    for (int v = 1; v <= V; v++)
        sort(adj[v].begin(), adj[v].end());

    // ---- role lookup ----
    vector<ll> srcAmt(V + 1, -1), sinkAmt(V + 1, -1);
    for (auto &p : srcs) srcAmt[p.first] = p.second;
    for (auto &p : sinks) sinkAmt[p.first] = p.second;

    // ---- internal baseline B ----
    double B = referenceBaseline(srcs, sinks);
    if (!(B > 0.0) || !isfinite(B)) B = 1e-9;

    // ---- read & validate participant output ----
    int M = ouf.readInt(0, K, "M");
    vector<char> boosted(E, 0);
    vector<char> usedBoost(E, 0);
    for (int i = 0; i < M; i++) {
        int b = ouf.readInt(1, E, "booster_edge") - 1;
        if (usedBoost[b]) quitf(_wa, "booster pipe %d listed more than once", b + 1);
        usedBoost[b] = 1;
        if (!ecand[b]) quitf(_wa, "pipe %d is not booster-ready (cand=0)", b + 1);
        boosted[b] = 1;
    }

    const double FMAX = 1e6;
    vector<double> f(E);
    vector<double> balance(V + 1, 0.0);
    for (int e = 0; e < E; e++) {
        double v = ouf.readDouble(-FMAX, FMAX, "flow");
        if (!isfinite(v)) quitf(_wa, "non-finite flow on pipe %d", e + 1);
        f[e] = v;
        balance[eu[e]] -= v;
        balance[ev[e]] += v;
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output after the flow list");

    const double EPS = 1e-3;
    for (int n = 1; n <= V; n++) {
        double need = 0.0;
        if (srcAmt[n] >= 0) need = -(double)srcAmt[n];
        else if (sinkAmt[n] >= 0) need = (double)sinkAmt[n];
        if (fabs(balance[n] - need) > EPS)
            quitf(_wa, "flow conservation violated at node %d: got %.6f expected %.6f",
                  n, balance[n], need);
    }

    double F = 0.0;
    for (int e = 0; e < E; e++) {
        ll rp = boosted[e] ? max(1LL, er[e] - egain[e]) : er[e];
        F += (double)rp * f[e] * f[e];
    }
    if (!isfinite(F)) quitf(_wa, "non-finite objective");

    double sc = min(1000.0, 100.0 * B / max(1e-9, F));
    quitp(sc / 1000.0, "OK F=%.6f B=%.6f M=%d Ratio: %.6f", F, B, M, sc / 1000.0);
    return 0;
}
