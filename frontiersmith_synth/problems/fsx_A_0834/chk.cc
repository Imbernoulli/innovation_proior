#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
static const ll INF = (ll)4e18;
static const ll PEN = 2000000;

// Checker / scorer for audit-cutset-inspector (canal patrols, maximize).
//
// Best-response mechanic (deterministic): every flow (s,t,vol,val) takes the
// shortest path from s to t under edge weight cost + PEN*[edge patrolled].
// Ties are broken canonically by ALWAYS preferring the smallest edge index
// among edges lying on a shortest path (so equal-cost parallel channels in a
// bundle are indifferent to smugglers -- any one unpatrolled channel is a
// free escape). The path is reconstructed by: Dijkstra on the REVERSED graph
// from t gives dist_to_t[]; walk forward from s always taking the smallest
// -index outgoing edge that lies on a shortest path (dist_to_t[u] == w +
// dist_to_t[v]). At the FIRST patrolled edge encountered, seize
// min(vol, cap_of_that_edge) units of cargo (value = seize*val); the rest of
// that shipment is not inspected further. A flow whose path never touches a
// patrolled edge contributes 0.
//
// Internal baseline B: the checker patrols ONLY edge index 0. Edge 0 is, by
// generator construction, the single edge of bundle 0 -- a genuine
// chokepoint (no parallel channel at all) -- so B is always well-defined and
// positive (every flow whose path must start by crossing hub 0->1 is forced
// through it regardless of the penalty, since there is no alternative).
//
// Score (maximization, standard convention): Ratio = min(1, F / (10*B)).
//   B itself (== patrolling just the harbor mouth)      -> Ratio 0.1
//   >= 10x the single-chokepoint baseline                -> Ratio capped 1.0

struct Edge { int u, v; ll cost, cap; };

static int n, m, K;
static ll Bbudget;
static vector<Edge> edges;
static vector<vector<pair<int,int>>> fwdAdj; // node -> (edgeIdx, v)
static vector<vector<pair<int,int>>> revAdj; // node -> (edgeIdx, u)   (reverse graph)
struct Flow { int s, t; ll vol, val; };
static vector<Flow> flows;

// Dijkstra on reverse graph from target t, edge weight = cost + PEN*patrolled.
static vector<ll> distToT(int t, const vector<char>& patrolled) {
    vector<ll> dist(n, INF);
    priority_queue<pair<ll,int>, vector<pair<ll,int>>, greater<>> pq;
    dist[t] = 0; pq.push({0, t});
    while (!pq.empty()) {
        auto [d, u] = pq.top(); pq.pop();
        if (d != dist[u]) continue;
        for (auto& [eidx, v] : revAdj[u]) { // reverse edge v->u came from original u2=v ...
            ll w = edges[eidx].cost + (patrolled[eidx] ? PEN : 0);
            if (dist[u] + w < dist[v]) { dist[v] = dist[u] + w; pq.push({dist[v], v}); }
        }
    }
    return dist;
}

// Simulate a patrol set; return total seized value F.
static ll simulate(const vector<char>& patrolled) {
    ll total = 0;
    // group flows by target to reuse Dijkstra runs
    map<int, vector<ll>> cache; // t -> dist vector, computed lazily
    unordered_map<int, vector<ll>> memo;
    for (auto& f : flows) {
        vector<ll>* distp;
        auto it = memo.find(f.t);
        if (it == memo.end()) {
            auto d = distToT(f.t, patrolled);
            it = memo.emplace(f.t, std::move(d)).first;
        }
        distp = &it->second;
        vector<ll>& dist = *distp;
        if (dist[f.s] >= INF) continue;
        int cur = f.s;
        int steps = 0;
        while (cur != f.t && steps <= n + 2) {
            steps++;
            int chosen = -1;
            for (auto& [eidx, v] : fwdAdj[cur]) {
                ll w = edges[eidx].cost + (patrolled[eidx] ? PEN : 0);
                if (dist[cur] == w + dist[v]) {
                    if (chosen == -1 || eidx < chosen) chosen = eidx;
                }
            }
            if (chosen == -1) break; // shouldn't happen
            if (patrolled[chosen]) {
                ll seize = min(f.vol, edges[chosen].cap);
                total += seize * f.val;
                break;
            }
            cur = edges[chosen].v;
        }
    }
    return total;
}

int main(int argc, char** argv) {
    registerTestlibCmd(argc, argv);

    n = inf.readInt();
    m = inf.readInt();
    K = inf.readInt();
    Bbudget = inf.readLong();

    edges.resize(m);
    fwdAdj.assign(n, {});
    revAdj.assign(n, {});
    for (int i = 0; i < m; i++) {
        int u = inf.readInt(0, n - 1, "u");
        int v = inf.readInt(0, n - 1, "v");
        ll cost = inf.readLong(1, (ll)1e9, "cost");
        ll cap = inf.readLong(1, (ll)1e9, "cap");
        edges[i] = {u, v, cost, cap};
        fwdAdj[u].push_back({i, v});
        revAdj[v].push_back({i, u});
    }
    flows.resize(K);
    for (int i = 0; i < K; i++) {
        int s = inf.readInt(0, n - 1, "s");
        int t = inf.readInt(0, n - 1, "t");
        ll vol = inf.readLong(1, (ll)1e9, "vol");
        ll val = inf.readLong(1, (ll)1e9, "val");
        flows[i] = {s, t, vol, val};
    }

    if (m < 1) quitf(_fail, "bad instance: no edges");

    // ---- internal baseline B: patrol edge 0 only ----
    vector<char> baseSet(m, 0);
    baseSet[0] = 1;
    ll B = simulate(baseSet);
    if (B <= 0) quitf(_fail, "bad instance: baseline B=%lld <= 0", B);

    // ---- read participant output ----
    ll c = ouf.readLong(0, Bbudget, "patrol count");
    vector<char> patrolled(m, 0);
    set<int> seen;
    for (ll i = 0; i < c; i++) {
        int idx = ouf.readInt(0, m - 1, "edge index");
        if (seen.count(idx)) quitf(_wa, "duplicate edge index %d", idx);
        seen.insert(idx);
        patrolled[idx] = 1;
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output after patrol list");

    ll F = simulate(patrolled);

    double sc = min(1000.0, 100.0 * (double)F / (double)max((ll)1, B));
    double ratio = sc / 1000.0;
    quitp(ratio, "OK F=%lld B=%lld Ratio: %.6f", F, B, ratio);
    return 0;
}
