// routing_lib.h -- shared, deterministic routing/scheduling primitives for
// "Tunnel Relay" (family: time-expanded-contention-routing).
// Included by chk.cc (to compute the internal baseline B) and by every
// solutions/*.cpp reference (NOT by gen.cpp, which only emits text).
// No randomness, no I/O here -- pure deterministic algorithms so that
// chk.cc's internal baseline and solutions/trivial.cpp are BIT-IDENTICAL
// when they run the same procedure.
#pragma once
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

struct Edge { int u, v; ll dur; };

static inline vector<vector<int>> buildAdj(int N, const vector<Edge>& edges) {
    vector<vector<int>> adj(N);
    for (int i = 0; i < (int)edges.size(); i++) adj[edges[i].u].push_back(i);
    return adj;
}

// Deterministic Dijkstra by edge duration (or a weight override, used to
// heavily penalize "contended" edges so the search prefers to avoid them
// when an alternative exists). Ties are broken deterministically: edges are
// relaxed in increasing edge-id order (adjacency built that way) and only a
// STRICT improvement updates a node, so the same graph always yields the
// same path. Returns the path as a list of edge indices (empty if unreachable).
static inline vector<int> shortestPath(int N, const vector<Edge>& edges,
                                        const vector<vector<int>>& adj,
                                        int s, int t,
                                        const vector<ll>* weightOverride = nullptr) {
    vector<ll> dist(N, LLONG_MAX);
    vector<int> parentEdge(N, -1);
    priority_queue<pair<ll,int>, vector<pair<ll,int>>, greater<pair<ll,int>>> pq;
    dist[s] = 0; pq.push({0, s});
    while (!pq.empty()) {
        auto pr = pq.top(); pq.pop();
        ll d = pr.first; int u = pr.second;
        if (d > dist[u]) continue;
        if (u == t) break;
        for (int eid : adj[u]) {
            ll w = weightOverride ? (*weightOverride)[eid] : edges[eid].dur;
            int v = edges[eid].v;
            ll nd = d + w;
            if (nd < dist[v]) { dist[v] = nd; parentEdge[v] = eid; pq.push({nd, v}); }
        }
    }
    if (dist[t] == LLONG_MAX) return {};
    vector<int> path;
    int cur = t;
    while (cur != s) {
        int eid = parentEdge[cur];
        path.push_back(eid);
        cur = edges[eid].u;
    }
    reverse(path.begin(), path.end());
    return path;
}

static inline ll realDuration(const vector<int>& path, const vector<Edge>& edges) {
    ll tot = 0;
    for (int e : path) tot += edges[e].dur;
    return tot;
}

// Key for a reserved (edge, departure-time) slot. Times are always small and
// non-negative in this problem (<= a few hundred), so this packing is safe.
static const ll SLOT_KEYMULT = 1000000LL;
static inline ll slotKey(int edgeId, ll t) { return (ll)edgeId * SLOT_KEYMULT + t; }

// Attempts to schedule a FIXED path (a sequence of edge indices) departing
// no earlier than `release`, taking at each hop the EARLIEST still-free
// (edge, time) slot (this is optimal for a fixed path: delaying a hop can
// only delay every later hop). Fails if the running departure time ever
// exceeds `deadline` (a hard bound: durations are >= 1, so once a
// departure time alone exceeds the deadline the packet cannot possibly
// still arrive in time) or if final arrival > deadline.
// If `commit` is true and the attempt succeeds, the chosen slots are
// inserted into `used` (irrevocably consumed).
static inline bool tryPath(const vector<int>& path, const vector<Edge>& edges,
                            unordered_set<ll>& used, ll release, ll deadline,
                            bool commit, vector<ll>* departOut = nullptr) {
    vector<ll> depart(path.size());
    ll t = release;
    for (size_t j = 0; j < path.size(); j++) {
        int eid = path[j];
        ll cand = t;
        while (true) {
            if (cand > deadline) return false;
            if (!used.count(slotKey(eid, cand))) break;
            cand++;
        }
        depart[j] = cand;
        t = cand + edges[eid].dur;
    }
    if (t > deadline) return false;
    if (commit) for (size_t j = 0; j < path.size(); j++) used.insert(slotKey(path[j], depart[j]));
    if (departOut) *departOut = depart;
    return true;
}
