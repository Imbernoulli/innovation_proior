#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// -----------------------------------------------------------------------------
// Checker / scorer for "Shelters Sited for the Worst Flood-Day Traffic".
//
// Input:  N M C S K ; pop[1..N] ; M edges (u v cap) ; C candidates (node cap) ;
//         K scenarios (b, then b blocked edge indices into the M list).
// Output: S distinct candidate indices in [0,C-1] -- the chosen shelter sites.
//
// For each scenario s: build the surviving road graph (base edges minus the
// scenario's blocked edges), run max-flow from a super-source (edge cap = pop_d
// into every district gateway) to a super-sink (edge cap = site intake capacity,
// only from SELECTED candidate nodes). F_s = that max-flow value (== the min
// surviving cut separating population from the chosen shelters, by max-flow /
// min-cut). Evacuation completion time T_s = ceil(P_total / max(F_s,1)).
//
// Objective (MIN): F = max over all K scenarios of T_s  (the worst flood day).
// Baseline B (checker-computed): pick the FIRST S candidates in the input's
//   presentation order -- a naive, order-only pick (the generator guarantees
//   it is never catastrophic: it is not an all-near-fragile pick). Score
//   (min): sc = min(1000, 100*B/max(1,F)).
// -----------------------------------------------------------------------------

struct Dinic {
    struct E { int to; ll cap; };
    vector<E> es;
    vector<vector<int>> g;
    vector<int> level, it;
    int n;
    Dinic(int n_) : n(n_), g(n_) {}
    void addEdge(int u, int v, ll cap) {
        g[u].push_back((int)es.size()); es.push_back({v, cap});
        g[v].push_back((int)es.size()); es.push_back({u, 0});
    }
    bool bfs(int s, int t) {
        level.assign(n, -1);
        queue<int> q; level[s] = 0; q.push(s);
        while (!q.empty()) {
            int u = q.front(); q.pop();
            for (int id : g[u]) {
                if (es[id].cap > 0 && level[es[id].to] < 0) {
                    level[es[id].to] = level[u] + 1;
                    q.push(es[id].to);
                }
            }
        }
        return level[t] >= 0;
    }
    ll dfs(int u, int t, ll f) {
        if (u == t) return f;
        for (int &i = it[u]; i < (int)g[u].size(); i++) {
            int id = g[u][i];
            int v = es[id].to;
            if (es[id].cap > 0 && level[v] == level[u] + 1) {
                ll d = dfs(v, t, min(f, es[id].cap));
                if (d > 0) {
                    es[id].cap -= d;
                    es[id ^ 1].cap += d;
                    return d;
                }
            }
        }
        return 0;
    }
    ll maxflow(int s, int t) {
        ll flow = 0;
        while (bfs(s, t)) {
            it.assign(n, 0);
            ll f;
            while ((f = dfs(s, t, LLONG_MAX)) > 0) flow += f;
        }
        return flow;
    }
};

int N, M, C, S, K;
vector<ll> pop_;
vector<int> eu, ev; vector<ll> ecap;
vector<int> candNode; vector<ll> candCap;
vector<vector<int>> scen;
ll P_total;

ll evalObjective(const vector<int>& sel) {
    // node ids: 1..N graph nodes, 0 = SRC, N+1 = SINK
    int SRC = 0, SINK = N + 1;
    ll worst = 0;
    for (int s = 0; s < K; s++) {
        vector<char> blocked(M, 0);
        for (int e : scen[s]) blocked[e] = 1;
        Dinic din(N + 2);
        for (int d = 1; d <= N; d++) if (pop_[d] > 0) din.addEdge(SRC, d, pop_[d]);
        for (int e = 0; e < M; e++) {
            if (blocked[e]) continue;
            din.addEdge(eu[e], ev[e], ecap[e]);
            din.addEdge(ev[e], eu[e], ecap[e]);
        }
        for (int idx : sel) din.addEdge(candNode[idx], SINK, candCap[idx]);
        ll Fs = din.maxflow(SRC, SINK);
        ll denom = max(Fs, 1LL);
        ll Ts = (P_total + denom - 1) / denom;
        worst = max(worst, Ts);
    }
    return worst;
}

int main(int argc, char* argv[]) {
    registerTestlibCmd(argc, argv);

    N = inf.readInt(); M = inf.readInt(); C = inf.readInt(); S = inf.readInt(); K = inf.readInt();
    pop_.assign(N + 1, 0);
    for (int i = 1; i <= N; i++) pop_[i] = inf.readLong();
    eu.resize(M); ev.resize(M); ecap.resize(M);
    for (int e = 0; e < M; e++) {
        eu[e] = inf.readInt(1, N, "u");
        ev[e] = inf.readInt(1, N, "v");
        ecap[e] = inf.readLong(1, (ll)1e9, "cap");
    }
    candNode.resize(C); candCap.resize(C);
    for (int i = 0; i < C; i++) {
        candNode[i] = inf.readInt(1, N, "candnode");
        candCap[i] = inf.readLong(1, (ll)1e9, "candcap");
    }
    scen.assign(K, {});
    for (int s = 0; s < K; s++) {
        int b = inf.readInt(0, M, "b");
        scen[s].resize(b);
        for (int j = 0; j < b; j++) scen[s][j] = inf.readInt(0, M - 1, "blocked_edge");
    }
    P_total = 0;
    for (int i = 1; i <= N; i++) P_total += pop_[i];
    if (P_total <= 0) P_total = 1;

    // ---- internal baseline B: first S candidates in presentation order ----
    vector<int> baseSel;
    for (int i = 0; i < S && i < C; i++) baseSel.push_back(i);
    ll B = evalObjective(baseSel);
    if (B <= 0) B = 1;

    // ---- read participant output: S distinct candidate indices ----
    vector<int> sel(S);
    vector<char> seen(C, 0);
    for (int i = 0; i < S; i++) {
        int idx = ouf.readInt(0, C - 1, "cand_index");
        if (seen[idx]) quitf(_wa, "duplicate candidate index %d", idx);
        seen[idx] = 1;
        sel[i] = idx;
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing tokens in output");

    ll F = evalObjective(sel);

    double sc = min(1000.0, 100.0 * (double)B / (double)max(1LL, F));
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}
