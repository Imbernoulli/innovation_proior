// Checker/scorer for "Which Lines to Thicken Before Winter" (grid-reinforcement-load-patterns).
//
// Feasibility: distinct edge indices, non-negative integer upgrade amounts, total
// upgrade cost <= Budget.
// Objective: for each of K scenarios (own demand pattern + own outaged-edge set),
// build the flow network with the participant's upgraded capacities (outaged edges
// forced to 0 regardless of upgrade), max-flow generator column -> demand nodes
// (capped at each node's scenario demand) = served demand. unserved_s = demand_s -
// served_s. F = MAX unserved_s over all scenarios (minimize the worst scenario).
// Baseline B = same MAX-unserved objective using NO upgrades at all (do-nothing).
// Score = min(1.0, 0.1 * B / F).
#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int R, C, M, K, N;
ll Budget;
vector<int> eu, ev;
vector<ll> ecapBase, ecost;

struct Scenario {
    vector<ll> demand;      // size N
    ll totalDemand;
    vector<char> outaged;   // size M
};
vector<Scenario> scen;

// ---------- Dinic max-flow over an explicit capacity array (one entry per edge, both
//            directions share it since power lines are undirected) ----------
struct Dinic {
    struct E { int to; ll cap; int rev; };
    vector<vector<E>> g; vector<int> lvl, it; int n;
    void init(int n_) { n = n_; g.assign(n, {}); }
    int addUndir(int u, int v, ll c) {
        int idu = (int)g[u].size(), idv = (int)g[v].size();
        g[u].push_back({v, c, idv});
        g[v].push_back({u, c, idu});
        return idu; // index into g[u] for this edge's forward arc
    }
    void addDir(int u, int v, ll c) {
        g[u].push_back({v, c, (int)g[v].size()});
        g[v].push_back({u, 0, (int)g[u].size() - 1});
    }
    bool bfs(int s, int t) {
        lvl.assign(n, -1); queue<int> q; lvl[s] = 0; q.push(s);
        while (!q.empty()) {
            int u = q.front(); q.pop();
            for (auto& e : g[u]) if (e.cap > 0 && lvl[e.to] < 0) { lvl[e.to] = lvl[u] + 1; q.push(e.to); }
        }
        return lvl[t] >= 0;
    }
    ll dfs(int u, int t, ll f) {
        if (u == t) return f;
        for (int& i = it[u]; i < (int)g[u].size(); i++) {
            E& e = g[u][i];
            if (e.cap > 0 && lvl[e.to] == lvl[u] + 1) {
                ll d = dfs(e.to, t, min(f, e.cap));
                if (d > 0) { e.cap -= d; g[e.to][e.rev].cap += d; return d; }
            }
        }
        return 0;
    }
    ll maxflow(int s, int t) {
        ll fl = 0;
        while (bfs(s, t)) { it.assign(n, 0); ll f; while ((f = dfs(s, t, (ll)4e18)) > 0) fl += f; }
        return fl;
    }
};

// compute F = max over scenarios of unserved demand, given an upgrade array (add[e] >= 0)
ll worstUnserved(const vector<ll>& add) {
    int S = N, T = N + 1;
    ll worst = 0;
    for (int s = 0; s < K; s++) {
        Dinic D; D.init(N + 2);
        for (int e = 0; e < M; e++) {
            if (scen[s].outaged[e]) continue;
            ll cap = ecapBase[e] + add[e];
            if (cap > 0) D.addUndir(eu[e], ev[e], cap);
        }
        for (int i = 0; i < R; i++) D.addDir(S, i * C + 0, (ll)4e18); // column-0 generators
        for (int nodeId = 0; nodeId < N; nodeId++)
            if (scen[s].demand[nodeId] > 0) D.addDir(nodeId, T, scen[s].demand[nodeId]);
        ll served = D.maxflow(S, T);
        ll unserved = scen[s].totalDemand - served;
        worst = max(worst, unserved);
    }
    return worst;
}

int main(int argc, char** argv) {
    registerTestlibCmd(argc, argv);
    R = inf.readInt(); C = inf.readInt(); M = inf.readInt(); K = inf.readInt(); Budget = inf.readLong();
    N = R * C;
    eu.resize(M); ev.resize(M); ecapBase.resize(M); ecost.resize(M);
    for (int e = 0; e < M; e++) {
        eu[e] = inf.readInt(); ev[e] = inf.readInt();
        ecapBase[e] = inf.readLong(); ecost[e] = inf.readLong();
    }
    scen.resize(K);
    for (int s = 0; s < K; s++) {
        scen[s].demand.assign(N, 0);
        scen[s].outaged.assign(M, 0);
        int L = inf.readInt();
        ll total = 0;
        for (int k = 0; k < L; k++) {
            int nodeId = inf.readInt(0, N - 1);
            ll d = inf.readLong(1, (ll)1e12);
            scen[s].demand[nodeId] += d;
            total += d;
        }
        scen[s].totalDemand = total;
        int O = inf.readInt();
        for (int k = 0; k < O; k++) {
            int idx = inf.readInt(0, M - 1);
            scen[s].outaged[idx] = 1;
        }
    }

    // ---- internal baseline: do-nothing (no upgrades) ----
    vector<ll> zeroAdd(M, 0);
    ll Bbase = worstUnserved(zeroAdd);
    if (Bbase <= 0) quitf(_fail, "bad instance: baseline worst-unserved B=%lld", Bbase);

    // ---- read participant upgrades ----
    int E = ouf.readInt(0, M, "E");
    vector<ll> add(M, 0);
    vector<char> seen(M, 0);
    ll spend = 0;
    for (int k = 0; k < E; k++) {
        int idx = ouf.readInt(0, M - 1, "edge index");
        ll amt = ouf.readLong(0, (ll)1000000000LL, "add amount");
        if (seen[idx]) quitf(_wa, "duplicate edge index %d in upgrade list", idx);
        seen[idx] = 1;
        add[idx] = amt;
        spend += amt * ecost[idx];
        if (spend > Budget) quitf(_wa, "upgrade budget exceeded: spend %lld > %lld", spend, Budget);
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output tokens");
    if (spend > Budget) quitf(_wa, "upgrade budget exceeded: spend %lld > %lld", spend, Budget);

    ll F = worstUnserved(add);
    double sc = min(1000.0, 100.0 * (double)Bbase / (double)max((ll)1, F));
    quitp(sc / 1000.0, "OK F=%lld B=%lld spend=%lld/%lld Ratio: %.6f", F, Bbase, spend, Budget, sc / 1000.0);
    return 0;
}
