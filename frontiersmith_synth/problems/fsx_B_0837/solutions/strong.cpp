// TIER: strong
// Insight: since the objective is the MAXIMUM unserved demand across scenarios, a
// budget unit is worth the most on an edge that sits on the binding min-cut of MANY
// scenarios at once -- not on whatever looks busiest in any single scenario. Each
// round: solve all K scenarios' max-flows, read off each one's min-cut (the residual-
// graph source-reachable frontier -- free from the same max-flow solve, no extra
// work), and score every edge by the TOTAL unserved demand of the scenarios whose cut
// it belongs to. Spend on the best score/cost edge, repeat. This is exactly
// "intersect the per-scenario cuts" -- an edge that recurs across many scenarios'
// cuts accumulates benefit from all of them simultaneously, so fixing it collapses
// several scenarios' deficits in one purchase.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int R, C, M, K, N;
ll Budget;
vector<int> eu, ev;
vector<ll> ecapBase, ecost;

struct Scenario {
    vector<ll> demand;
    ll totalDemand;
    vector<char> outaged;
};
vector<Scenario> scen;

struct Dinic {
    struct E { int to; ll cap; int rev; };
    vector<vector<E>> g; vector<int> lvl, it; int n;
    void init(int n_) { n = n_; g.assign(n, {}); }
    pair<int,int> addUndir(int u, int v, ll c) {
        int idu = (int)g[u].size(), idv = (int)g[v].size();
        g[u].push_back({v, c, idv});
        g[v].push_back({u, c, idu});
        return {u, idu};
    }
    void addDir(int u, int v, ll c) {
        g[u].push_back({v, c, (int)g[v].size()});
        g[v].push_back({u, 0, (int)g[u].size() - 1});
    }
    bool bfs(int s, int t) {
        lvl.assign(n, -1); queue<int> q; lvl[s] = 0; q.push(s);
        while (!q.empty()) { int u = q.front(); q.pop();
            for (auto& e : g[u]) if (e.cap > 0 && lvl[e.to] < 0) { lvl[e.to] = lvl[u] + 1; q.push(e.to); } }
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

int main() {
    scanf("%d %d %d %d %lld", &R, &C, &M, &K, &Budget);
    N = R * C;
    eu.resize(M); ev.resize(M); ecapBase.resize(M); ecost.resize(M);
    for (int e = 0; e < M; e++) scanf("%d %d %lld %lld", &eu[e], &ev[e], &ecapBase[e], &ecost[e]);
    scen.resize(K);
    for (int s = 0; s < K; s++) {
        scen[s].demand.assign(N, 0);
        scen[s].outaged.assign(M, 0);
        int L; scanf("%d", &L);
        ll total = 0;
        for (int k = 0; k < L; k++) { int nodeId; ll d; scanf("%d %lld", &nodeId, &d); scen[s].demand[nodeId] += d; total += d; }
        scen[s].totalDemand = total;
        int O; scanf("%d", &O);
        for (int k = 0; k < O; k++) { int idx; scanf("%d", &idx); scen[s].outaged[idx] = 1; }
    }

    vector<ll> add(M, 0);
    ll remaining = Budget;
    ll minCost = *min_element(ecost.begin(), ecost.end());
    int S = N, T = N + 1;

    for (int iter = 0; iter < 400 && remaining >= minCost; iter++) {
        vector<ll> benefit(M, 0);
        ll worst = 0;
        for (int s = 0; s < K; s++) {
            Dinic D; D.init(N + 2);
            vector<pair<int,int>> arcLoc(M, {-1, -1});
            for (int e = 0; e < M; e++) {
                if (scen[s].outaged[e]) continue;
                ll cap = ecapBase[e] + add[e];
                if (cap > 0) arcLoc[e] = D.addUndir(eu[e], ev[e], cap);
            }
            for (int i = 0; i < R; i++) D.addDir(S, i * C + 0, (ll)4e18);
            for (int nodeId = 0; nodeId < N; nodeId++)
                if (scen[s].demand[nodeId] > 0) D.addDir(nodeId, T, scen[s].demand[nodeId]);
            ll served = D.maxflow(S, T);
            ll unserved = scen[s].totalDemand - served;
            worst = max(worst, unserved);
            if (unserved <= 0) continue;
            // D.lvl[] after maxflow() = final source-reachable set (BFS that found no
            // augmenting path); any edge crossing reachable/unreachable is on this
            // scenario's min-cut and gets credited with this scenario's deficit.
            for (int e = 0; e < M; e++) {
                if (arcLoc[e].first < 0) continue;
                bool ru = D.lvl[eu[e]] >= 0, rv = D.lvl[ev[e]] >= 0;
                if (ru != rv) benefit[e] += unserved;
            }
        }
        if (worst <= 0) break;
        int best = -1; double bestEff = -1.0;
        for (int e = 0; e < M; e++) {
            if (benefit[e] <= 0) continue;
            if (remaining / ecost[e] < 1) continue;
            double eff = (double)benefit[e] / (double)ecost[e];
            if (eff > bestEff) { bestEff = eff; best = e; }
        }
        if (best < 0) break; // no recurring bottleneck we can still afford
        ll affordable = remaining / ecost[best];
        ll amt = max((ll)1, affordable / 5);
        amt = min(amt, affordable);
        add[best] += amt;
        remaining -= amt * ecost[best];
    }

    vector<pair<int,ll>> plan;
    for (int e = 0; e < M; e++) if (add[e] > 0) plan.push_back({e, add[e]});
    printf("%d\n", (int)plan.size());
    for (auto& pr : plan) printf("%d %lld\n", pr.first, pr.second);
    return 0;
}
