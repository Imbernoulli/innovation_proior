// TIER: greedy
// The obvious single-pass engineer's heuristic: solve the max-flow on the CALMEST
// scenario (fewest outages -- "normal operating conditions"), rank edges by the raw
// megawatts flowing through them in that one solve, and pour the budget into the
// biggest-flow edges first. This never looks at any other scenario, so it cannot see
// which cut recurs once contingencies (line outages) hit.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int R, C, M, K, N;
ll Budget;
vector<int> eu, ev;
vector<ll> ecap, ecost;

struct Scenario {
    vector<ll> demand;
    ll totalDemand;
    vector<char> outaged;
    int outCount;
};

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
    eu.resize(M); ev.resize(M); ecap.resize(M); ecost.resize(M);
    for (int e = 0; e < M; e++) scanf("%d %d %lld %lld", &eu[e], &ev[e], &ecap[e], &ecost[e]);
    vector<Scenario> scen(K);
    for (int s = 0; s < K; s++) {
        scen[s].demand.assign(N, 0);
        scen[s].outaged.assign(M, 0);
        int L; scanf("%d", &L);
        ll total = 0;
        for (int k = 0; k < L; k++) { int nodeId; ll d; scanf("%d %lld", &nodeId, &d); scen[s].demand[nodeId] += d; total += d; }
        scen[s].totalDemand = total;
        int O; scanf("%d", &O);
        scen[s].outCount = O;
        for (int k = 0; k < O; k++) { int idx; scanf("%d", &idx); scen[s].outaged[idx] = 1; }
    }

    // pick the scenario with fewest outages as "normal operating conditions"
    int nominal = 0;
    for (int s = 1; s < K; s++) if (scen[s].outCount < scen[nominal].outCount) nominal = s;

    int S = N, T = N + 1;
    Dinic D; D.init(N + 2);
    vector<pair<int,int>> arcLoc(M, {-1, -1});
    for (int e = 0; e < M; e++) {
        if (scen[nominal].outaged[e]) continue;
        arcLoc[e] = D.addUndir(eu[e], ev[e], ecap[e]);
    }
    for (int i = 0; i < R; i++) D.addDir(S, i * C + 0, (ll)4e18);
    for (int nodeId = 0; nodeId < N; nodeId++)
        if (scen[nominal].demand[nodeId] > 0) D.addDir(nodeId, T, scen[nominal].demand[nodeId]);
    D.maxflow(S, T);

    // raw flow used through each edge = original capacity - remaining residual cap
    vector<pair<ll,int>> byFlow; // (flow, edge index)
    for (int e = 0; e < M; e++) {
        if (arcLoc[e].first < 0) continue;
        ll resid = D.g[arcLoc[e].first][arcLoc[e].second].cap;
        ll used = ecap[e] - resid;
        if (used > 0) byFlow.push_back({used, e});
    }
    // Rank by raw megawatts carried (descending); among ties (common along a simple
    // feeder->crossing chain, which necessarily carry identical flow) prefer the
    // smaller-capacity edge -- the one actually pinched, a plain "how close to its
    // own rating is this line running" tie-break an engineer would also use.
    sort(byFlow.begin(), byFlow.end(), [](const pair<ll,int>&a, const pair<ll,int>&b){
        if (a.first != b.first) return a.first > b.first;
        if (ecap[a.second] != ecap[b.second]) return ecap[a.second] < ecap[b.second];
        return a.second < b.second;
    });

    // Pour the whole budget into the single busiest line, then whatever's left into
    // the next busiest, and so on -- the obvious single-pass "fix the biggest line"
    // recipe, blind to every scenario but the one calm solve it started from.
    ll remaining = Budget;
    vector<pair<int,ll>> plan;
    for (auto& pr : byFlow) {
        int e = pr.second;
        if (remaining <= 0) break;
        ll amt = remaining / ecost[e];
        if (amt > 0) { plan.push_back({e, amt}); remaining -= amt * ecost[e]; }
    }

    printf("%d\n", (int)plan.size());
    for (auto& pr : plan) printf("%d %lld\n", pr.first, pr.second);
    return 0;
}
