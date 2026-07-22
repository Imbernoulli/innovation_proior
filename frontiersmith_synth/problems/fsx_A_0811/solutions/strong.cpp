// TIER: strong
// Insight: (1) contract every non-candidate (branch) pipe to recover the
// "hub" contraction tree whose only edges are the valve-eligible candidates;
// (2) for every double-burst pair, PRE-GLUE the two hubs it touches by
// forbidding every candidate edge on the tree path between them -- anticipating
// that separating them would union two segments into one loss; (3) binary-
// search the smallest achievable max-segment weight removing at most V of the
// remaining (non-forbidden) candidate edges, a demand-balanced tree partition
// (edge weights = surviving bridge demand, node weights = hub's own branch
// demand). This beats "protect the biggest single pipe" because it balances
// TOTAL segment demand and never severs a double-burst-linked pair for free.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

struct DSU {
    vector<int> p;
    DSU(int n) : p(n + 1) { iota(p.begin(), p.end(), 0); }
    int find(int x) { while (p[x] != x) { p[x] = p[p[x]]; x = p[x]; } return x; }
    void uni(int a, int b) { a = find(a); b = find(b); if (a != b) p[a] = b; }
};

int N, M, V, C, L;
vector<int> eu, ev, EUsrc, EVsrc, PI, PJ;
vector<ll> ed;
vector<char> isCand;

// super-tree
vector<vector<pair<int,int>>> adj; // adj[node] = list of (neighbor, edgeIdx), indexed by super-root ids
vector<ll> superDemand;
int V_budget;
int cutCountGlobal;
vector<int>* outCuts;
ll g_X; // current binary-search threshold

const ll FAILV = -1;

ll dfs(int node, int parent) {
    ll cur = superDemand[node];
    if (cur < 0) return FAILV; // shouldn't happen
    for (auto& pr : adj[node]) {
        int child = pr.first, eidx = pr.second;
        if (child == parent) continue;
        ll childPending = dfs(child, node);
        if (childPending == FAILV) return FAILV;
        ll merged = cur + childPending + ed[eidx];
        if (merged <= g_X) {
            cur = merged;
        } else {
            cutCountGlobal++;
            if (cutCountGlobal > V_budget) return FAILV;
            if (outCuts) outCuts->push_back(eidx);
        }
    }
    if (cur > g_X) return FAILV;
    return cur;
}

int main() {
    scanf("%d %d %d %d", &N, &M, &V, &C);
    isCand.assign(M + 1, 0);
    for (int i = 0; i < C; i++) { int c; scanf("%d", &c); isCand[c] = 1; }

    eu.assign(M + 1, 0); ev.assign(M + 1, 0); ed.assign(M + 1, 0);
    for (int k = 1; k <= M; k++) scanf("%d %d %lld", &eu[k], &ev[k], &ed[k]);

    scanf("%d", &L);
    PI.assign(L, 0); PJ.assign(L, 0);
    for (int t = 0; t < L; t++) scanf("%d %d", &PI[t], &PJ[t]);

    // ---- base blocks: union all NON-candidate edges ----
    DSU dsu1(N);
    for (int e = 1; e <= M; e++) if (!isCand[e]) dsu1.uni(eu[e], ev[e]);

    vector<ll> blockDemand(N + 1, 0);
    for (int e = 1; e <= M; e++) if (!isCand[e]) blockDemand[dsu1.find(eu[e])] += ed[e];

    // block-adjacency tree (nodes = base blocks, edges = candidate pipes)
    vector<vector<pair<int,int>>> blockAdj(N + 1);
    for (int e = 1; e <= M; e++) if (isCand[e]) {
        int a = dsu1.find(eu[e]), b = dsu1.find(ev[e]);
        blockAdj[a].push_back({b, e});
        blockAdj[b].push_back({a, e});
    }

    // ---- force-glue every double-burst pair's blocks along their tree path ----
    vector<char> forced(M + 1, 0);
    for (int t = 0; t < L; t++) {
        int a = dsu1.find(eu[PI[t]]), b = dsu1.find(eu[PJ[t]]);
        if (a == b) continue;
        // BFS from a to b in blockAdj, recover path edges
        vector<int> parentNode(N + 1, -1), parentEdge(N + 1, -1);
        vector<char> vis(N + 1, 0);
        queue<int> q; q.push(a); vis[a] = 1;
        while (!q.empty()) {
            int u = q.front(); q.pop();
            if (u == b) break;
            for (auto& pr : blockAdj[u]) {
                int v = pr.first, eidx = pr.second;
                if (!vis[v]) { vis[v] = 1; parentNode[v] = u; parentEdge[v] = eidx; q.push(v); }
            }
        }
        int cur = b;
        while (cur != a && parentEdge[cur] != -1) {
            forced[parentEdge[cur]] = 1;
            cur = parentNode[cur];
        }
    }

    // ---- DSU2: merge base blocks joined by forced candidate edges ----
    DSU dsu2(N);
    for (int e = 1; e <= M; e++) if (isCand[e] && forced[e]) {
        int a = dsu1.find(eu[e]), b = dsu1.find(ev[e]);
        dsu2.uni(a, b);
    }

    superDemand.assign(N + 1, -1);
    for (int v = 1; v <= N; v++) if (dsu1.find(v) == v) {
        int r = dsu2.find(v);
        if (superDemand[r] < 0) superDemand[r] = 0;
        superDemand[r] += blockDemand[v];
    }
    for (int e = 1; e <= M; e++) if (isCand[e] && forced[e]) {
        int r = dsu2.find(dsu1.find(eu[e]));
        superDemand[r] += ed[e];
    }

    // ---- super-tree: available (non-forced) candidate edges ----
    adj.assign(N + 1, {});
    ll totalDemand = 0;
    for (int e = 1; e <= M; e++) totalDemand += ed[e];
    for (int e = 1; e <= M; e++) if (isCand[e] && !forced[e]) {
        int a = dsu2.find(dsu1.find(eu[e])), b = dsu2.find(dsu1.find(ev[e]));
        if (a == b) continue; // internal, cannot help split further
        adj[a].push_back({b, e});
        adj[b].push_back({a, e});
    }

    int root = dsu2.find(dsu1.find(1));

    // ---- binary search smallest feasible max-segment threshold ----
    V_budget = V;
    ll lo = 1, hi = max((ll)1, totalDemand);
    ll best = hi;
    while (lo <= hi) {
        ll mid = lo + (hi - lo) / 2;
        g_X = mid;
        cutCountGlobal = 0;
        outCuts = nullptr;
        ll res = dfs(root, -1);
        if (res != FAILV) { best = mid; hi = mid - 1; }
        else lo = mid + 1;
    }

    // ---- reconstruct the cut set achieving `best` ----
    g_X = best;
    cutCountGlobal = 0;
    vector<int> cuts;
    outCuts = &cuts;
    dfs(root, -1);

    if ((int)cuts.size() > V) cuts.resize(V); // defensive clamp, should not trigger

    printf("%d\n", (int)cuts.size());
    for (size_t i = 0; i < cuts.size(); i++) printf("%d%c", cuts[i], i + 1 == cuts.size() ? '\n' : ' ');
    if (cuts.empty()) printf("\n");
    return 0;
}
