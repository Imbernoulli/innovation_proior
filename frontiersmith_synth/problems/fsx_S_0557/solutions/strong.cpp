// TIER: strong
// Insight: with a threshold r a lone seed never spreads -- contagion must
// NUCLEATE, i.e. a seeded over-threshold PAIR inside a dense pocket ignites the
// whole pocket (a seeded 2-core cascades), while a seed on a hub does nothing.
// So instead of ranking single nodes by degree, we rank adjacent PAIRS that sit
// in a triangle (a dense-core edge) by their true simulated marginal spread, and
// commit the best pair each round until the budget runs out. Hub-leaf edges are
// not in any triangle, so the search never wastes seeds on hubs.
//
// (This is intentionally a strong heuristic, not the optimum: it always spends a
//  full pair per pocket and ignores the chain synergy by which an already-ignited
//  pocket lets its neighbour ignite with a single extra seed -- so real headroom
//  remains above this reference.)
#include <bits/stdc++.h>
using namespace std;

int n, m, r, k;
vector<vector<int>> adj;
vector<char> act;
vector<int>  cnt;

// Add seeds u,v and percolate. If commit==false, restore all state afterwards.
// Returns the number of newly-activated nodes (marginal gain).
long long percolateAdd(int u, int v, bool commit) {
    vector<int> touchedAct;   // nodes we flipped to active
    vector<int> incNodes;     // nodes whose cnt we incremented
    vector<int> q;
    long long gain = 0;
    auto seed = [&](int x) {
        if (!act[x]) { act[x] = 1; touchedAct.push_back(x); q.push_back(x); gain++; }
    };
    seed(u); seed(v);
    for (size_t i = 0; i < q.size(); i++) {
        int x = q[i];
        for (int w : adj[x]) if (!act[w]) {
            cnt[w]++; incNodes.push_back(w);
            if (cnt[w] >= r) { act[w] = 1; touchedAct.push_back(w); q.push_back(w); gain++; }
        }
    }
    if (!commit) {
        for (int x : touchedAct) act[x] = 0;
        for (int x : incNodes)   cnt[x]--;
    }
    return gain;
}

int main() {
    if (scanf("%d %d %d %d", &n, &m, &r, &k) != 4) return 0;
    adj.assign(n + 1, {});
    vector<pair<int,int>> elist(m);
    for (int i = 0; i < m; i++) {
        int u, v; scanf("%d %d", &u, &v);
        elist[i] = {u, v};
        adj[u].push_back(v);
        adj[v].push_back(u);
    }
    act.assign(n + 1, 0);
    cnt.assign(n + 1, 0);

    // candidate edges = "dense-core" edges (endpoints share a common neighbour).
    vector<char> mark(n + 1, 0);
    vector<pair<int,int>> cand;
    cand.reserve(m);
    for (auto &e : elist) {
        int u = e.first, v = e.second;
        // ensure u is the lower-degree endpoint for a cheap common-neighbour test
        if (adj[u].size() > adj[v].size()) swap(u, v);
        if (adj[u].size() < 2) continue;      // a degree-1 leaf edge is never a core edge
        for (int w : adj[v]) mark[w] = 1;
        bool tri = false;
        for (int w : adj[u]) if (mark[w]) { tri = true; break; }
        for (int w : adj[v]) mark[w] = 0;
        if (tri) cand.push_back({e.first, e.second});
    }

    vector<int> chosen;
    int budget = k;
    while (budget >= 2) {
        long long best = 0;
        int bu = -1, bv = -1;
        for (auto &c : cand) {
            int u = c.first, v = c.second;
            if (u > v) swap(u, v);
            if (act[u] || act[v]) continue;
            long long g = percolateAdd(u, v, false);
            if (g > best || (g == best && g > 0 && (u < bu || (u == bu && v < bv)))) {
                best = g; bu = u; bv = v;
            }
        }
        if (best <= 0 || bu < 0) break;
        percolateAdd(bu, bv, true);   // commit
        chosen.push_back(bu);
        chosen.push_back(bv);
        budget -= 2;
    }

    // spend any leftover single seed on the highest-degree still-inactive node
    // (rarely helps under a threshold, but never hurts feasibility/score).
    if (budget >= 1) {
        int bestNode = -1;
        for (int v = 1; v <= n; v++) if (!act[v]) {
            if (bestNode == -1 || adj[v].size() > adj[bestNode].size()) bestNode = v;
        }
        if (bestNode != -1) { chosen.push_back(bestNode); budget--; }
    }

    printf("%d\n", (int)chosen.size());
    for (int v : chosen) printf("%d\n", v);
    return 0;
}
