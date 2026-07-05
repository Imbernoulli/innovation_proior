// TIER: strong
// Degree-aware Kruskal seed + edge-swap local search with randomized restarts.
// Each swap: cut an installed tree cable (splitting the fabric into two sides), then
// reconnect with the cheapest candidate cable that crosses the cut and respects port
// limits. Keep the cheapest feasible fabric found across restarts.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int n, m;
vector<int> eu, ev; vector<ll> ew; vector<int> cap;

struct DSU {
    vector<int> p;
    DSU(int n) : p(n + 1) { for (int i = 0; i <= n; i++) p[i] = i; }
    int f(int x) { while (p[x] != x) x = p[x] = p[p[x]]; return x; }
    bool u(int a, int b) { a = f(a); b = f(b); if (a == b) return false; p[a] = b; return true; }
};

mt19937 rng(20250701u);

// build the degree-aware Kruskal tree; returns edge-index list (a spanning tree) or
// falls back to daisy chain if it cannot span.
vector<int> kruskalSeed(const vector<int>& chainIdx) {
    vector<int> order(m);
    for (int i = 0; i < m; i++) order[i] = i + 1;
    stable_sort(order.begin(), order.end(), [&](int a, int b){ return ew[a] < ew[b]; });
    DSU dsu(n);
    vector<int> deg(n + 1, 0), chosen;
    int comps = n;
    for (int idx : order) {
        int u = eu[idx], v = ev[idx];
        if (deg[u] >= cap[u] || deg[v] >= cap[v]) continue;
        if (dsu.f(u) == dsu.f(v)) continue;
        dsu.u(u, v); deg[u]++; deg[v]++; chosen.push_back(idx); comps--;
        if (comps == 1) break;
    }
    if (comps != 1) {
        chosen.clear();
        for (int i = 1; i < n; i++) chosen.push_back(chainIdx[i]);
    }
    return chosen;
}

// degree-aware Prim growth from station 1 (one-pass seed), daisy fallback.
vector<int> primSeed(const vector<int>& chainIdx) {
    vector<char> inTree(n + 1, 0);
    vector<int> deg(n + 1, 0), chosen;
    inTree[1] = 1; int cnt = 1; bool stuck = false;
    while (cnt < n) {
        ll bestW = LLONG_MAX; int bestE = -1;
        for (int e = 1; e <= m; e++) {
            int a = eu[e], b = ev[e];
            if (inTree[a] == inTree[b]) continue;
            if (deg[a] >= cap[a] || deg[b] >= cap[b]) continue;
            if (ew[e] < bestW) { bestW = ew[e]; bestE = e; }
        }
        if (bestE == -1) { stuck = true; break; }
        int a = eu[bestE], b = ev[bestE];
        deg[a]++; deg[b]++; inTree[a] = inTree[b] = 1; chosen.push_back(bestE); cnt++;
    }
    if (stuck || cnt != n) { chosen.clear(); for (int i = 1; i < n; i++) chosen.push_back(chainIdx[i]); }
    return chosen;
}

ll totalCost(const vector<int>& tree) { ll s = 0; for (int e : tree) s += ew[e]; return s; }

// local search: repeatedly try to improve by cutting one tree edge and reconnecting.
void localSearch(vector<int>& tree, bool explore) {
    bool improved = true;
    int guard = 0;
    while (improved && guard++ < 200) {
        improved = false;
        // current degrees
        vector<int> deg(n + 1, 0);
        for (int e : tree) { deg[eu[e]]++; deg[ev[e]]++; }
        vector<int> tord(tree.size());
        for (size_t i = 0; i < tree.size(); i++) tord[i] = (int)i;
        if (explore) shuffle(tord.begin(), tord.end(), rng);
        for (int ti : tord) {
            int cutIdx = tree[ti];
            int cu = eu[cutIdx], cv = ev[cutIdx];
            // BFS on tree excluding this edge to find the side containing cu
            vector<vector<int>> adj(n + 1);
            for (size_t j = 0; j < tree.size(); j++) {
                if ((int)j == ti) continue;
                int e = tree[j]; adj[eu[e]].push_back(ev[e]); adj[ev[e]].push_back(eu[e]);
            }
            vector<char> side(n + 1, 0); // 1 = same side as cu
            queue<int> q; q.push(cu); side[cu] = 1;
            while (!q.empty()) { int x = q.front(); q.pop();
                for (int y : adj[x]) if (!side[y]) { side[y] = 1; q.push(y); } }
            // freed degree after cut
            vector<int> dg = deg; dg[cu]--; dg[cv]--;
            // cheapest feasible cross-cut replacement
            ll bestW = LLONG_MAX; int bestE = -1;
            vector<int> feas;
            for (int e = 1; e <= m; e++) {
                int a = eu[e], b = ev[e];
                if (side[a] == side[b]) continue;            // not crossing the cut
                if (dg[a] >= cap[a] || dg[b] >= cap[b]) continue;
                if (ew[e] < bestW) { bestW = ew[e]; bestE = e; }
                if (explore) feas.push_back(e);
            }
            if (bestE == -1) continue; // must at least be able to put the old edge back
            int pick = bestE;
            if (explore && !feas.empty() && (rng() % 100) < 20)
                pick = feas[rng() % feas.size()];
            if (ew[pick] < ew[cutIdx]) {
                tree[ti] = pick;
                deg[cu]--; deg[cv]--; deg[eu[pick]]++; deg[ev[pick]]++;
                improved = true;
            }
        }
    }
}

int main() {
    if (scanf("%d %d", &n, &m) != 2) return 0;
    cap.assign(n + 1, 0);
    for (int v = 1; v <= n; v++) scanf("%d", &cap[v]);
    eu.assign(m + 1, 0); ev.assign(m + 1, 0); ew.assign(m + 1, 0);
    vector<ll> chainW(n + 1, LLONG_MAX); vector<int> chainIdx(n + 1, -1);
    for (int i = 1; i <= m; i++) {
        int u, v; ll w; scanf("%d %d %lld", &u, &v, &w);
        eu[i] = u; ev[i] = v; ew[i] = w;
        int lo = min(u, v), hi = max(u, v);
        if (hi == lo + 1 && w < chainW[lo]) { chainW[lo] = w; chainIdx[lo] = i; }
    }

    vector<int> kseed = kruskalSeed(chainIdx);
    vector<int> pseed = primSeed(chainIdx);

    vector<int> best = kseed;
    localSearch(best, false);
    ll bestCost = totalCost(best);

    // also improve the Prim seed so strong dominates the greedy heuristic
    {
        vector<int> cur = pseed;
        localSearch(cur, false);
        ll c = totalCost(cur);
        if (c < bestCost) { bestCost = c; best = cur; }
    }

    int restarts = 8;
    for (int r = 0; r < restarts; r++) {
        vector<int> cur = (r % 2 == 0) ? kseed : pseed;
        localSearch(cur, true);
        ll c = totalCost(cur);
        if (c < bestCost) { bestCost = c; best = cur; }
    }

    printf("%d\n", (int)best.size());
    for (int e : best) printf("%d\n", e);
    return 0;
}
