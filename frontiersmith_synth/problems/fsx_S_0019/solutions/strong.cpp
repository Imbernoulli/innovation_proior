// TIER: strong
// Degree-constrained Kruskal over the artifact chambers: repeatedly add the globally
// shortest chamber-chamber trench whose endpoints still have free collars and that merges
// two components (union-find). Any components left unmerged (blocked by caps) are stitched
// together through free-collar leaves. The resulting degree-capped spanning tree is compared
// against a nearest-neighbour path and the shorter total-length network is emitted. Then a
// bounded edge-swap local search (remove a tree edge, reconnect the two halves with the
// shortest valid crossing trench) squeezes out further length on smaller instances.
#include <bits/stdc++.h>
using namespace std;

int N, T;
vector<long long> X, Y;
vector<int> cap;
static inline long long dist2(int a, int b) {
    long long dx = X[a] - X[b], dy = Y[a] - Y[b];
    return dx * dx + dy * dy;
}
static inline long long len(int a, int b) {
    return (long long)llround(sqrt((double)dist2(a, b)));
}

struct DSU {
    vector<int> p;
    void init(int n) { p.resize(n); for (int i = 0; i < n; i++) p[i] = i; }
    int find(int x) { while (p[x] != x) { p[x] = p[p[x]]; x = p[x]; } return x; }
    void unite(int a, int b) { p[find(a)] = find(b); }
};

int main() {
    if (scanf("%d %d", &N, &T) != 2) return 0;
    X.resize(N); Y.resize(N); cap.assign(N, 0);
    for (int i = 0; i < N; i++) scanf("%lld %lld %d", &X[i], &Y[i], &cap[i]);

    // ---- Kruskal (degree-constrained) over chambers 0..T-1 ----
    vector<array<long long,3>> pairs; // dist2, i, j
    pairs.reserve((size_t)T * (T - 1) / 2);
    for (int i = 0; i < T; i++)
        for (int j = i + 1; j < T; j++)
            pairs.push_back({dist2(i, j), (long long)i, (long long)j});
    sort(pairs.begin(), pairs.end(),
         [](const array<long long,3>& a, const array<long long,3>& b){ return a[0] < b[0]; });

    DSU dsu; dsu.init(T);
    vector<int> deg(T, 0);
    set<pair<int,int>> seen;
    vector<pair<int,int>> tedges; // 0-indexed chamber edges
    int comps = T;
    for (auto& pr : pairs) {
        if (comps == 1) break;
        int i = (int)pr[1], j = (int)pr[2];
        if (dsu.find(i) == dsu.find(j)) continue;
        if (deg[i] >= cap[i] || deg[j] >= cap[j]) continue;
        dsu.unite(i, j); deg[i]++; deg[j]++;
        seen.insert({min(i,j), max(i,j)});
        tedges.push_back({i, j});
        comps--;
    }
    // repair leftover components via free-collar nodes
    if (comps > 1) {
        // representative root of the "main" component: root of chamber 0
        for (int c = 1; c < T; c++) {
            if (dsu.find(c) == dsu.find(0)) continue;
            // find a free-collar node in main
            int a = -1, root0 = dsu.find(0);
            for (int u = 0; u < T; u++)
                if (dsu.find(u) == root0 && deg[u] < cap[u]) { a = u; break; }
            // find a free-collar node in c's component
            int b = -1, rootc = dsu.find(c);
            for (int u = 0; u < T; u++)
                if (dsu.find(u) == rootc && deg[u] < cap[u]) { b = u; break; }
            if (a < 0 || b < 0) continue; // should not happen (trees have free leaves)
            dsu.unite(a, b); deg[a]++; deg[b]++;
            seen.insert({min(a,b), max(a,b)});
            tedges.push_back({a, b});
        }
    }

    long long treeLen = 0;
    for (auto& e : tedges) treeLen += len(e.first, e.second);

    // ---- nearest-neighbour path (fallback / comparison) ----
    vector<pair<int,int>> pathEdges;
    {
        vector<char> vis(T, 0);
        int cur = 0; vis[0] = 1;
        for (int step = 1; step < T; step++) {
            int best = -1; long long bd = LLONG_MAX;
            for (int j = 0; j < T; j++) {
                if (vis[j]) continue;
                long long d = dist2(cur, j);
                if (d < bd) { bd = d; best = j; }
            }
            vis[best] = 1;
            pathEdges.push_back({cur, best});
            cur = best;
        }
    }
    long long pathLen = 0;
    for (auto& e : pathEdges) pathLen += len(e.first, e.second);

    vector<pair<int,int>>* chosen = (treeLen <= pathLen) ? &tedges : &pathEdges;
    bool usingTree = (treeLen <= pathLen);

    // ---- bounded edge-swap local search on the chosen tree (small instances) ----
    if (T <= 260) {
        // rebuild deg + seen for the chosen network
        vector<pair<int,int>> E = *chosen;
        vector<int> d2(T, 0);
        set<pair<int,int>> present;
        for (auto& e : E) { d2[e.first]++; d2[e.second]++; present.insert({min(e.first,e.second), max(e.first,e.second)}); }
        bool improved = true;
        int passes = 0;
        while (improved && passes < 4) {
            improved = false; passes++;
            for (int ei = 0; ei < (int)E.size(); ei++) {
                int a = E[ei].first, b = E[ei].second;
                long long old = len(a, b);
                // remove edge ei -> two components
                DSU t; t.init(T);
                for (int k = 0; k < (int)E.size(); k++) if (k != ei) t.unite(E[k].first, E[k].second);
                int ra = t.find(a);
                // find shortest crossing edge (u in comp(a), v not) with free collars
                // after removal a,b each have one freed collar
                vector<int> dd = d2; dd[a]--; dd[b]--;
                long long bestLen = old; int bu = -1, bv = -1;
                for (int u = 0; u < T; u++) {
                    if (dd[u] >= cap[u]) continue;
                    bool uin = (t.find(u) == ra);
                    for (int v = u + 1; v < T; v++) {
                        if (dd[v] >= cap[v]) continue;
                        bool vin = (t.find(v) == ra);
                        if (uin == vin) continue; // must cross the cut
                        if (present.count({min(u,v), max(u,v)})) continue;
                        long long L = len(u, v);
                        if (L < bestLen) { bestLen = L; bu = u; bv = v; }
                    }
                }
                if (bu >= 0 && (bu != a || bv != b) && (bu != b || bv != a)) {
                    // perform swap
                    present.erase({min(a,b), max(a,b)});
                    d2[a]--; d2[b]--;
                    E[ei] = {bu, bv};
                    present.insert({min(bu,bv), max(bu,bv)});
                    d2[bu]++; d2[bv]++;
                    improved = true;
                }
            }
        }
        *chosen = E;
    }

    printf("%d\n", (int)chosen->size());
    for (auto& e : *chosen) printf("%d %d\n", e.first + 1, e.second + 1);
    return 0;
}
