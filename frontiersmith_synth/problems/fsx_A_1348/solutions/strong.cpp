// TIER: strong
// Two genuine insights the greedy recursive-centroid heuristic never uses:
//   (1) edge-cut-vs-vertex-cut: scan every plot for a real hub bargain (clearing costs less
//       than the sum of its incident path costs), and accept a hub ONLY if actually
//       re-constructing the whole partition with it cleared beats leaving it alone -- this
//       bundles many path cuts into one cheap operation exactly where the payoff is real.
//   (2) joint reconsideration (exchange argument): after an initial per-fragment centroid
//       construction, repeatedly try to UNDO one existing cut and RE-CUT a different edge
//       inside the region that reunites, keeping the same K pieces but re-examining the
//       split jointly with everything already decided -- something a one-pass sequential
//       centroid recursion can never do, since it never revisits an earlier round's choice.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int n, K, LAMBDA;
vector<ll> w;
vector<int> p;
vector<int> eu, ev, ec;
vector<vector<pair<int,int>>> adj; // node -> (neighbor, edgeIdx)
ll totalS = 0;

struct DSU {
    vector<int> par;
    DSU(int n) : par(n + 1) { for (int i = 0; i <= n; i++) par[i] = i; }
    int find(int x) { while (par[x] != x) { par[x] = par[par[x]]; x = par[x]; } return x; }
    void unite(int a, int b) { a = find(a); b = find(b); if (a != b) par[a] = b; }
};

// exact integer objective, matching the checker formula bit-for-bit; -1 if piece count != K
ll computeF(const vector<char> &removed, const vector<char> &cutFlag) {
    DSU dsu(n);
    for (int i = 1; i <= n - 1; i++) {
        if (removed[eu[i]] || removed[ev[i]]) continue;
        if (cutFlag[i]) continue;
        dsu.unite(eu[i], ev[i]);
    }
    map<int,ll> compW;
    for (int v = 1; v <= n; v++) if (!removed[v]) compW[dsu.find(v)] += w[v];
    if ((int)compW.size() != K) return -1;
    ll P = 0;
    for (auto &kv : compW) { ll dev = (ll)K * kv.second - totalS; P += dev * dev; }
    ll penalty = (LAMBDA * P) / ((ll)K * totalS);
    ll cost = 0;
    for (int v = 1; v <= n; v++) if (removed[v]) cost += p[v];
    for (int i = 1; i <= n - 1; i++) if (cutFlag[i]) cost += ec[i];
    return cost + penalty;
}

// Per-fragment (or whole-tree) recursive weighted-centroid splitting: starting from the
// components already induced by `removed`, repeatedly cut the edge that most evenly
// bisects the current largest surviving component by yield, until exactly `targetK`
// components remain. Skips removed vertices entirely.
vector<char> centroidConstruct(const vector<char> &removed, int targetK) {
    vector<char> cutFlag(n, 0);
    vector<int> comp(n + 1, -1);
    DSU dsu(n);
    for (int i = 1; i <= n - 1; i++) {
        if (removed[eu[i]] || removed[ev[i]]) continue;
        dsu.unite(eu[i], ev[i]);
    }
    map<int,int> rootId;
    int nextId = 0;
    for (int v = 1; v <= n; v++) {
        if (removed[v]) continue;
        int r = dsu.find(v);
        auto it = rootId.find(r);
        if (it == rootId.end()) { rootId[r] = nextId; comp[v] = nextId; nextId++; }
        else comp[v] = it->second;
    }
    int cur = nextId;

    while (cur < targetK) {
        unordered_map<int, ll> compW;
        for (int v = 1; v <= n; v++) if (comp[v] != -1) compW[comp[v]] += w[v];
        vector<int> ids;
        for (auto &kv : compW) ids.push_back(kv.first);
        sort(ids.begin(), ids.end());
        int big = ids[0]; ll bestW = -1;
        for (int id : ids) if (compW[id] > bestW) { bestW = compW[id]; big = id; }

        int start = -1;
        for (int v = 1; v <= n; v++) if (comp[v] == big) { start = v; break; }

        vector<int> parent(n + 1, 0), parentEdge(n + 1, 0), order;
        vector<char> vis(n + 1, 0);
        vector<int> stk; stk.push_back(start); vis[start] = 1;
        while (!stk.empty()) {
            int u = stk.back(); stk.pop_back();
            order.push_back(u);
            for (auto &pr : adj[u]) {
                int to = pr.first, eidx = pr.second;
                if (comp[to] != big || vis[to]) continue;
                vis[to] = 1; parent[to] = u; parentEdge[to] = eidx;
                stk.push_back(to);
            }
        }
        vector<ll> subW(n + 1, 0);
        for (int v = 1; v <= n; v++) if (comp[v] == big) subW[v] = w[v];
        for (int i = (int)order.size() - 1; i >= 0; i--) {
            int v = order[i];
            if (v != start) subW[parent[v]] += subW[v];
        }

        int bestEdgeIdx = -1, bestChild = -1;
        ll bestDiff = -1, bestCost = -1;
        for (int v : order) {
            if (v == start) continue;
            ll diff = llabs(2 * subW[v] - bestW);
            int eidx = parentEdge[v];
            if (bestEdgeIdx == -1 || diff < bestDiff ||
                (diff == bestDiff && (ec[eidx] < bestCost ||
                 (ec[eidx] == bestCost && eidx < bestEdgeIdx)))) {
                bestDiff = diff; bestEdgeIdx = eidx; bestChild = v; bestCost = ec[eidx];
            }
        }

        vector<char> vis2(n + 1, 0);
        vector<int> stk2; stk2.push_back(bestChild); vis2[bestChild] = 1;
        comp[bestChild] = cur;
        while (!stk2.empty()) {
            int u = stk2.back(); stk2.pop_back();
            for (auto &pr : adj[u]) {
                int to = pr.first, eidx = pr.second;
                if (eidx == bestEdgeIdx) continue;
                if (comp[to] != big || vis2[to]) continue;
                vis2[to] = 1; comp[to] = cur;
                stk2.push_back(to);
            }
        }
        cutFlag[bestEdgeIdx] = 1;
        cur++;
    }
    return cutFlag;
}

int main() {
    scanf("%d %d %d", &n, &K, &LAMBDA);
    w.assign(n + 1, 0); p.assign(n + 1, 0);
    for (int v = 1; v <= n; v++) { ll t; scanf("%lld", &t); w[v] = t; }
    for (int v = 1; v <= n; v++) { int t; scanf("%d", &t); p[v] = t; }
    eu.assign(n, 0); ev.assign(n, 0); ec.assign(n, 0);
    adj.assign(n + 1, {});
    for (int i = 1; i <= n - 1; i++) {
        int a, b, c; scanf("%d %d %d", &a, &b, &c);
        eu[i] = a; ev[i] = b; ec[i] = c;
        adj[a].push_back({b, i});
        adj[b].push_back({a, i});
    }
    for (int v = 1; v <= n; v++) totalS += w[v];

    // ---------------- Phase A: greedy hub acceptance, VERIFIED by full re-construction ---
    vector<int> deg(n + 1, 0);
    vector<ll> sumEdgeCost(n + 1, 0);
    for (int i = 1; i <= n - 1; i++) {
        deg[eu[i]]++; deg[ev[i]]++;
        sumEdgeCost[eu[i]] += ec[i]; sumEdgeCost[ev[i]] += ec[i];
    }
    vector<int> cand;
    for (int v = 1; v <= n; v++)
        if (deg[v] >= 3 && (ll)p[v] < sumEdgeCost[v]) cand.push_back(v);
    sort(cand.begin(), cand.end(), [&](int a, int b) {
        return (sumEdgeCost[a] - p[a]) > (sumEdgeCost[b] - p[b]);
    });

    vector<char> removed(n + 1, 0);
    {
        auto cf0 = centroidConstruct(removed, K);
        ll cur = computeF(removed, cf0);
        for (int v : cand) {
            if (removed[v]) continue;
            bool adjRemoved = false;
            for (auto &pr : adj[v]) if (removed[pr.first]) { adjRemoved = true; break; }
            if (adjRemoved) continue;
            removed[v] = 1;
            DSU dsu(n);
            for (int i = 1; i <= n - 1; i++) {
                if (removed[eu[i]] || removed[ev[i]]) continue;
                dsu.unite(eu[i], ev[i]);
            }
            set<int> roots;
            for (int u = 1; u <= n; u++) if (!removed[u]) roots.insert(dsu.find(u));
            if ((int)roots.size() > K) { removed[v] = 0; continue; }
            auto cfTrial = centroidConstruct(removed, K);
            ll trial = computeF(removed, cfTrial);
            if (trial != -1 && trial < cur) cur = trial;
            else removed[v] = 0;
        }
    }

    // ---------------- Phase B: initial construction + local-search polish ----------------
    vector<char> cutFlag = centroidConstruct(removed, K);
    ll curF = computeF(removed, cutFlag);

    for (int round = 0; round < 8; round++) {
        bool improvedAny = false;
        vector<int> cutList;
        for (int i = 1; i <= n - 1; i++) if (cutFlag[i]) cutList.push_back(i);
        for (int e1 : cutList) {
            if (!cutFlag[e1]) continue;
            cutFlag[e1] = 0;
            // P = the component that reunites (BFS from eu[e1], within survivors, current cuts)
            vector<char> inP(n + 1, 0);
            vector<int> stk = {eu[e1]}; inP[eu[e1]] = 1;
            vector<int> Pnodes;
            while (!stk.empty()) {
                int u = stk.back(); stk.pop_back();
                Pnodes.push_back(u);
                for (auto &pr : adj[u]) {
                    int to = pr.first, eidx = pr.second;
                    if (removed[to] || inP[to] || cutFlag[eidx]) continue;
                    inP[to] = 1; stk.push_back(to);
                }
            }
            ll bestF = curF; int bestE2 = -1;
            for (int u : Pnodes) {
                for (auto &pr : adj[u]) {
                    int to = pr.first, eidx = pr.second;
                    if (eidx == e1) continue;
                    if (!inP[to]) continue; // must be internal to P (both endpoints in P)
                    if (cutFlag[eidx]) continue;
                    cutFlag[eidx] = 1;
                    ll trial = computeF(removed, cutFlag);
                    if (trial != -1 && trial < bestF) { bestF = trial; bestE2 = eidx; }
                    cutFlag[eidx] = 0;
                }
            }
            if (bestE2 != -1) {
                cutFlag[bestE2] = 1;
                curF = bestF;
                improvedAny = true;
            } else {
                cutFlag[e1] = 1; // revert: no improving swap found
            }
        }
        if (!improvedAny) break;
    }

    vector<int> removedList;
    for (int v = 1; v <= n; v++) if (removed[v]) removedList.push_back(v);
    vector<int> cutList;
    for (int i = 1; i <= n - 1; i++) if (cutFlag[i]) cutList.push_back(i);

    printf("%d\n", (int)removedList.size());
    for (int v : removedList) printf("%d ", v);
    printf("\n");
    printf("%d\n", (int)cutList.size());
    for (int e : cutList) printf("%d ", e);
    printf("\n");
    return 0;
}
