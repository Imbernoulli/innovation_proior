// TIER: strong
#include <bits/stdc++.h>
using namespace std;

// Insight: the coupling Laplacian's spectral gap between two weakly-linked
// regions is governed by how many *independent* edges cross the sparsest cut
// (Cheeger-style), not by how much total coupling money is spent overall or
// on high-degree nodes. So: (1) build a spanning tree of the candidate graph
// (Kruskal/union-find, any fixed order); (2) for every tree edge, count how
// many of the M candidate edges (tree or not) have their tree-path crossing
// it -- via the classic "+1/+1/-2 at the LCA, then subtree-sum" trick -- and
// normalize by the smaller side's population. A LOW ratio flags a genuine
// bottleneck cut (few parallel routes serving many nodes). Fund every edge
// that directly crosses a flagged cut first (a fixed, deliberately modest
// per-edge target -- leaving room for a sharper split to do even better),
// then spread the rest as evenly as possible (max-min water-filling) over
// everything else, subject to the per-edge cap and the per-node degree cap.

static const double DEGCAP = 1.0;
static const double BOOST_TARGET = 0.18;

int N, M;
vector<int> EU, EV;
vector<double> ECap;

int main() {
    scanf("%d", &N);
    scanf("%d", &M);
    double R, C; int T, W;
    scanf("%lf %lf %d %d", &R, &C, &T, &W);
    C -= 1e-6; // tiny safety margin so float accumulation never reports over-budget
    vector<double> x0(N + 1);
    for (int i = 1; i <= N; i++) scanf("%lf", &x0[i]);
    EU.resize(M); EV.resize(M); ECap.resize(M);
    for (int e = 0; e < M; e++) scanf("%d %d %lf", &EU[e], &EV[e], &ECap[e]);

    // ---- union-find spanning tree over all M candidate edges ----
    vector<int> dsu(N + 1);
    iota(dsu.begin(), dsu.end(), 0);
    function<int(int)> find = [&](int x) {
        while (dsu[x] != x) { dsu[x] = dsu[dsu[x]]; x = dsu[x]; }
        return x;
    };
    vector<vector<int>> treeAdj(N + 1);
    for (int e = 0; e < M; e++) {
        int ru = find(EU[e]), rv = find(EV[e]);
        if (ru != rv) {
            dsu[ru] = rv;
            treeAdj[EU[e]].push_back(EV[e]);
            treeAdj[EV[e]].push_back(EU[e]);
        }
    }

    // ---- root at node 1: parent, depth, BFS order ----
    vector<int> parent(N + 1, 0), depth(N + 1, 0), order;
    vector<char> vis(N + 1, 0);
    { queue<int> q; q.push(1); vis[1] = 1;
      while (!q.empty()) {
          int x = q.front(); q.pop(); order.push_back(x);
          for (int y : treeAdj[x]) if (!vis[y]) {
              vis[y] = 1; parent[y] = x; depth[y] = depth[x] + 1; q.push(y);
          }
      }
    }
    // ---- Euler tin/tout for subtree containment (iterative DFS, explicit
    //      child-index cursor per node; skip exactly the parent edge) ----
    vector<int> tin(N + 1, 0), tout(N + 1, 0); int timer = 0;
    {
        vector<int> idx(N + 1, 0);
        vector<char> entered(N + 1, 0);
        vector<int> stk; stk.push_back(1); entered[1] = 1; tin[1] = timer++;
        while (!stk.empty()) {
            int x = stk.back();
            bool advanced = false;
            while (idx[x] < (int)treeAdj[x].size()) {
                int y = treeAdj[x][idx[x]++];
                if (y == parent[x]) continue;       // skip edge back to parent
                if (entered[y]) continue;            // safety (shouldn't happen in a tree)
                entered[y] = 1; tin[y] = timer++;
                stk.push_back(y);
                advanced = true;
                break;
            }
            if (!advanced && stk.back() == x) {
                tout[x] = timer++;
                stk.pop_back();
            }
        }
    }
    // subtreeSize via reverse BFS order
    vector<int> subSize(N + 1, 1);
    for (int i = (int)order.size() - 1; i >= 0; i--) {
        int x = order[i];
        for (int y : treeAdj[x]) if (parent[y] == x) subSize[x] += subSize[y];
    }

    auto lca = [&](int u, int v) {
        while (depth[u] > depth[v]) u = parent[u];
        while (depth[v] > depth[u]) v = parent[v];
        while (u != v) { u = parent[u]; v = parent[v]; }
        return u;
    };

    vector<long long> acc(N + 1, 0);
    for (int e = 0; e < M; e++) {
        int u = EU[e], v = EV[e], l = lca(u, v);
        acc[u]++; acc[v]++; acc[l] -= 2;
    }
    // propagate child totals into parent, processing deepest nodes first
    vector<int> byDepth = order;
    sort(byDepth.begin(), byDepth.end(), [&](int a, int b){ return depth[a] > depth[b]; });
    for (int x : byDepth) if (x != 1) acc[parent[x]] += acc[x];

    // ratio[c] for each tree edge (parent[c], c), c != root
    double rmin = 1e18;
    vector<double> ratio(N + 1, 1e18);
    for (int c = 2; c <= N; c++) {
        if (c == 1) continue;
        // c may not be reached if graph disconnected; guard via vis
        if (!vis[c]) continue;
        int mn = min(subSize[c], N - subSize[c]);
        if (mn <= 0) continue;
        ratio[c] = (double)acc[c] / (double)mn;
        rmin = min(rmin, ratio[c]);
    }
    vector<char> critical(N + 1, 0);
    if (rmin < 1e17) {
        for (int c = 1; c <= N; c++)
            if (vis[c] && c != 1 && ratio[c] <= 2.0 * rmin + 1e-9) critical[c] = 1;
    }

    auto inSub = [&](int x, int c) { return tin[c] <= tin[x] && tin[x] < tout[c]; };

    vector<double> c_assign(M, 0.0), deg(N + 1, 0.0);
    double rem = C;
    vector<char> funded(M, 0);
    for (int c = 1; c <= N; c++) {
        if (!critical[c]) continue;
        for (int e = 0; e < M; e++) {
            if (funded[e]) continue;
            bool a = inSub(EU[e], c), b = inSub(EV[e], c);
            if (a != b) funded[e] = 1;
        }
    }
    for (int e = 0; e < M; e++) {
        if (!funded[e]) continue;
        int u = EU[e], v = EV[e];
        double amt = min({ECap[e], BOOST_TARGET, rem, DEGCAP - deg[u], DEGCAP - deg[v]});
        if (amt > 1e-12) { c_assign[e] = amt; deg[u] += amt; deg[v] += amt; rem -= amt; }
    }

    // ---- phase 2: max-min water-filling over everything still below cap ----
    vector<char> active(M, 1);
    for (int e = 0; e < M; e++) if (c_assign[e] >= ECap[e] - 1e-9) active[e] = 0;
    for (int iter = 0; iter < 200000 && rem > 1e-9; iter++) {
        vector<int> cnt(N + 1, 0);
        int nActive = 0;
        for (int e = 0; e < M; e++) if (active[e]) { cnt[EU[e]]++; cnt[EV[e]]++; nActive++; }
        if (nActive == 0) break;
        double delta = rem / nActive;
        for (int e = 0; e < M; e++) if (active[e]) delta = min(delta, ECap[e] - c_assign[e]);
        for (int i = 1; i <= N; i++) if (cnt[i] > 0) delta = min(delta, (DEGCAP - deg[i]) / cnt[i]);
        if (delta <= 1e-12) {
            bool changed = false;
            for (int e = 0; e < M; e++) if (active[e]) {
                int u = EU[e], v = EV[e];
                if (!(ECap[e] - c_assign[e] > 1e-9 && DEGCAP - deg[u] > 1e-9 && DEGCAP - deg[v] > 1e-9)) {
                    active[e] = 0; changed = true;
                }
            }
            if (!changed) break;
            continue;
        }
        for (int e = 0; e < M; e++) if (active[e]) {
            c_assign[e] += delta; deg[EU[e]] += delta; deg[EV[e]] += delta;
        }
        rem -= delta * nActive;
        for (int e = 0; e < M; e++) if (active[e]) {
            int u = EU[e], v = EV[e];
            if (!(ECap[e] - c_assign[e] > 1e-9 && DEGCAP - deg[u] > 1e-9 && DEGCAP - deg[v] > 1e-9))
                active[e] = 0;
        }
    }

    for (int e = 0; e < M; e++) printf("%.9f%c", c_assign[e], e + 1 == M ? '\n' : ' ');
    return 0;
}
