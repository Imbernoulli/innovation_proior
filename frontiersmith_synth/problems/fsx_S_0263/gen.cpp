#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

// Neon Midway: maximum-weight independent set on a general conflict graph, at
// LARGE scale (n up to 5000). Structure ladder: testId 1 is tiny (example
// scale), growing geometrically to n=5000 by testId 10.
//
// The conflict graph is deliberately a *general* graph that defeats the easy
// special cases:
//   * power-substation cliques  -> "pick at most one ride per substation"
//   * a midway crowd-flow backbone (a random cycle) -> adjacent rides jam
//   * random cross conflicts     -> density grows with testId
//   * a few "headliner" super-connectors (high weight, high degree) -> classic
//     traps for weight-greedy.
// Weights are skewed (mostly modest, occasional big-ticket) and nodes are
// randomly relabelled so index order != structure. This makes index-order
// greedy weak, weight-aware greedy better, and randomized local search better
// still -> heuristics genuinely diverge.

int main(int argc, char** argv) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);

    // geometric size ladder: 8, ~14, ~24, ..., 5000
    double lo = 8.0, hi = 5000.0;
    double t = (testId - 1) / 9.0;
    int n = (int)llround(lo * pow(hi / lo, t));
    if (testId == 1) n = 8;
    if (n < 2) n = 2;
    if (n > 5000) n = 5000;

    // ---- partition nodes into power-substation cliques ----
    int cluHi = 2 + (testId % 5);        // max cluster size 2..6
    vector<int> clusterOf(n + 1, 0);
    int cur = 1, cid = 0;
    while (cur <= n) {
        int sz = rnd.next(1, max(2, cluHi));
        sz = min(sz, n - cur + 1);
        cid++;
        for (int j = 0; j < sz; j++) clusterOf[cur + j] = cid;
        cur += sz;
    }

    // weights: mixture, skewed.
    vector<int> w(n + 1);
    for (int i = 1; i <= n; i++) {
        if (rnd.next(0.0, 1.0) < 0.25) w[i] = rnd.next(600, 1000); // big-ticket
        else                            w[i] = rnd.next(1, 350);   // modest
    }

    // headliner super-connectors: a few high-weight, high-degree nodes.
    int nHead = (n >= 200) ? (2 + testId / 3) : 0;
    vector<int> heads;
    {
        vector<int> pool(n);
        for (int i = 0; i < n; i++) pool[i] = i + 1;
        for (int k = 0; k < nHead && k < n; k++) {
            int j = rnd.next(k, n - 1);
            swap(pool[k], pool[j]);
            heads.push_back(pool[k]);
            w[pool[k]] = rnd.next(700, 1000);
        }
    }

    set<pair<int,int>> E;
    auto addEdge = [&](int a, int b) {
        if (a == b) return;
        if (a > b) swap(a, b);
        E.insert({a, b});
    };

    // clique edges inside each substation cluster
    {
        // group by cluster to avoid O(n^2) scan
        vector<vector<int>> byClu(cid + 1);
        for (int i = 1; i <= n; i++) byClu[clusterOf[i]].push_back(i);
        for (int c = 1; c <= cid; c++) {
            auto& g = byClu[c];
            for (size_t a = 0; a < g.size(); a++)
                for (size_t b = a + 1; b < g.size(); b++)
                    addEdge(g[a], g[b]);
        }
    }

    // midway crowd-flow backbone: a random cyclic tour over all rides.
    {
        vector<int> order(n);
        for (int i = 0; i < n; i++) order[i] = i + 1;
        for (int i = n - 1; i >= 1; i--) swap(order[i], order[rnd.next(0, i)]);
        for (int i = 0; i < n; i++) addEdge(order[i], order[(i + 1) % n]);
    }

    // random cross conflicts; density grows with testId, capped for file size.
    long long crossTarget = (long long)n * (2 + (testId % 4)); // ~2n..5n
    long long capEdges = 550000;                                // file-size guard
    long long added = 0, tries = 0, cap = crossTarget * 8 + 100;
    while (added < crossTarget && tries < cap && (long long)E.size() < capEdges) {
        tries++;
        int a = rnd.next(1, n), b = rnd.next(1, n);
        if (a == b) continue;
        int aa = a, bb = b; if (aa > bb) swap(aa, bb);
        if (E.count({aa, bb})) continue;
        E.insert({aa, bb});
        added++;
    }

    // wire the headliner super-connectors to many random rides.
    for (int h : heads) {
        int deg = rnd.next(n / 8, n / 3 + 1);
        for (int k = 0; k < deg && (long long)E.size() < capEdges; k++) {
            int b = rnd.next(1, n);
            addEdge(h, b);
        }
    }

    // relabel nodes with a random permutation so index order != structure.
    vector<int> perm(n + 1);
    for (int i = 1; i <= n; i++) perm[i] = i;
    for (int i = n; i >= 2; i--) {
        int j = rnd.next(1, i);
        swap(perm[i], perm[j]);
    }
    vector<int> wNew(n + 1);
    for (int i = 1; i <= n; i++) wNew[perm[i]] = w[i];

    vector<pair<int,int>> edges;
    edges.reserve(E.size());
    for (auto& e : E) {
        int a = perm[e.first], b = perm[e.second];
        if (a > b) swap(a, b);
        edges.push_back({a, b});
    }
    for (int i = (int)edges.size() - 1; i >= 1; i--)
        swap(edges[i], edges[rnd.next(0, i)]);

    int m = (int)edges.size();

    // buffered output
    string out;
    out.reserve((size_t)m * 12 + n * 5 + 32);
    char buf[64];
    sprintf(buf, "%d %d\n", n, m); out += buf;
    for (int i = 1; i <= n; i++) {
        sprintf(buf, "%d%c", wNew[i], i == n ? '\n' : ' ');
        out += buf;
    }
    for (auto& e : edges) {
        sprintf(buf, "%d %d\n", e.first, e.second);
        out += buf;
    }
    fputs(out.c_str(), stdout);
    return 0;
}
