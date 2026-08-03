#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

// Grain-Lot Slate: maximum-weight independent set on a general conflict graph.
// Structure ladder: testId 1 is tiny (example scale); grows to n=200 by
// testId 10. The graph is a union of small conflict-cliques (silo bays / barge
// slots that force "pick at most one") plus random cross-cluster conflicts, so
// index-order greedy is weak, weight-aware greedy is better, and local search
// is better still -> heuristics diverge.

int main(int argc, char** argv) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);

    int n = 6 + (testId - 1) * 22;      // 6, 28, 50, ..., 204 -> clamp
    if (n > 200) n = 200;

    // ---- partition nodes into conflict-cliques ("shared bays") ----
    // cluster sizes 2..5; larger clusters on higher testIds (denser).
    int cluHi = 2 + (testId % 4);        // max cluster size 2..5
    vector<int> clusterOf(n + 1, 0);
    int cur = 1, cid = 0;
    while (cur <= n) {
        int sz = rnd.next(2, max(2, cluHi));
        sz = min(sz, n - cur + 1);
        cid++;
        for (int j = 0; j < sz; j++) clusterOf[cur + j] = cid;
        cur += sz;
    }
    int nClusters = cid;

    // weights: skewed so the "right" pick inside a cluster matters a lot.
    vector<int> w(n + 1);
    for (int i = 1; i <= n; i++) {
        // mixture: mostly modest, occasionally a big-ticket lot
        if (rnd.next(0.0, 1.0) < 0.30) w[i] = rnd.next(60, 99);
        else                            w[i] = rnd.next(1, 40);
    }

    // edge set (use a set to dedup; store as ordered pairs a<b)
    set<pair<int,int>> E;
    auto addEdge = [&](int a, int b) {
        if (a == b) return;
        if (a > b) swap(a, b);
        E.insert({a, b});
    };

    // clique edges inside each cluster
    for (int a = 1; a <= n; a++)
        for (int b = a + 1; b <= n; b++)
            if (clusterOf[a] == clusterOf[b]) addEdge(a, b);

    // random cross-cluster conflicts; density grows with testId
    long long crossTarget = (long long)n * (1 + (testId % 3)); // ~1n..3n
    long long added = 0, tries = 0, cap = crossTarget * 12 + 50;
    while (added < crossTarget && tries < cap) {
        tries++;
        int a = rnd.next(1, n), b = rnd.next(1, n);
        if (clusterOf[a] == clusterOf[b]) continue;
        int aa = a, bb = b; if (aa > bb) swap(aa, bb);
        if (E.count({aa, bb})) continue;
        E.insert({aa, bb});
        added++;
    }
    (void)nClusters;

    // relabel nodes with a random permutation so index order != structure.
    vector<int> perm(n + 1);
    for (int i = 1; i <= n; i++) perm[i] = i;
    for (int i = n; i >= 2; i--) {
        int j = rnd.next(1, i);
        swap(perm[i], perm[j]);
    }
    // perm[i] = new label of old node i
    vector<int> wNew(n + 1);
    for (int i = 1; i <= n; i++) wNew[perm[i]] = w[i];

    vector<pair<int,int>> edges;
    edges.reserve(E.size());
    for (auto& e : E) {
        int a = perm[e.first], b = perm[e.second];
        if (a > b) swap(a, b);
        edges.push_back({a, b});
    }
    shuffle(edges.begin(), edges.end());

    int m = (int)edges.size();
    printf("%d %d\n", n, m);
    for (int i = 1; i <= n; i++) printf("%d%c", wNew[i], i == n ? '\n' : ' ');
    for (auto& e : edges) printf("%d %d\n", e.first, e.second);
    return 0;
}
