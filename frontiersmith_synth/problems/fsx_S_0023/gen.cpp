#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

int main(int argc, char** argv) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);

    // ---- structure ladder: general (non-bipartite) conflict graph ----
    // testId 1 is tiny (example scale); grows to a large, cliquey graph by testId 10.
    int n;
    if (testId == 1) n = 6;
    else n = min(3000, 300 * testId);       // 600, 900, ..., 3000

    // average degree grows with testId -> denser conflict graphs are harder.
    double avgDeg = 3.0 + 1.0 * testId;     // ~4 .. ~13
    long long targetEdges = (long long)(avgDeg * n / 2.0);
    if (testId == 1) targetEdges = 6;       // keep the tiny example-scale case sparse

    // ---- skewed hype weights: mostly modest, a few stars ----
    // heavy-tailed so the single best (baseline B) is much smaller than a good roster.
    vector<int> w(n + 1);
    for (int i = 1; i <= n; i++) {
        if (rnd.next(0.0, 1.0) < 0.12)
            w[i] = rnd.next(500, 1000);     // rare stars
        else
            w[i] = rnd.next(1, 200);        // common performers
    }

    // ---- edges ----
    // We build a set of undirected edges. Mix of Erdos-Renyi random rivalries plus
    // planted cliques (so the graph is genuinely general / non-bipartite and hard).
    set<pair<int,int>> es;
    auto addEdge = [&](int a, int b) {
        if (a == b) return;
        if (a > b) swap(a, b);
        es.insert({a, b});
    };

    // planted cliques among random performers -> triangles/cliques force hardness
    int nCliques = 2 + testId;
    for (int c = 0; c < nCliques && n >= 12; c++) {
        int sz = rnd.next(3, min(6, n));
        vector<int> pick;
        for (int j = 0; j < sz; j++) pick.push_back(rnd.next(1, n));
        for (int a = 0; a < sz; a++)
            for (int b = a + 1; b < sz; b++)
                addEdge(pick[a], pick[b]);
    }

    // random rivalries until we reach the target edge count (capped by limits)
    long long cap = min<long long>(targetEdges, 25000);
    int guard = 0;
    while ((long long)es.size() < cap && guard < 40 * (int)cap + 1000) {
        guard++;
        int a = rnd.next(1, n), b = rnd.next(1, n);
        addEdge(a, b);
    }

    // ---- emit (shuffle edge order so structure is hidden) ----
    vector<pair<int,int>> edges(es.begin(), es.end());
    shuffle(edges.begin(), edges.end());
    // also randomly flip endpoint order
    for (auto& e : edges)
        if (rnd.next(0, 1)) swap(e.first, e.second);

    int m = (int)edges.size();
    printf("%d %d\n", n, m);
    for (int i = 1; i <= n; i++)
        printf("%d%c", w[i], i == n ? '\n' : ' ');
    for (auto& e : edges)
        printf("%d %d\n", e.first, e.second);
    return 0;
}
