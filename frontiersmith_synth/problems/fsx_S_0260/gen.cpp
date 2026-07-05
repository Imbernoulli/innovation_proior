#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

int main(int argc, char** argv) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);

    // ---- size ladder: testId 1 tiny (example scale), growing to large ----
    int n = (testId == 1) ? 6 : min(2000, 200 * (testId - 1)); // 6,200,400,...,1800
    if (n % 2 == 1) n++;                                        // keep even for clean balance

    // planted balanced hidden bipartition, UNCORRELATED with wagon index parity
    vector<int> label(n + 1, 0);
    for (int i = 1; i <= n; i++) label[i] = (i <= n / 2) ? 0 : 1;
    for (int i = n; i >= 2; i--) { int j = rnd.next(1, i); swap(label[i], label[j]); }

    // structure / weight knobs vary a little per test
    int degCross = 5 + (testId % 3);      // 5..7 heavy cross links per wagon
    int heavyLo  = 40 + (testId % 4) * 8; // 40..64
    int heavyHi  = heavyLo + 30;          // heavy band
    int lightHi  = 4 + (testId % 3);      // 4..6 noise weight ceiling
    int noiseCnt = 2 * n;                 // background noise edges

    vector<array<int,3>> edges;
    set<pair<int,int>> seen;

    auto addPair = [&](int u, int v, int w) {
        edges.push_back({u, v, w});
    };

    // planted heavy cut: connect each wagon to random OPPOSITE-label wagons
    // (these SHOULD be split; the hidden bipartition splits them all)
    for (int u = 1; u <= n; u++) {
        int tries = 0, made = 0;
        while (made < degCross && tries < degCross * 6) {
            tries++;
            int v = rnd.next(1, n);
            if (v == u || label[v] == label[u]) continue;
            int a = min(u, v), b = max(u, v);
            if (seen.count({a, b})) continue;
            seen.insert({a, b});
            addPair(u, v, rnd.next(heavyLo, heavyHi));
            made++;
        }
    }

    // background noise: random light pairs anywhere (multigraph allowed)
    for (int e = 0; e < noiseCnt; e++) {
        int u = rnd.next(1, n), v = rnd.next(1, n);
        if (u == v) { e--; continue; }
        addPair(u, v, rnd.next(1, lightHi));
    }

    // shuffle so listing order is unrelated to structure
    shuffle(edges.begin(), edges.end());

    int m = (int)edges.size();
    // balance tolerance: modest slack; parity baseline (diff 0) always fits
    int tol = max(1, n / 10);

    printf("%d %d %d\n", n, m, tol);
    for (auto& e : edges) printf("%d %d %d\n", e[0], e[1], e[2]);
    return 0;
}
