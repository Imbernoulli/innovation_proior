#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

// Glacier Sensor Net: maximum-weight independent set on a general conflict graph.
// testId is a structure ladder: tiny sparse (example scale) -> large, denser,
// with a handful of mutually-conflicting high-value "beacon" sites that force a
// real trade-off (take one big beacon, or assemble many small conflict-free sites).
int main(int argc, char** argv) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);

    int n = 16 + 18 * (testId - 1);          // 16, 34, ..., 178
    if (n < 1) n = 1;

    // ---- vertex weights: heavy-tailed so the single best site (baseline B) is
    // large enough that a good deployment lands in a graded, non-saturating range.
    int nbeacon = 2 + testId / 3;
    nbeacon = min(nbeacon, n);
    vector<int> perm(n);
    for (int i = 0; i < n; i++) perm[i] = i + 1;
    shuffle(perm.begin(), perm.end());
    vector<char> isBeacon(n + 1, 0);
    for (int i = 0; i < nbeacon; i++) isBeacon[perm[i]] = 1;

    vector<long long> w(n + 1);
    for (int i = 1; i <= n; i++) {
        if (isBeacon[i]) w[i] = rnd.next(120, 300);
        else             w[i] = rnd.next(3, 30);
    }

    // ---- conflict edges: Erdos-Renyi base + beacons made mutually conflicting and
    // high-degree, so at most one beacon fits and picking it blocks many sites.
    double p = 0.06 + 0.02 * (testId % 4);
    vector<vector<char>> A(n + 1, vector<char>(n + 1, 0));
    vector<pair<int,int>> edges;
    auto add = [&](int u, int v) {
        if (u == v) return;
        if (A[u][v]) return;
        A[u][v] = A[v][u] = 1;
        edges.push_back({u, v});
    };

    for (int u = 1; u <= n; u++)
        for (int v = u + 1; v <= n; v++)
            if (rnd.next(0.0, 1.0) < p) add(u, v);

    vector<int> bl;
    for (int i = 1; i <= n; i++) if (isBeacon[i]) bl.push_back(i);
    for (size_t i = 0; i < bl.size(); i++)
        for (size_t j = i + 1; j < bl.size(); j++)
            add(bl[i], bl[j]);
    for (int b : bl) {
        int lo = max(1, n / 6), hi = max(lo, n / 3);
        int de = rnd.next(lo, hi);
        for (int e = 0; e < de; e++) add(b, rnd.next(1, n));
    }

    shuffle(edges.begin(), edges.end());

    int m = (int)edges.size();
    printf("%d %d\n", n, m);
    for (int i = 1; i <= n; i++) {
        printf("%lld", w[i]);
        printf(i == n ? "\n" : " ");
    }
    for (auto& e : edges) printf("%d %d\n", e.first, e.second);
    return 0;
}
