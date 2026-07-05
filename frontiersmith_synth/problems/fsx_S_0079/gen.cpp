#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main(int argc, char** argv) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);

    // ---- structure ladder: number of arena stations ----
    // testId 1 tiny (example scale); grows to ~58 by testId 10.
    int n = 4 + 6 * (testId - 1);
    if (n > 60) n = 60;

    // floor coordinate range (fixed; ratios are scale-invariant)
    const int L = 2000;

    // distinct integer floor positions
    vector<int> px(n + 1), py(n + 1);
    set<pair<int,int>> used;
    for (int v = 1; v <= n; v++) {
        int x, y;
        do { x = rnd.next(0, L); y = rnd.next(0, L); } while (used.count({x, y}));
        used.insert({x, y});
        px[v] = x; py[v] = y;
    }

    auto man = [&](int a, int b) -> int {
        int d = abs(px[a] - px[b]) + abs(py[a] - py[b]);
        return d < 1 ? 1 : d;
    };

    // port limits: 2 or 3, with a share of tight (=2) switches that grows a bit.
    // higher testId -> more constrained fabric.
    double tightProb = 0.30 + 0.03 * (testId % 4); // 0.30..0.39
    vector<int> d(n + 1);
    for (int v = 1; v <= n; v++)
        d[v] = (rnd.next(0.0, 1.0) < tightProb) ? 2 : 3;

    // ---- candidate cable runs ----
    vector<array<int,3>> edges; // u, v, w

    // (1) guaranteed daisy-chain edges (i, i+1) so a feasible layout always exists
    for (int i = 1; i < n; i++)
        edges.push_back({i, i + 1, man(i, i + 1)});

    // (2) nearest-neighbour cables: geometry that a good fabric exploits
    int knn = min(n - 1, 6 + testId % 3); // 6..8 nearest neighbours
    for (int v = 1; v <= n; v++) {
        vector<pair<int,int>> cand; // (dist, other)
        for (int u = 1; u <= n; u++) if (u != v) cand.push_back({man(v, u), u});
        sort(cand.begin(), cand.end());
        for (int t = 0; t < knn && t < (int)cand.size(); t++) {
            int u = cand[t].second;
            if (u > v) edges.push_back({v, u, cand[t].first}); // avoid dup direction
        }
    }

    // (3) random long-range cables add alternative routings (more open-ended search)
    int extra = 3 * n;
    for (int e = 0; e < extra; e++) {
        int a = rnd.next(1, n), b = rnd.next(1, n);
        if (a == b) { e--; continue; }
        edges.push_back({a, b, man(a, b)});
    }

    // cap total edge count to the stated bound
    if ((int)edges.size() > 6000) edges.resize(6000);

    // shuffle so edge index != structural role (but keep at least one (i,i+1) each:
    // they were pushed first and are never dropped since count stays < 6000 for n<=60)
    shuffle(edges.begin(), edges.end());

    int m = (int)edges.size();
    printf("%d %d\n", n, m);
    for (int v = 1; v <= n; v++) printf("%d%c", d[v], v == n ? '\n' : ' ');
    for (auto& e : edges)
        printf("%d %d %d\n", e[0], e[1], e[2]);
    return 0;
}
