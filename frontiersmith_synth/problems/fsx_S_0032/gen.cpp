#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main(int argc, char** argv) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);

    // ---- size ladder: testId 1 tiny (sanity), growing to large by testId 10 ----
    int n;
    if (testId == 1) n = 12;
    else n = min(2000, 150 * (testId - 1));   // 150, 300, ..., up to 1350 (cap 2000)
    if (n < 4) n = 4;

    // planted (noisy) two-community structure; labels random per index so the
    // input-order balance baseline is uncorrelated with the good bipartition.
    double gf = rnd.next(0.40, 0.60);
    vector<int> grp(n + 1);
    for (int i = 1; i <= n; i++) grp[i] = (rnd.next(0.0, 1.0) < gf) ? 0 : 1;
    grp[1] = 0; grp[2] = 1;                    // guarantee both communities nonempty

    vector<int> a(n + 1);
    ll totalDraw = 0, maxDraw = 0;
    for (int i = 1; i <= n; i++) {
        a[i] = rnd.next(1, 10);
        totalDraw += a[i];
        maxDraw = max(maxDraw, (ll)a[i]);
    }

    vector<int> g0, g1;
    for (int i = 1; i <= n; i++) (grp[i] == 0 ? g0 : g1).push_back(i);

    vector<array<int, 3>> edges;               // u, v, w

    // heavy cross-community "shared-fan" flows dominate -> big cut room
    int dcross = 3 + (testId % 3);             // 3..5 per node
    for (int i = 1; i <= n; i++) {
        vector<int>& opp = (grp[i] == 0) ? g1 : g0;
        for (int c = 0; c < dcross; c++) {
            int j = opp[rnd.next(0, (int)opp.size() - 1)];
            if (j == i) continue;
            int w = rnd.next(4, 12);
            edges.push_back({i, j, w});
        }
    }

    // sparse light intra-community flows: cutting them is a minor bonus, keeping
    // them uncut is a minor loss -> the planted cut is not exactly optimal.
    int dintra = 1;
    for (int i = 1; i <= n; i++) {
        vector<int>& same = (grp[i] == 0) ? g0 : g1;
        if (same.size() < 2) continue;
        for (int c = 0; c < dintra; c++) {
            int j = same[rnd.next(0, (int)same.size() - 1)];
            if (j == i) continue;
            int w = rnd.next(1, 3);
            edges.push_back({i, j, w});
        }
    }

    // random noise flows anywhere -> the optimum is genuinely unknown
    int noise = n / 2;
    for (int c = 0; c < noise; c++) {
        int u = rnd.next(1, n), v = rnd.next(1, n);
        if (u == v) { c--; continue; }
        int w = rnd.next(1, 12);
        edges.push_back({u, v, w});
    }

    // balance tolerance: always >= max draw (baseline feasible), but tight enough
    // for the constraint to bite on the large instances.
    ll tau = max(maxDraw, (ll)(0.12 * (double)totalDraw));

    shuffle(edges.begin(), edges.end());

    int m = (int)edges.size();
    printf("%d %d %lld\n", n, m, tau);
    for (int i = 1; i <= n; i++) printf("%d%c", a[i], i == n ? '\n' : ' ');
    for (auto& e : edges) printf("%d %d %d\n", e[0], e[1], e[2]);
    return 0;
}
