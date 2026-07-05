#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

int main(int argc, char** argv) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);

    // ---- structure ladder: cluster of asteroids to split between two refineries ----
    // testId 1 is tiny (example scale); grows to a larger, denser cluster by testId 10.
    int n = 8 + 4 * testId;          // 12, 16, ..., 48  (always even)
    if (n % 2) n++;

    // Plant a hidden bipartition (two ore families) whose node ids are permuted,
    // so the "alternating dock" baseline is essentially a random cut (~half of the
    // heavy cross weight) while a good balanced bisection can recover much more.
    vector<int> perm(n);
    for (int i = 0; i < n; i++) perm[i] = i + 1;
    shuffle(perm.begin(), perm.end());
    vector<int> grp(n + 1, 0);       // hidden ore family 0/1 per node id
    vector<int> g0, g1;
    for (int i = 0; i < n; i++) {
        if (i < n / 2) { grp[perm[i]] = 0; g0.push_back(perm[i]); }
        else           { grp[perm[i]] = 1; g1.push_back(perm[i]); }
    }

    // heavy, dominant cross incompatibilities (between the two ore families) +
    // sparser intra-family frustration edges (some heavy) -> genuine hardness.
    int crossCount = n * (3 + testId);
    int intraCount = n + n * (testId % 3);

    vector<array<int, 3>> edges;      // u, v, w
    edges.reserve(crossCount + intraCount);

    auto pick = [&](vector<int>& v) { return v[rnd.next((size_t)0, v.size() - 1)]; };

    for (int c = 0; c < crossCount; c++) {
        int u = pick(g0), v = pick(g1);
        int w = rnd.next(15, 45);     // heavy cross weight
        edges.push_back({u, v, w});
    }
    for (int c = 0; c < intraCount; c++) {
        vector<int>& gg = (rnd.next(0, 1) ? g1 : g0);
        int u = pick(gg), v = pick(gg);
        int tries = 0;
        while (v == u && tries++ < 8) v = pick(gg);
        if (u == v) continue;
        int w;
        if (rnd.next(0.0, 1.0) < 0.30) w = rnd.next(15, 45); // heavy frustration
        else                          w = rnd.next(1, 8);    // light noise
        edges.push_back({u, v, w});
    }

    // shuffle edge order so listing position carries no structural signal
    shuffle(edges.begin(), edges.end());

    int m = (int)edges.size();
    printf("%d %d\n", n, m);
    for (auto& e : edges) printf("%d %d %d\n", e[0], e[1], e[2]);
    return 0;
}
