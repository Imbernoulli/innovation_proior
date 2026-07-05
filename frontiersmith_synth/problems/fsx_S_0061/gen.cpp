#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main(int argc, char** argv) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);

    // difficulty ladder: tiny at testId 1, growing grid otherwise.
    int R, C;
    if (testId == 1) { R = 2; C = 3; }
    else { R = 6 + 3 * testId; C = 6 + 3 * testId; } // up to 36x36 = 1296 nodes at testId 10

    int n = R * C;
    auto id = [&](int r, int c) { return r * C + c + 1; };
    int s = id(0, 0); // correlator hub at a corner (degree 2)

    struct E { int u, v; ll w; };
    vector<E> es;
    // Grid edges. A weight-1 "backbone" runs along the top row (row 0 horizontals) and the
    // left column (column 0 verticals): a cheap L-corridor anchored at the hub. All interior
    // links are slow (latency 8..20), so streams ride the backbone then cut across expensive
    // interior fiber. Cutting backbone links near the hub forces long interior detours.
    for (int r = 0; r < R; r++) {
        for (int c = 0; c < C; c++) {
            if (c + 1 < C) {
                bool bb = (r == 0);
                ll w = bb ? 1 : rnd.next(8, 20);
                es.push_back({id(r, c), id(r, c + 1), w});
            }
            if (r + 1 < R) {
                bool bb = (c == 0);
                ll w = bb ? 1 : rnd.next(8, 20);
                es.push_back({id(r, c), id(r + 1, c), w});
            }
        }
    }
    int m = (int)es.size();

    // Science antennas (targets): distinct nodes in the "far" half of the plateau, so their
    // streams travel long paths that are worth interdicting. Exclude the hub.
    int thr = (R - 1 + C - 1) / 2; // require r+c >= thr
    vector<int> pool;
    for (int r = 0; r < R; r++)
        for (int c = 0; c < C; c++) {
            int node = id(r, c);
            if (node == s) continue;
            if (r + c >= thr) pool.push_back(node);
        }
    if (pool.empty()) {
        for (int v = 1; v <= n; v++) if (v != s) pool.push_back(v);
    }
    shuffle(pool.begin(), pool.end());
    int q = max(3, (n) / 20);
    q = min(q, (int)pool.size());
    vector<int> targets(pool.begin(), pool.begin() + q);
    sort(targets.begin(), targets.end());

    // Cut budget: cardinality, sized so several bottlenecks are reachable but full 10x cap
    // is not free. Grows with the array footprint.
    int k = max(3, (R + C) / 4);
    k = min(k, m);

    printf("%d %d %d %d %d\n", n, m, s, q, k);
    for (int i = 0; i < q; i++) printf("%d%c", targets[i], i + 1 == q ? '\n' : ' ');
    for (auto& e : es) printf("%d %d %lld\n", e.u, e.v, e.w);
    return 0;
}
