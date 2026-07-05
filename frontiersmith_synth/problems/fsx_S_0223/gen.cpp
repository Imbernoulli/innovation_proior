#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

int main(int argc, char** argv) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);

    // ---- size ladder: tiny (example scale) at testId 1, large by testId 10 ----
    int n = 12 + 2 * testId * testId * testId; // 14, 28, 66, 140, ..., ~2012
    if (n > 2100) n = 2100;

    const int LO = 0, HI = 20000;

    vector<int> X(n + 1), Y(n + 1), C(n + 1);

    // structure varies per test so different heuristics diverge:
    // 0 uniform, 1 clustered blobs, 2 jittered grid, 3 skewed line-heavy
    int mode = testId % 4;

    if (mode == 0) {
        for (int i = 1; i <= n; i++) {
            X[i] = rnd.next(LO, HI);
            Y[i] = rnd.next(LO, HI);
        }
    } else if (mode == 1) {
        int blobs = 4 + testId;
        vector<pair<int,int>> ctr(blobs);
        for (auto& c : ctr) c = {rnd.next(LO, HI), rnd.next(LO, HI)};
        int spread = 400 + 200 * (testId % 3);
        for (int i = 1; i <= n; i++) {
            auto& c = ctr[rnd.next(0, blobs - 1)];
            int x = c.first + (int)rnd.next(-spread, spread);
            int y = c.second + (int)rnd.next(-spread, spread);
            X[i] = min(HI, max(LO, x));
            Y[i] = min(HI, max(LO, y));
        }
    } else if (mode == 2) {
        int side = max(2, (int)ceil(sqrt((double)n)));
        int step = (HI - LO) / side;
        int jit = max(1, step / 4);
        for (int i = 1; i <= n; i++) {
            int gx = (i - 1) % side, gy = (i - 1) / side;
            int x = LO + gx * step + (int)rnd.next(-jit, jit);
            int y = LO + gy * step + (int)rnd.next(-jit, jit);
            X[i] = min(HI, max(LO, x));
            Y[i] = min(HI, max(LO, y));
        }
    } else {
        // skewed: most points near a diagonal corridor, some outliers
        for (int i = 1; i <= n; i++) {
            if (rnd.next(0, 9) < 8) {
                int t = rnd.next(LO, HI);
                int off = (int)rnd.next(-800, 800);
                X[i] = min(HI, max(LO, t));
                Y[i] = min(HI, max(LO, t + off));
            } else {
                X[i] = rnd.next(LO, HI);
                Y[i] = rnd.next(LO, HI);
            }
        }
    }

    // heterogeneous per-node capacities in {2,3,4}; mix varies with test.
    // tighter tests (more 2s) force path-like meshes; looser tests allow branching.
    int p2 = 30 + (testId % 5) * 8;   // percent capacity-2 nodes
    int p4 = 15 + (testId % 3) * 5;   // percent capacity-4 nodes
    for (int i = 1; i <= n; i++) {
        int r = rnd.next(0, 99);
        if (r < p2) C[i] = 2;
        else if (r < p2 + p4) C[i] = 4;
        else C[i] = 3;
    }
    // guarantee at least a couple of high-capacity hubs so branching is possible
    if (n >= 3) {
        C[rnd.next(1, n)] = 4;
        C[rnd.next(1, n)] = 4;
    }

    // shuffle input order so the baseline chain (index order) is geometrically bad
    vector<int> perm(n);
    for (int i = 0; i < n; i++) perm[i] = i + 1;
    shuffle(perm.begin(), perm.end());

    printf("%d\n", n);
    for (int i = 0; i < n; i++) {
        int j = perm[i];
        printf("%d %d %d\n", X[j], Y[j], C[j]);
    }
    return 0;
}
