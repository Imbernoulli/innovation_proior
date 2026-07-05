#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

// gen.cpp -- Solar Farm DC Collection Grid: port-capped, fee-weighted, L1-trench
// connected network.  testId is a difficulty/structure ladder: tiny at 1 growing to
// N=2000 at 10.  Structure varies: even tests are clustered (long chains, worst-case for
// naive paths); weight profiles alternate uniform / skewed-expensive so the per-port
// termination fee is a real optimization dimension.  Coordinates are distinct integers;
// caps in {2,3,4} with cap>=2 so the input-order chain baseline is always feasible.  Input
// order is shuffled so that chain baseline is a genuinely long reference path.
int main(int argc, char* argv[]) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);
    int idx = min(max(testId, 1), 10);

    int Ns[10] = {4, 25, 60, 120, 260, 450, 700, 1100, 1600, 2000};
    int N = Ns[idx - 1];
    const int C = 10000;
    const int W = 2000;
    bool clustered = (idx % 2 == 0);       // even -> clustered blobs
    bool skewedFee = (idx % 3 != 0);       // most tests: a few cheap combiners, rest expensive

    // ---- coordinates (distinct) ----
    set<pair<int,int>> used;
    vector<int> xs, ys;
    if (clustered) {
        int nClusters = max(2, (int)round(sqrt((double)N) / 2.0));
        vector<pair<int,int>> centers;
        for (int c = 0; c < nClusters; c++)
            centers.push_back({rnd.next(0, C), rnd.next(0, C)});
        int spread = max(50, C / (2 * nClusters));
        while ((int)xs.size() < N) {
            auto& ct = centers[rnd.next(0, nClusters - 1)];
            int x = ct.first + rnd.next(-spread, spread);
            int y = ct.second + rnd.next(-spread, spread);
            x = min(C, max(0, x));
            y = min(C, max(0, y));
            if (used.insert({x, y}).second) { xs.push_back(x); ys.push_back(y); }
        }
    } else {
        while ((int)xs.size() < N) {
            int x = rnd.next(0, C), y = rnd.next(0, C);
            if (used.insert({x, y}).second) { xs.push_back(x); ys.push_back(y); }
        }
    }

    // ---- caps in {2,3,4} ----
    vector<int> cap(N);
    for (int i = 0; i < N; i++) {
        int r = rnd.next(0, 99);
        cap[i] = (r < 45) ? 2 : (r < 80 ? 3 : 4);
    }

    // ---- per-port termination fees ----
    vector<int> w(N);
    for (int i = 0; i < N; i++) {
        if (skewedFee) {
            // ~20% cheap combiners, the rest expensive -> hubs must be placed carefully
            if (rnd.next(0, 99) < 20) w[i] = rnd.next(1, W / 20 + 1);
            else                      w[i] = rnd.next(W / 2, W);
        } else {
            w[i] = rnd.next(1, W);
        }
    }

    // ---- shuffle so input order is an arbitrary (long) chain ----
    vector<int> perm(N);
    for (int i = 0; i < N; i++) perm[i] = i;
    shuffle(perm.begin(), perm.end());

    printf("%d\n", N);
    for (int k = 0; k < N; k++) {
        int i = perm[k];
        printf("%d %d %d %d\n", xs[i], ys[i], cap[i], w[i]);
    }
    return 0;
}
