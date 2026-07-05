#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

// gen.cpp -- Orbital Debris Cleanup: Shared-Aperture Sweeper Deployment.
// testId 1..10 is a difficulty/structure ladder. N grows 6 -> 2600, field grows so the
// aperture menu makes disk-packing genuinely BINDING at useful radii (small aperture ->
// many clusters fit but low per-sweeper value; large aperture -> few fit / fuel dominates).
// Structure varies to be ADVERSARIAL:
//   * clustered blobs (even ids)  -> input-order greedy grabs colliding neighbours, loses.
//   * NEEDLE (id % 3 == 0)        -> one dominant cluster amid noise (defines a large B).
//   * PLANTED (id % 3 == 1, big)  -> a hidden grid of high-value clusters mutually
//                                    independent at one specific aperture Rp -> a strong
//                                    solution must FIND that aperture, not the largest one.
// Every test also carries a guaranteed high-value "prime" cluster so B > 0.
// Apertures are fixed; coordinates are distinct integers.

int main(int argc, char* argv[]) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);
    int idx = min(max(testId, 1), 10);

    int Ns[10] = {6, 30, 80, 150, 300, 500, 800, 1200, 1800, 2600};
    int N = Ns[idx - 1];
    const int K = 7;
    long long R[7] = {60, 100, 160, 260, 420, 680, 1100};
    int SP = 90;
    int L = max(300, (int)llround(sqrt((double)N) * SP));
    if (L > 60000) L = 60000;
    int NB = min(16, max(2, N / 8));
    long long D = 8000;

    bool clustered = (idx % 2 == 0);
    bool needle    = (idx % 3 == 0);
    bool planted   = (idx % 3 == 1) && (idx >= 4);

    set<pair<int,int>> used;
    vector<int> X, Y, W, C, A;
    auto tryAdd = [&](int x, int y, int w, int c, int a) -> bool {
        x = min(L, max(0, x)); y = min(L, max(0, y));
        if (!used.insert({x, y}).second) return false;
        X.push_back(x); Y.push_back(y); W.push_back(w); C.push_back(c); A.push_back(a);
        return true;
    };
    auto randCluster = [&](int& w, int& c, int& a) {
        w = rnd.next(400, 4000);
        c = rnd.next(1, 4);
        int act = rnd.next(40, 120);              // activation ~ act units of aperture
        a = (int)min(600000LL, (long long)w * act);
    };

    // ---- planted hidden grid: independent at aperture Rp, high value there ----
    if (planted) {
        long long Rp = R[3];                       // r = 260, exclusion 520
        int step = (int)(2 * Rp) + 40;             // > 2r -> mutually independent at Rp
        int cols = max(1, L / step);
        int placed = 0, target = min(N * 2 / 5, cols * cols);
        for (int gy = 0; gy < cols && placed < target; gy++)
            for (int gx = 0; gx < cols && placed < target; gx++) {
                int x = gx * step + rnd.next(0, 12);
                int y = gy * step + rnd.next(0, 12);
                // tuned so v is strongly positive exactly at Rp
                int w = rnd.next(2600, 3400);
                int c = 1;
                int a = (int)((long long)w * Rp - (long long)c * Rp * Rp - rnd.next(120000, 220000));
                a = max(0, a);
                if (tryAdd(x, y, w, c, a)) placed++;
            }
    }

    // ---- clustered blobs ----
    vector<pair<int,int>> centers;
    if (clustered) {
        int nc = max(2, (int)round(sqrt((double)N) / 2.0));
        for (int cc = 0; cc < nc; cc++) centers.push_back({rnd.next(0, L), rnd.next(0, L)});
    }

    // ---- fill the rest with noise clusters ----
    int guard = 0;
    while ((int)X.size() < N && guard < 40 * N + 1000) {
        guard++;
        int x, y;
        if (clustered && !centers.empty()) {
            auto& ct = centers[rnd.next(0, (int)centers.size() - 1)];
            int spread = max(30, L / (2 * (int)centers.size()));
            x = ct.first + rnd.next(-spread, spread);
            y = ct.second + rnd.next(-spread, spread);
        } else {
            x = rnd.next(0, L); y = rnd.next(0, L);
        }
        int w, c, a; randCluster(w, c, a);
        tryAdd(x, y, w, c, a);
    }
    // fallback: pure random until we hit N distinct points
    while ((int)X.size() < N) {
        int x = rnd.next(0, L), y = rnd.next(0, L);
        int w, c, a; randCluster(w, c, a);
        tryAdd(x, y, w, c, a);
    }

    // ---- guaranteed prime (large B) ----
    {
        int i = rnd.next(0, N - 1);
        W[i] = 3900; C[i] = 1;
        A[i] = (int)min(600000LL, (long long)3900 * rnd.next(45, 70));
    }
    // ---- needle: make one cluster utterly dominant amid noise ----
    if (needle) {
        int i = rnd.next(0, N - 1);
        W[i] = 4000; C[i] = 1;
        A[i] = (int)min(600000LL, (long long)4000 * 45);
    }

    // ---- bands from altitude (y) ----
    vector<int> B(N);
    for (int i = 0; i < N; i++) {
        long long bb = (long long)Y[i] * NB / ((long long)L + 1);
        B[i] = (int)min((long long)NB - 1, max(0LL, bb));
    }

    // ---- shuffle print order so input order != value order (weakens naive greedy) ----
    vector<int> ord(N);
    for (int i = 0; i < N; i++) ord[i] = i;
    shuffle(ord.begin(), ord.end());

    printf("%d %d %lld %d\n", N, K, D, NB);
    for (int t = 0; t < K; t++) printf("%lld%c", R[t], t + 1 < K ? ' ' : '\n');
    for (int k = 0; k < N; k++) {
        int i = ord[k];
        printf("%d %d %d %d %d %d\n", X[i], Y[i], W[i], C[i], A[i], B[i]);
    }
    return 0;
}
