#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

// ---------------------------------------------------------------------------
// Aurora Festival: Stage Reconfiguration Scheduling  (family: scheduling-geometric-setup)
//
// Adversarial generator. Each test embeds:
//   (a) PLANTED cluster geometry -- acts fall into K clusters in profile space, so a
//       geometry-aware partition can nearly zero out per-stage setup.
//   (b) TRAP structure -- runtimes are drawn INDEPENDENTLY of cluster, so any
//       shortest/longest-processing-time ordering scrambles the clusters and pays huge
//       setup (the SPT/LPT trap named in the brief).
//   (c) NEEDLE (testId >= 5) -- one distinctive far-away, long act that must be handled
//       specially amid the cluster noise.
//   (d) the largest tests FILL the constraint envelope (N ~ 1200, M ~ 12, d = 5).
// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);

    const int Nmax = 1200, Mmax = 12;
    int N, M, d;
    if (testId <= 1) { N = 6; M = 2; d = 2; }
    else {
        double f = (testId - 1) / 9.0;
        N = (int)round(8 + f * (Nmax - 8));
        M = 2 + (int)round(f * (Mmax - 2));
        d = 2 + (testId - 1) / 3;            // 2,2,2,3,3,3,4,4,4,5
    }
    if (M >= N) M = max(2, N - 1);

    // number of clusters ~ M with variation (some tests fewer clusters than stages,
    // some more -> partition/routing tension)
    int K = M + (int)rnd.next(-1, 2);        // M-1 .. M+2
    if (K < 2) K = 2;
    if (K > N) K = N;

    const int COORD = 50;                    // cluster centers live in [0, COORD]^d
    const int JIT   = 3;                     // within-cluster jitter

    vector<vector<int>> center(K, vector<int>(d));
    for (int c = 0; c < K; c++)
        for (int k = 0; k < d; k++)
            center[c][k] = rnd.next(0, COORD);

    // skewed cluster weights -> unequal cluster sizes (load imbalance vs geometry)
    vector<double> w(K);
    double wsum = 0;
    for (int c = 0; c < K; c++) { w[c] = rnd.next(1, 5); wsum += w[c]; }

    vector<int> dur(N);
    vector<vector<int>> prof(N, vector<int>(d));

    for (int j = 0; j < N; j++) {
        double r = rnd.next(0.0, wsum), acc = 0; int c = 0;
        for (c = 0; c < K; c++) { acc += w[c]; if (r <= acc) break; }
        if (c >= K) c = K - 1;
        for (int k = 0; k < d; k++) {
            int v = center[c][k] + (int)rnd.next(-JIT, JIT);
            if (v < 0) v = 0;
            prof[j][k] = v;
        }
        // runtime INDEPENDENT of cluster -> ordering by runtime scrambles geometry (trap)
        dur[j] = rnd.next(1, 500);
    }

    // TRAP reinforcement: a few heavy acts scattered across clusters, so a pure
    // load-balancer (LPT) spreads clusters across all stages and blows up setup.
    int heavy = (testId >= 3) ? 1 + testId / 2 : 0;
    for (int h = 0; h < heavy && h < N; h++) {
        int j = rnd.next(0, N - 1);
        dur[j] = rnd.next(1500, 3000);
    }

    // NEEDLE: one distinctive far-away long act amid the clusters.
    if (testId >= 5) {
        int j = rnd.next(0, N - 1);
        for (int k = 0; k < d; k++) {
            int v = COORD + (int)rnd.next(20, 40);   // outside every cluster
            if (v > 90) v = 90;
            prof[j][k] = v;
        }
        dur[j] = rnd.next(2000, 3400);
    }

    // shuffle so input index != cluster grouping
    vector<int> ord(N);
    for (int j = 0; j < N; j++) ord[j] = j;
    shuffle(ord.begin(), ord.end());

    printf("%d %d %d\n", N, M, d);
    for (int i = 0; i < N; i++) {
        int j = ord[i];
        printf("%d", dur[j]);
        for (int k = 0; k < d; k++) printf(" %d", prof[j][k]);
        printf("\n");
    }
    return 0;
}
