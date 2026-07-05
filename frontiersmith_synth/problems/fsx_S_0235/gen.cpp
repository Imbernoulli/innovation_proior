#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

int main(int argc, char* argv[]) {
    registerGen(argc, argv, 1);
    int t = atoi(argv[1]);

    // Difficulty ladder: N grows from tiny (t=1) to ~250 (t=10).
    int N = 6 + (int)std::llround((t - 1) * 27.0);   // t=1 ->6 ... t=10 ->249
    if (N < 4) N = 4;
    if (N > 260) N = 260;

    // Number of extra shortcut trenches. Alternate sparse/dense to force
    // different structure across tests (divergence between heuristics).
    int spine = N - 1;
    int maxExtra = 3 * N;                 // keep M <= 4*N
    int extra;
    if (t % 3 == 1)      extra = rnd.next(N / 2, N);              // sparse
    else if (t % 3 == 2) extra = rnd.next(N, 2 * N);              // medium
    else                 extra = rnd.next(2 * N, maxExtra);       // dense
    extra = max(1, min(extra, maxExtra));
    int M = spine + extra;

    // Number of research huts.
    int Kmax = max(2, N / 3);
    int K = rnd.next(2, Kmax);

    // Build edge list. Backbone first (heavier), then shortcuts (cheaper on
    // average so a Steiner tree can substantially beat the backbone).
    vector<array<int,3>> edges;
    edges.reserve(M);
    for (int i = 1; i <= N - 1; i++) {
        int w = rnd.next(60, 100);                 // heavy backbone
        edges.push_back({i, i + 1, w});
    }
    for (int e = 0; e < extra; e++) {
        int u = rnd.next(1, N);
        int v = rnd.next(1, N);
        while (v == u) v = rnd.next(1, N);
        // cheaper shortcuts, occasionally very cheap to create real structure
        int w;
        int roll = rnd.next(0, 9);
        if (roll < 3) w = rnd.next(1, 15);
        else          w = rnd.next(10, 70);
        edges.push_back({u, v, w});
    }

    // Choose K distinct hut sites. Mix: sometimes spread across the ids,
    // sometimes clustered, to vary the optimal structure.
    vector<int> ids(N);
    for (int i = 0; i < N; i++) ids[i] = i + 1;
    vector<int> huts;
    if (t % 2 == 0 && N >= 2 * K) {
        // clustered: pick a random window and sample within it
        int lo = rnd.next(1, N - K);
        set<int> chosen;
        while ((int)chosen.size() < K) chosen.insert(rnd.next(lo, min(N, lo + 2 * K)));
        huts.assign(chosen.begin(), chosen.end());
    } else {
        shuffle(ids.begin(), ids.end());
        huts.assign(ids.begin(), ids.begin() + K);
        sort(huts.begin(), huts.end());
    }

    // Emit.
    printf("%d %d %d\n", N, M, K);
    for (auto &e : edges) printf("%d %d %d\n", e[0], e[1], e[2]);
    for (int i = 0; i < K; i++) printf("%d%c", huts[i], i + 1 == K ? '\n' : ' ');
    return 0;
}
