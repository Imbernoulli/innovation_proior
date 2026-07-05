#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

// Reef hydrophone channel-assignment generator (family: coloring-labeling, variant #8).
// testId is a difficulty/structure ladder:
//   - testId 1 is tiny (example scale) so the sanity checks are cheap;
//   - it grows to a large, dense, clustered, heavy-tailed instance by testId 10.
// Structure knobs that make greedy / local-search / restart heuristics diverge:
//   * clustered reef "patches" (dense local overlap) + sparse long-range cross-reef pairs,
//   * heavy-tailed interference weights (a few very costly overlaps),
//   * a tight channel budget K relative to cluster density so conflicts are forced.

int main(int argc, char** argv) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);

    int n, K, targetDeg, nClusters;
    if (testId == 1) {
        n = 6; K = 3; targetDeg = 3; nClusters = 2;
    } else {
        n = 120 * testId;                 // 240, 360, ..., 1200
        K = 4 + (testId % 5);             // 4..8  (tight vs. cluster density)
        targetDeg = 8 + (testId % 4) * 2; // 8..14 average degree
        nClusters = 3 + testId;           // more, denser reef patches later
    }

    int maxW = 20;
    int maxD = 3;
    // heavy-tail fraction of very costly overlaps grows with testId
    double heavyProb = 0.05 + 0.01 * (testId % 5);

    // assign each hydrophone to a reef "patch" (cluster)
    vector<int> cluster(n + 1);
    for (int i = 1; i <= n; i++) cluster[i] = rnd.next(0, nClusters - 1);
    vector<vector<int>> members(nClusters);
    for (int i = 1; i <= n; i++) members[cluster[i]].push_back(i);

    // edge set with dedup-free multi-pairs allowed; we just cap total count
    long long targetEdges = (long long)targetDeg * n / 2;
    long long maxEdges = 40000;
    if (targetEdges > maxEdges) targetEdges = maxEdges;

    struct E { int u, v, w, d; };
    vector<E> edges;
    edges.reserve(targetEdges + 8);

    auto sampleWD = [&](int &w, int &d) {
        if (rnd.next(0.0, 1.0) < heavyProb) w = rnd.next(maxW - 4, maxW);
        else                                w = rnd.next(1, 6);
        d = rnd.next(1, maxD);
    };

    // 1) local, intra-cluster overlaps (dense) -- the bulk of interference
    long long localTarget = (long long)(targetEdges * 0.75);
    int guard = 0;
    while ((long long)edges.size() < localTarget && guard < localTarget * 20 + 100) {
        guard++;
        int c = rnd.next(0, nClusters - 1);
        if (members[c].size() < 2) continue;
        int a = members[c][rnd.next(0, (int)members[c].size() - 1)];
        int b = members[c][rnd.next(0, (int)members[c].size() - 1)];
        if (a == b) continue;
        int w, d; sampleWD(w, d);
        edges.push_back({a, b, w, d});
    }

    // 2) sparse long-range cross-reef overlaps (couple distant clusters)
    while ((long long)edges.size() < targetEdges) {
        int a = rnd.next(1, n), b = rnd.next(1, n);
        if (a == b) continue;
        int w, d; sampleWD(w, d);
        edges.push_back({a, b, w, d});
    }

    // guarantee at least one edge so B > 0
    if (edges.empty()) {
        int w, d; sampleWD(w, d);
        edges.push_back({1, (n >= 2 ? 2 : 1), w, d});
        if (n < 2) { edges.back().v = 1; /* degenerate guard, n>=2 by constraints */ }
    }

    // shuffle so input order carries no structural hint
    shuffle(edges.begin(), edges.end());

    int m = (int)edges.size();
    printf("%d %d %d\n", n, m, K);
    for (auto &e : edges)
        printf("%d %d %d %d\n", e.u, e.v, e.w, e.d);
    return 0;
}
