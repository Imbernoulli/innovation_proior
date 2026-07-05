#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

// Adversarial generator for "Abyssal Mesh".
// testId is a difficulty/structure ladder (1 = tiny example scale, 10 = large adversarial).
// Every test contains:
//   - a PLANTED well-connected mesh (a fat-cabled ring + chords) of cheap, moderate-bandwidth
//     hubs: the high-F "needle" that a good plan should build,
//   - TRAP hubs: very high bandwidth but joined only by flimsy cables and moderately expensive,
//     so a coverage-priority greedy is lured into wasting budget for little resilience,
//   - NOISE hubs/cables filling the constraint envelope on the largest tests.
int main(int argc, char** argv) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);

    int N = min(300, 8 + 32 * (testId - 1)); // 8, 40, 72, ..., 296
    if (N < 6) N = 6;

    int pm = min(5 + testId % 4, N - 4);     // planted mesh size (needle, stays small)
    if (pm < 3) pm = 3;
    int tt = min(2 + testId % 3, N - 2 - pm);// trap count
    if (tt < 0) tt = 0;

    vector<int> p(N + 1), c(N + 1);
    // default = noise hubs: low/moderate bandwidth, moderate cost
    for (int i = 1; i <= N; i++) { p[i] = rnd.next(1, 30); c[i] = rnd.next(5, 20); }

    // anchors 1,2: cheap, moderate bandwidth (guarantee an affordable pair)
    p[1] = rnd.next(10, 25); c[1] = rnd.next(2, 4);
    p[2] = rnd.next(10, 25); c[2] = rnd.next(2, 4);

    // planted mesh hubs: cheap, moderate bandwidth
    int pmL = 3, pmR = 2 + pm;
    for (int i = pmL; i <= pmR; i++) { p[i] = rnd.next(20, 45); c[i] = rnd.next(3, 6); }

    // trap hubs: huge bandwidth, moderately expensive, flimsy cables (added below)
    int trL = pmR + 1, trR = pmR + tt;
    for (int i = trL; i <= trR; i++) { p[i] = rnd.next(80, 100); c[i] = rnd.next(14, 22); }

    vector<array<int,3>> E;
    auto add = [&](int u, int v, int w) { if (u != v && u >= 1 && v >= 1 && u <= N && v <= N) E.push_back({u, v, w}); };

    // anchor edge (thin)
    add(1, 2, rnd.next(2, 4));

    // PLANTED mesh: fat-cabled ring
    for (int i = pmL; i <= pmR; i++) { int j = (i == pmR ? pmL : i + 1); add(i, j, rnd.next(14, 20)); }
    // fat chords to raise the min sever cut
    for (int e = 0; e < pm; e++) { int a = rnd.next(pmL, pmR), b = rnd.next(pmL, pmR); add(a, b, rnd.next(12, 20)); }
    // moderate cables anchoring the planted mesh to hubs 1,2
    add(1, rnd.next(pmL, pmR), rnd.next(6, 10));
    add(2, rnd.next(pmL, pmR), rnd.next(6, 10));

    // TRAP cables: flimsy, connecting trap hubs to the rest and to each other
    for (int i = trL; i <= trR; i++) {
        add(i, rnd.next(1, N), rnd.next(1, 2));
        if (trR > trL) add(i, rnd.next(trL, trR), rnd.next(1, 2));
        // a flimsy cable into the planted mesh: tempting but weak
        add(i, rnd.next(pmL, pmR), rnd.next(1, 2));
    }

    // NOISE cables filling the envelope
    int noiseE = min(2600, 3 * N);
    for (int e = 0; e < noiseE; e++) {
        int a = rnd.next(1, N), b = rnd.next(1, N);
        add(a, b, rnd.next(1, 6));
    }

    // budget: affords the whole planted mesh + anchors + a few noise/one trap, but NOT everything.
    int C = 10 + 4 * pm + 4 * testId;

    shuffle(E.begin(), E.end());
    if ((int)E.size() > 3000) E.resize(3000);
    int M = (int)E.size();
    if (M < 1) { add(1, 2, 3); M = 1; }

    printf("%d %d %d\n", N, M, C);
    for (int i = 1; i <= N; i++) printf("%d %d\n", p[i], c[i]);
    for (auto& e : E) printf("%d %d %d\n", e[0], e[1], e[2]);
    return 0;
}
