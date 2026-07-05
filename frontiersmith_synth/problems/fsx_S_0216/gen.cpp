#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

// Wildlife Corridor Night Watch generator.
// testId is a difficulty/structure ladder: testId 1 is tiny (example scale),
// growing to a large, contended season by testId 10. Determinism via testlib rnd.
int main(int argc, char** argv) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);

    int T = 6 + 9 * (testId - 1);     // 6, 15, 24, ..., 87 nights
    int K = 2 + 4 * (testId - 1);     // 2, 6, 10, ..., 38 segments

    // Volunteer scarcity: enough to matter, never enough to cover everything.
    // Some nights get zero volunteers (forcing contract), higher testIds slightly
    // more contention.
    int volHi = 1 + (testId % 3);     // 1..3 volunteers on "good" nights

    // Mobilization dominates per-guard price -> clustering contract work pays off.
    int Dlo = 12 + 2 * (testId % 4);  // mobilization floor
    int Dhi = Dlo + 10;
    int pLo = 1;
    int pHi = 8 + (testId % 3) * 2;   // 8..12 per-guard ceiling

    vector<int> vol(T + 1), D(T + 1), p(T + 1);
    for (int t = 1; t <= T; t++) {
        // ~30% of nights have no volunteers at all
        if (rnd.next(0, 9) < 3) vol[t] = 0;
        else vol[t] = rnd.next(1, volHi);
        D[t] = rnd.next(Dlo, Dhi);
        p[t] = rnd.next(pLo, pHi);
    }

    // Segments with overlapping windows so contract nights can be shared.
    vector<array<int,3>> seg; // L, R, W
    int maxWin = min(T, 4 + testId);  // window length cap grows with testId
    for (int i = 0; i < K; i++) {
        int len = rnd.next(2, max(2, maxWin));
        int L = rnd.next(1, T - len + 1);
        int R = L + len - 1;
        int W = rnd.next(1, R - L + 1);   // 1 <= W <= window length (always feasible)
        seg.push_back({L, R, W});
    }

    printf("%d %d\n", T, K);
    for (int t = 1; t <= T; t++) printf("%d%c", vol[t], t == T ? '\n' : ' ');
    for (int t = 1; t <= T; t++) printf("%d%c", D[t], t == T ? '\n' : ' ');
    for (int t = 1; t <= T; t++) printf("%d%c", p[t], t == T ? '\n' : ' ');
    for (auto& s : seg) printf("%d %d %d\n", s[0], s[1], s[2]);
    return 0;
}
