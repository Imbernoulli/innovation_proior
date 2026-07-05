#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

// Generalized assignment problem skinned as a wildlife-corridor herd-routing task.
//
// Herds x routes: each (herd, route) pair carries its OWN conservation value v_ij and its
// OWN capacity footprint d_ij, drawn INDEPENDENTLY. Because value and footprint are
// decorrelated, the value-blind first-fit baseline is poor and value-aware / density /
// local-search heuristics diverge and pull ahead.
//
// Structure ladder: testId 1 is tiny (a handful of herds / routes); the instance grows
// quadratically in the herd count up to a large, tightly-contended packing by testId 10.
int main(int argc, char** argv) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);

    int H = min(2000, 20 * testId * testId);   // 20, 80, 180, ..., 2000
    int R = min(60, 6 * testId);               // 6, 12, 18, ..., 60
    if (H < 3) H = 3;
    if (R < 2) R = 2;

    // Capacity scale: total capacity is deliberately smaller than the total footprint that
    // would be needed to route every herd, so the routes are genuinely contended.
    // avg footprint ~ 8; if all H herds routed -> ~8H demand. We provision ~ (4.5/8) of it.
    int base = max(20, (int)(4.5 * (double)H / (double)R));
    base = min(base, 260);                     // keep C_j well within the stated cap of 300

    printf("%d %d\n", H, R);
    for (int j = 0; j < R; j++) {
        int lo = max(1, base * 3 / 5);
        int hi = min(300, base * 7 / 5);
        if (hi < lo) hi = lo;
        int C = (int)rnd.next(lo, hi);
        printf("%d%c", C, j + 1 < R ? ' ' : '\n');
    }

    for (int i = 0; i < H; i++) {
        for (int j = 0; j < R; j++) {
            int v = (int)rnd.next(1, 100);
            int d = (int)rnd.next(1, 15);
            printf("%d %d%c", v, d, j + 1 < R ? ' ' : '\n');
        }
    }
    return 0;
}
