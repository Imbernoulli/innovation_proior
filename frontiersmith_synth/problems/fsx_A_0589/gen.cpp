#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

// ---------------------------------------------------------------------------
// Generator for "Polyphonic Facade Bands"  (family: frieze-band-polyphony)
//
// testId is a ladder: tiny at 1, large/adversarial at 10.
//   - B  bands 5..9, palette P 5..7, max period Q 4..12, height h=3.
//   - Harmony table M: diagonal held at D=7 (so the monochrome baseline is
//     substantial); off-diagonals a low background in [2,6] EXCEPT a few
//     PLANTED high-harmony pairs (value HI=11). These scarce high pairs are the
//     trap: a diversity-blind solver just spams one high pair with one signature
//     (Groups=1); the payoff hides in weaving many signatures across a
//     divisibility-ordered stack while still landing the scarce high pairs on
//     compatible seams.
//   - Larger tests widen Q (introducing periods 3,4,6,8,.. whose non-divisible
//     pairs make seam ORDER matter) and make the high pairs scarcer.
// ---------------------------------------------------------------------------

int main(int argc, char* argv[]) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);
    double f = (testId - 1) / 9.0;

    int h = 3;
    int B = 5 + (int)round(4.0 * f);         // 5..9
    int P = 5 + (int)round(2.0 * f);         // 5..7
    int Q = 4 + (int)round(8.0 * f);         // 4..12
    if (Q < 2) Q = 2;
    int D = 7;                                // diagonal harmony (baseline anchor)
    int HI = 11;                              // planted high-harmony value
    // number of planted high pairs: fewer (scarcer) on harder tests
    int nPairs = max(1, (int)round((2.0 - 1.0 * f) * (P / 2.0)));

    // symmetric harmony table
    vector<vector<int>> M(P, vector<int>(P));
    for (int i = 0; i < P; i++)
        for (int j = i; j < P; j++) {
            int v = (i == j) ? D : rnd.next(2, 6);
            M[i][j] = v; M[j][i] = v;
        }
    // plant scarce high-harmony off-diagonal pairs
    int placed = 0, guard = 0;
    while (placed < nPairs && guard++ < 500) {
        int i = rnd.next(0, P - 1), j = rnd.next(0, P - 1);
        if (i == j) continue;
        if (M[i][j] == HI) continue;
        M[i][j] = HI; M[j][i] = HI;
        placed++;
    }

    printf("%d %d %d %d\n", B, h, P, Q);
    for (int i = 0; i < P; i++) {
        for (int j = 0; j < P; j++) printf("%d%c", M[i][j], j + 1 < P ? ' ' : '\n');
    }
    return 0;
}
