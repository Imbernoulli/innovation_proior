#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

int main(int argc, char** argv) {
    registerGen(argc, argv, 1);
    int tid = atoi(argv[1]);
    if (tid < 1) tid = 1;
    if (tid > 10) tid = 10;

    //             idx: 1    2    3    4    5    6    7    8    9    10
    int Narr[11] = {0,  3,  40,  80, 150, 250, 400, 600, 900, 1400, 2000};
    int Marr[11] = {0,  3,   4,   5,   6,   6,   7,   7,   8,    8,    8};
    int Farr[11] = {0,  2,   3,   3,   4,   3,   4,   4,   5,    4,    5};
    int Parr[11] = {0, 20,  50,  80, 100, 100, 100, 100, 100,  100,  100};
    int Sarr[11] = {0,  8,  15,  20,  25,  30,  20,  25,  30,   20,   25};

    int n = Narr[tid], m = Marr[tid], F = Farr[tid];
    int P = Parr[tid], Smax = Sarr[tid];

    // heavy-tailed processing times on a few tests to vary structure
    bool heavy = (tid == 5 || tid == 8 || tid == 10);
    // clustered families (a job tends to reuse one family) on later tests -> setups matter more
    bool clustered = (tid >= 6);

    println(n, m, F);

    // sequence-dependent (asymmetric) setup matrix, zero diagonal
    for (int a = 0; a < F; a++) {
        vector<int> row(F);
        for (int b = 0; b < F; b++) row[b] = (a == b) ? 0 : rnd.next(0, Smax);
        printf("%d", row[0]);
        for (int b = 1; b < F; b++) printf(" %d", row[b]);
        printf("\n");
    }

    for (int j = 0; j < n; j++) {
        // routing = random permutation of 0..m-1
        vector<int> perm(m);
        for (int k = 0; k < m; k++) perm[k] = k;
        shuffle(perm.begin(), perm.end());

        vector<int> pr(m), fa(m);
        int baseFam = rnd.next(0, F - 1);
        for (int k = 0; k < m; k++) {
            if (heavy) {
                // bimodal: mostly small, occasionally large
                if (rnd.next(0, 4) == 0) pr[k] = rnd.next(P / 2 + 1, P);
                else pr[k] = rnd.next(1, max(1, P / 5));
            } else {
                pr[k] = rnd.next(1, P);
            }
            if (clustered && rnd.next(0, 2) != 0) fa[k] = baseFam;
            else fa[k] = rnd.next(0, F - 1);
        }

        // routing line
        printf("%d", perm[0]);
        for (int k = 1; k < m; k++) printf(" %d", perm[k]);
        printf("\n");
        // processing line
        printf("%d", pr[0]);
        for (int k = 1; k < m; k++) printf(" %d", pr[k]);
        printf("\n");
        // family line
        printf("%d", fa[0]);
        for (int k = 1; k < m; k++) printf(" %d", fa[k]);
        printf("\n");
    }
    return 0;
}
