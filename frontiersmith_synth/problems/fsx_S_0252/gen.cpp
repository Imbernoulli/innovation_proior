#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main(int argc, char** argv) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);

    // ---- structure ladder: cold-chain hub over a recorded horizon ----
    // testId 1 is tiny (example scale); grows to many rooms / long horizon.
    int N, T;
    if (testId == 1) { N = 2; T = 6; }
    else { N = min(8, testId + 1); T = 100 * testId; }

    int Lmax = 20;
    int P = 100;
    int S = 200;

    // per-room heat load traces
    vector<vector<int>> L(N, vector<int>(T));
    for (int i = 0; i < N; i++) {
        // some rooms run hotter than others -> uneven reserves, richer allocation
        int hi = 8 + (i % 4) * 4;           // 8,12,16,20 ...
        hi = min(hi, Lmax);
        for (int t = 0; t < T; t++) L[i][t] = rnd.next(1, hi);
    }

    // plant throughput: comfortably above the peak instantaneous demand so the
    // baseline is feasible AND there is headroom to pre-store during cheap windows.
    ll maxSum = 0;
    for (int t = 0; t < T; t++) {
        ll s = 0; for (int i = 0; i < N; i++) s += L[i][t];
        maxSum = max(maxSum, s);
    }
    ll Q = (ll)ceil(1.4 * (double)maxSum);
    if (Q < maxSum) Q = maxSum;

    // room capacities / initial reserves (start full: buffer to coast)
    int capMult = 8 + testId % 6;           // 8..13 steps of storage
    vector<int> Cap(N), I0(N);
    for (int i = 0; i < N; i++) { Cap[i] = Lmax * capMult; I0[i] = Cap[i]; }

    // solar availability pattern: bursty (blocks) on even tests, i.i.d. on odd.
    vector<int> solar(T, 0);
    bool bursty = (testId % 2 == 0);
    double sp = 0.30 + 0.03 * testId;       // fraction of steps with spot power
    if (sp > 0.7) sp = 0.7;
    if (bursty) {
        int t = 0;
        while (t < T) {
            int len = rnd.next(2, 8);
            int on = (rnd.next(0.0, 1.0) < sp) ? 1 : 0;
            for (int k = 0; k < len && t < T; k++, t++) solar[t] = on;
        }
    } else {
        for (int t = 0; t < T; t++) solar[t] = (rnd.next(0.0, 1.0) < sp) ? 1 : 0;
    }

    // spot prices (cheap vs grid); range widens with testId for score spread.
    int pLo = 3, pHi = 20 + 2 * testId;
    if (pHi >= P) pHi = P - 1;
    vector<int> pr(T);
    for (int t = 0; t < T; t++) pr[t] = rnd.next(pLo, pHi);

    // ---- emit ----
    printf("%d %d %lld %d %d\n", N, T, Q, S, P);
    for (int t = 0; t < T; t++) printf("%d%c", solar[t], t + 1 < T ? ' ' : '\n');
    for (int t = 0; t < T; t++) printf("%d%c", pr[t], t + 1 < T ? ' ' : '\n');
    for (int i = 0; i < N; i++) {
        printf("%d %d\n", Cap[i], I0[i]);
        for (int t = 0; t < T; t++) printf("%d%c", L[i][t], t + 1 < T ? ' ' : '\n');
    }
    return 0;
}
