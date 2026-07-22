// TIER: trivial
// Naive baseline: chop the tapestry into a fixed 16x16 grid of tiles and never compress
// anything (level 0 everywhere). Ignores the entropy map and the access trace entirely.
// This is EXACTLY the internal baseline B the checker computes, so it scores ratio ~= 0.1.
#include <bits/stdc++.h>
using namespace std;

static const int TS = 16;

int main() {
    int R, C, K;
    if (scanf("%d %d %d", &R, &C, &K) != 3) return 0;
    // we don't need anything else from the input for this construction.

    vector<int> rb, cb;
    for (int i = 0; i <= R; i += TS) rb.push_back(i);
    if (rb.back() != R) rb.push_back(R);
    for (int j = 0; j <= C; j += TS) cb.push_back(j);
    if (cb.back() != C) cb.push_back(C);

    int rB = (int)rb.size() - 1, cB = (int)cb.size() - 1;

    printf("%d\n", rB);
    for (int x : rb) printf("%d ", x);
    printf("\n%d\n", cB);
    for (int x : cb) printf("%d ", x);
    printf("\n");
    for (int r = 0; r < rB; r++) {
        for (int c = 0; c < cB; c++) printf("0 ");
        printf("\n");
    }
    return 0;
}
