// TIER: greedy
// The obvious single-pass recipe: keep the same fixed 16x16 uniform grid layout (the
// space-filling reflex), but for each tile locally pick the highest compression level its
// OWN weakest cell allows. Correct per-tile logic, blind to tile SHAPE and to what sits in
// neighboring tiles or the trace -- it cannot see the cross-region capacity externality.
#include <bits/stdc++.h>
using namespace std;

static const int TS = 16;

int main() {
    int R, C, K;
    if (scanf("%d %d %d", &R, &C, &K) != 3) return 0;
    vector<long long> levelBytes(K);
    for (int k = 0; k < K; k++) scanf("%lld", &levelBytes[k]);
    long long cacheBytes; scanf("%lld", &cacheBytes);

    vector<vector<int>> ent(R, vector<int>(C));
    for (int i = 0; i < R; i++)
        for (int j = 0; j < C; j++)
            scanf("%d", &ent[i][j]);
    // trace is unused by this strategy; no need to read it.

    vector<int> rb, cb;
    for (int i = 0; i <= R; i += TS) rb.push_back(i);
    if (rb.back() != R) rb.push_back(R);
    for (int j = 0; j <= C; j += TS) cb.push_back(j);
    if (cb.back() != C) cb.push_back(C);
    int rB = (int)rb.size() - 1, cB = (int)cb.size() - 1;

    vector<vector<int>> lvl(rB, vector<int>(cB, K - 1));
    for (int r = 0; r < rB; r++) {
        for (int c = 0; c < cB; c++) {
            int cap = K - 1;
            for (int i = rb[r]; i < rb[r + 1]; i++)
                for (int j = cb[c]; j < cb[c + 1]; j++)
                    if (ent[i][j] < cap) cap = ent[i][j];
            lvl[r][c] = cap;
        }
    }

    printf("%d\n", rB);
    for (int x : rb) printf("%d ", x);
    printf("\n%d\n", cB);
    for (int x : cb) printf("%d ", x);
    printf("\n");
    for (int r = 0; r < rB; r++) {
        for (int c = 0; c < cB; c++) printf("%d ", lvl[r][c]);
        printf("\n");
    }
    return 0;
}
