// TIER: trivial
// Baseline: guard every segment with contract only, on the earliest W_i nights of
// its window. Reproduces the judge's internal baseline B exactly -> ratio ~ 0.1.
#include <bits/stdc++.h>
using namespace std;
int main() {
    int T, K;
    if (scanf("%d %d", &T, &K) != 2) return 0;
    vector<int> vol(T + 1), D(T + 1), p(T + 1);
    for (int t = 1; t <= T; t++) scanf("%d", &vol[t]);
    for (int t = 1; t <= T; t++) scanf("%d", &D[t]);
    for (int t = 1; t <= T; t++) scanf("%d", &p[t]);
    for (int i = 1; i <= K; i++) {
        int L, R, W; scanf("%d %d %d", &L, &R, &W);
        printf("%d\n", W);
        for (int j = 0; j < W; j++) printf("%d 1\n", L + j);
    }
    return 0;
}
