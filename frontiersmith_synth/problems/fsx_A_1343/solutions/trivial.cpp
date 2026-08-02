// TIER: trivial
// Do-nothing baseline: claim every pile always contributes invariant value 0 (period
// 1). This predicts every position is a loss for the player to move, and it is also
// the exact construction the checker uses to compute its internal baseline B.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int N;
    scanf("%d", &N);
    for (int i = 0; i < N; i++) {
        int K; scanf("%d", &K);
        for (int j = 0; j < K; j++) { int x; scanf("%d", &x); }
    }
    for (int i = 0; i < N; i++) printf("1 0\n");
    return 0;
}
