// TIER: invalid
// Deliberately infeasible: declares a table value far outside the allowed [0,63]
// range for the very first pile. The checker must reject this with WA (score 0).
#include <bits/stdc++.h>
using namespace std;

int main() {
    int N;
    scanf("%d", &N);
    for (int i = 0; i < N; i++) {
        int K; scanf("%d", &K);
        for (int j = 0; j < K; j++) { int x; scanf("%d", &x); }
    }
    printf("1 999999\n");
    for (int i = 1; i < N; i++) printf("1 0\n");
    return 0;
}
