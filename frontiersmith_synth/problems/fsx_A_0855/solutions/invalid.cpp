// TIER: invalid
// Deliberately infeasible: repeats job index 1 for every step instead of
// printing a permutation. The checker must reject this with score 0.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int R, C, N, T;
    scanf("%d %d %d %d", &R, &C, &N, &T);
    for (int r = 0; r < R; r++) { char buf[64]; scanf("%s", buf); }
    for (int i = 0; i < N; i++) { int a, b, c; scanf("%d %d %d", &a, &b, &c); }
    for (int i = 0; i < N; i++) printf("1\n");
    return 0;
}
