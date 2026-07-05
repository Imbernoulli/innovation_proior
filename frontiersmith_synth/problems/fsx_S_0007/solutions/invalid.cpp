// TIER: invalid
// Deliberately infeasible: build a star centred at station 1. It prints N-1 links (in
// range) but station 1 then has degree N-1, blowing past its relay-head cap (cap_1 <= 4),
// so the checker must reject it with score 0.
#include <bits/stdc++.h>
using namespace std;
int main() {
    int N;
    if (scanf("%d", &N) != 1) return 0;
    for (int i = 0; i < N; i++) { int x, y, c; scanf("%d %d %d", &x, &y, &c); }
    printf("%d\n", N - 1);
    for (int i = 2; i <= N; i++) printf("1 %d\n", i);
    return 0;
}
