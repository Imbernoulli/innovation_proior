// TIER: trivial
// Do-nothing baseline: link the stations in input order into a single lift chain.
// Always feasible (every cap >= 2). This is exactly the checker's baseline B, so it
// scores ratio ~= 0.1.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int N;
    if (scanf("%d", &N) != 1) return 0;
    for (int i = 0; i < N; i++) {
        int x, y, h, c;
        scanf("%d %d %d %d", &x, &y, &h, &c);
    }
    if (N <= 1) { printf("0\n"); return 0; }
    printf("%d\n", N - 1);
    for (int i = 1; i < N; i++) printf("%d %d\n", i, i + 1);
    return 0;
}
