// TIER: trivial
// Place a sensor at EVERY tide pool. Always feasible (each pool covers itself),
// achieves exactly the checker's baseline cost -> calibration point ~0.1.
#include <bits/stdc++.h>
using namespace std;
int main() {
    int N, M; long long R;
    if (scanf("%d %d %lld", &N, &M, &R) != 3) return 0;
    printf("%d\n", N);
    for (int v = 1; v <= N; v++) printf("%d\n", v);
    return 0;
}
