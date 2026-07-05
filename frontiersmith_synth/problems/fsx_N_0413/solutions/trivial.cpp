// TIER: trivial
// Do-nothing plan: route nobody (K = 0).  This is exactly the checker's baseline
// construction, so it scores the calibration point ratio ~= 0.1.  Always feasible.
#include <bits/stdc++.h>
using namespace std;

int main() {
    // read and ignore the whole input
    int N, M, D; long long P;
    if (scanf("%d %d %d %lld", &N, &M, &D, &P) != 4) return 0;
    for (int e = 0; e < M; e++) { int u, v, c, a; scanf("%d %d %d %d", &u, &v, &c, &a); }
    for (int d = 0; d < D; d++) { int s, t, vol; scanf("%d %d %d", &s, &t, &vol); }
    printf("0\n");
    return 0;
}
