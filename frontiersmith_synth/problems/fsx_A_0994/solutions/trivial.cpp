// TIER: trivial
// Naive baseline: build mirrors at the first min(M,P) candidate sites in the
// order they appear in the input -- unrelated to quality or geometry. This
// reproduces the checker's internal baseline B exactly.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int P, M, K;
    if (scanf("%d %d %d", &P, &M, &K) != 3) return 0;
    for (int i = 0; i < P; i++) { long long x, y, q; scanf("%lld %lld %lld", &x, &y, &q); }
    for (int k = 0; k < K; k++) { long long dx, dy, e; scanf("%lld %lld %lld", &dx, &dy, &e); }
    int m = min(M, P);
    printf("%d\n", m);
    for (int i = 1; i <= m; i++) printf("%d ", i);
    printf("\n");
    return 0;
}
