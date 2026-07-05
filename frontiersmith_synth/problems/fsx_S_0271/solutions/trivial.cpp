// TIER: trivial
// Loop baseline: connect modules 1-2-...-N-1 in input order into a single Hamiltonian
// cycle. Always 2-edge-connected (a cycle has no bridge) and every module has degree 2,
// within any cap>=2. This is exactly the checker's baseline, so it scores ratio ~0.1.
#include <bits/stdc++.h>
using namespace std;
int main() {
    int N; if (scanf("%d", &N) != 1) return 0;
    for (int i = 0; i < N; i++) { long long x, y, c; scanf("%lld %lld %lld", &x, &y, &c); }
    printf("%d\n", N);
    for (int i = 1; i < N; i++) printf("%d %d\n", i, i + 1);
    printf("%d %d\n", N, 1);
    return 0;
}
