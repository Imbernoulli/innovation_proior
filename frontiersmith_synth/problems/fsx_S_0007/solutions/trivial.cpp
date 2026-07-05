// TIER: trivial
// Chain baseline: link stations in input order 1-2-...-N. Exactly the network the
// checker measures as B, so it scores ratio ~0.1. Every cap_i >= 2 => feasible.
#include <bits/stdc++.h>
using namespace std;
int main() {
    int N;
    if (scanf("%d", &N) != 1) return 0;
    for (int i = 0; i < N; i++) { int x, y, c; scanf("%d %d %d", &x, &y, &c); }
    printf("%d\n", N - 1);
    for (int i = 1; i < N; i++) printf("%d %d\n", i, i + 1);
    return 0;
}
