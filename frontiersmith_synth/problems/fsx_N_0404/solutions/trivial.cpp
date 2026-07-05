// TIER: trivial
// Do-nothing one-stage schedule: all acts on stage 1 in input order (== baseline B).
// Scores ratio 0.1 exactly.
#include <bits/stdc++.h>
using namespace std;
int main() {
    int N, M, d;
    if (scanf("%d %d %d", &N, &M, &d) != 3) return 0;
    for (int j = 0; j < N; j++) { int p, x; scanf("%d", &p); for (int k = 0; k < d; k++) scanf("%d", &x); }
    printf("%d", N);
    for (int j = 1; j <= N; j++) printf(" %d", j);
    printf("\n");
    for (int i = 2; i <= M; i++) printf("0\n");
    return 0;
}
