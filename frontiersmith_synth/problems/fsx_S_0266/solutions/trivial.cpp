// TIER: trivial
// Build a relay on every system: always feasible, equals baseline cost B -> ratio 0.1.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int N, M, r;
    scanf("%d %d %d", &N, &M, &r);
    for (int i = 0; i < M; i++) { int u, v; scanf("%d %d", &u, &v); }
    for (int i = 0; i < N; i++) { int c; scanf("%d", &c); }
    printf("%d\n", N);
    for (int i = 1; i <= N; i++) printf("%d\n", i);
    return 0;
}
