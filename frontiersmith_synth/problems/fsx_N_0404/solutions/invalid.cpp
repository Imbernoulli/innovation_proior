// TIER: invalid
// Deliberately infeasible: references an out-of-range act index (N+1). Must score 0.
#include <bits/stdc++.h>
using namespace std;
int main() {
    int N, M, d;
    if (scanf("%d %d %d", &N, &M, &d) != 3) return 0;
    for (int j = 0; j < N; j++) { int p, x; scanf("%d", &p); for (int k = 0; k < d; k++) scanf("%d", &x); }
    printf("1 %d\n", N + 1);          // act N+1 does not exist -> checker rejects
    for (int i = 2; i <= M; i++) printf("0\n");
    return 0;
}
