// TIER: invalid
// Deliberately infeasible: emits one path flow whose "path" repeats station 1 twice, which
// is never a real track segment, so the checker rejects it and the output scores 0.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int N, M, D; long long P;
    if (scanf("%d %d %d %lld", &N, &M, &D, &P) != 4) return 0;
    for (int e = 0; e < M; e++) { int u, v, c, a; scanf("%d %d %d %d", &u, &v, &c, &a); }
    for (int d = 0; d < D; d++) { int s, t, vol; scanf("%d %d %d", &s, &t, &vol); }
    // K = 1 ; demand 1 ; L = 2 ; path "1 1" (no such segment) ; f = 1
    printf("1\n");
    printf("1 2 1 1 1\n");
    return 0;
}
