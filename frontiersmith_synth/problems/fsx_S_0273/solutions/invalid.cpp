// TIER: invalid
// Deliberately infeasible: emits channel K+1 (out of range [1,K]) for the first station.
// The checker's bounded read rejects it -> score 0.
#include <bits/stdc++.h>
using namespace std;
int main() {
    int N, M, K;
    if (scanf("%d %d %d", &N, &M, &K) != 3) return 0;
    for (int i = 0; i < M; i++) { int a, b, g, w; if (scanf("%d %d %d %d", &a, &b, &g, &w) != 4) break; }
    printf("%d\n", K + 1);                 // out of range -> infeasible
    for (int j = 2; j <= N; j++) printf("1\n");
    return 0;
}
