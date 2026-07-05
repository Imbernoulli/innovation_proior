// TIER: invalid
// Deliberately infeasible: first station gets channel C+1 (out of range) -> checker rejects -> score 0.
#include <bits/stdc++.h>
using namespace std;
int main() {
    int N, M, C;
    if (scanf("%d %d %d", &N, &M, &C) != 3) return 0;
    for (int i = 0; i < M; i++) { int u, v, w; scanf("%d %d %d", &u, &v, &w); }
    printf("%d\n", C + 1);
    for (int i = 1; i < N; i++) printf("1\n");
    return 0;
}
