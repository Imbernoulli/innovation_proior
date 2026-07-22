// TIER: invalid
// Deliberately infeasible: emit task id 0 (out of range) -> checker rejects -> score 0.
#include <bits/stdc++.h>
using namespace std;
int main() {
    int N, M, C, E;
    if (scanf("%d %d %d %d", &N, &M, &C, &E) != 4) return 0;
    for (int i = 0; i < N; i++) printf("0 ");
    printf("\n");
    return 0;
}
