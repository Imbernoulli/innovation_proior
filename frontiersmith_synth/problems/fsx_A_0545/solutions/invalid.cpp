// TIER: invalid
// Deliberately infeasible: routes flow on a single pipe, which cannot satisfy
// material balance (its endpoints end up unbalanced) -> must score 0.
#include <bits/stdc++.h>
using namespace std;
int main() {
    int N, M;
    if (scanf("%d %d", &N, &M) != 2) return 0;
    // consume input (not required, but tidy)
    printf("1\n");
    printf("1 1\n");   // pipe 1, flow 1: breaks inflow==outflow at its endpoints
    return 0;
}
