// TIER: invalid
// Deliberately infeasible: every truck backs into door slot 1 (not a permutation).
// The checker must reject this and score 0.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int Tin, Tout, M, K, cap;
    scanf("%d %d %d %d %d", &Tin, &Tout, &M, &K, &cap);
    for (int e = 0; e < M; e++) { int a,b,c; scanf("%d %d %d", &a, &b, &c); }
    int D = Tin + Tout;
    for (int k = 1; k <= D; k++) printf("1%c", k == D ? '\n' : ' ');
    return 0;
}
