// TIER: invalid
// Deliberately infeasible: clears nothing and cuts nothing, so the whole tree survives as
// ONE block. Since K>=2 always, this is always exactly one component short of the required
// K -- a clean, well-formed-but-infeasible output that the checker must reject with score 0.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n, K, LAMBDA;
    scanf("%d %d %d", &n, &K, &LAMBDA);
    for (int i = 0; i < n; i++) { int t; scanf("%d", &t); }
    for (int i = 0; i < n; i++) { int t; scanf("%d", &t); }
    for (int i = 0; i < n - 1; i++) { int a, b, c; scanf("%d %d %d", &a, &b, &c); }

    printf("0\n0\n\n");
    return 0;
}
