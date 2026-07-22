// TIER: invalid
#include <bits/stdc++.h>
using namespace std;

// Deliberately infeasible: claims one more matched pair than the matching size
// bound (min(n,m)) allows. The checker must reject this via its bounded read of
// k, scoring 0 on every test regardless of the market's edge data.
int main() {
    int n, m, e; long long lambda;
    scanf("%d %d %d %lld", &n, &m, &e, &lambda);
    int cap = min(n, m);
    printf("%d\n", cap + 5);
    for (int t = 0; t < cap + 5; t++) printf("1 1 0\n");
    return 0;
}
