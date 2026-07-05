// TIER: invalid
// Deliberately infeasible: references an out-of-range edge index (and is not
// spanning). The checker must reject this and score 0.
#include <bits/stdc++.h>
using namespace std;
int main() {
    int n, m, D; long long L;
    if (scanf("%d %d %d %lld", &n, &m, &D, &L) != 4) return 0;
    printf("1\n");
    printf("%d\n", m + 1000000);   // out of [1,m]
    return 0;
}
