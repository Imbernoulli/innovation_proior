// TIER: invalid
// Deliberately infeasible: prints k self-loops "1 1", which are not edges of the
// graph (u==v). The checker must reject this and score 0.
#include <bits/stdc++.h>
using namespace std;
int main() {
    int n, m; long long k;
    scanf("%d %d %lld", &n, &m, &k);
    for (long long j = 0; j < k; j++) printf("1 1\n");
    return 0;
}
