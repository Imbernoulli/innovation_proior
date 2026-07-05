// TIER: trivial
// Output the backbone path (candidate edges 1..n-1). This is exactly the
// checker's calibrated baseline: an expensive AND fragile chain. Scores ~0.1.
#include <bits/stdc++.h>
using namespace std;
int main() {
    int n, m, D; long long L;
    if (scanf("%d %d %d %lld", &n, &m, &D, &L) != 4) return 0;
    printf("%d\n", n - 1);
    for (int i = 1; i <= n - 1; i++) printf("%d\n", i);
    return 0;
}
