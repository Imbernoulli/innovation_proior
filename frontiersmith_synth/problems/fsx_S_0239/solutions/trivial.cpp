// TIER: trivial
// Activate only the single highest-value site: F = B, ratio = 0.1.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n, m;
    if (scanf("%d %d", &n, &m) != 2) return 0;
    vector<int> w(n + 1);
    int best = 1;
    for (int i = 1; i <= n; i++) {
        scanf("%d", &w[i]);
        if (w[i] > w[best]) best = i;
    }
    for (int i = 0; i < m; i++) { int u, v; scanf("%d %d", &u, &v); }
    printf("1\n%d\n", best);
    return 0;
}
