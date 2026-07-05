// TIER: invalid
#include <bits/stdc++.h>
using namespace std;
int main() {
    int n, m;
    if (scanf("%d %d", &n, &m) != 2) return 0;
    for (int j = 0; j < m; j++) {
        int k; scanf("%d", &k);
        for (int e = 0; e < k; e++) { int s, o; scanf("%d %d", &s, &o); }
        long long w; scanf("%lld", &w);
    }
    // Deliberately infeasible: an out-of-range alignment value.
    printf("2\n");
    return 0;
}
