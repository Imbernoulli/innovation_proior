// TIER: trivial
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
    for (int i = 0; i < n; i++) printf("0%c", i + 1 == n ? '\n' : ' ');
    if (n == 0) printf("\n");
    return 0;
}
