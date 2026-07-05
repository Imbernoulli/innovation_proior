// TIER: trivial
// Monitor only the single most valuable individual -> F = B -> ratio 0.1.
#include <bits/stdc++.h>
using namespace std;
int main() {
    int n, m;
    scanf("%d %d", &n, &m);
    int best = 1; long long bw = -1;
    for (int i = 1; i <= n; i++) {
        long long x; scanf("%lld", &x);
        if (x > bw) { bw = x; best = i; }
    }
    // edges are irrelevant for a single vertex
    printf("1\n%d\n", best);
    return 0;
}
