// TIER: trivial
// Baseline: On-Demand for the first ceil(W/q) slots as one contiguous block.
// This is exactly the checker's internal baseline B -> ratio == 0.1.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
int main() {
    int T; ll W; int q; ll K;
    if (scanf("%d %lld %d %lld", &T, &W, &q, &K) != 4) return 0;
    for (int t = 1; t <= T; t++) { int a, g, s, d; scanf("%d %d %d %d", &a, &g, &s, &d); }
    ll h = (W + (ll)q - 1) / (ll)q;
    if (h < 1) h = 1;
    for (int t = 1; t <= T; t++) {
        printf("%d", (t <= h) ? 2 : 0);
        putchar(t == T ? '\n' : ' ');
    }
    return 0;
}
