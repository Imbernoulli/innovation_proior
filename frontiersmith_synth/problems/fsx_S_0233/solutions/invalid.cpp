// TIER: invalid
// Deliberately infeasible: start every step at time 0. Any telescope with >= 2 steps has
// its precedence violated (step 1 would start before step 0 finishes), so the checker must
// reject this with score 0.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n, m;
    if (scanf("%d %d", &n, &m) != 2) return 0;
    for (int j = 0; j < n; j++) {
        long long r; int o;
        scanf("%lld %d", &r, &o);
        for (int k = 0; k < o; k++) {
            int mac; long long d;
            scanf("%d %lld", &mac, &d);
        }
        for (int k = 0; k < o; k++)
            printf("0%c", k + 1 < o ? ' ' : '\n');
    }
    return 0;
}
