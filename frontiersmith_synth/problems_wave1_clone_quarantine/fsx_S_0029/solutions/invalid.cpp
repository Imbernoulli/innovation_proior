// TIER: invalid
// Deliberately infeasible: start every segment on day 0. For any corridor with >= 2
// segments this violates precedence (segment 1 would start before segment 0 ends),
// so the checker must score this 0.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n, m;
    scanf("%d %d", &n, &m);
    for (int j = 0; j < n; j++) {
        int o; long long w;
        scanf("%d %lld", &o, &w);
        for (int k = 0; k < o; k++) {
            int c; long long d;
            scanf("%d %lld", &c, &d);
        }
        for (int k = 0; k < o; k++)
            printf("0%c", k + 1 < o ? ' ' : '\n');
    }
    return 0;
}
