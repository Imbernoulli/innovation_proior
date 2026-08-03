// TIER: trivial
// Fully-serial baseline: build every corridor one at a time on one global timeline,
// in input order. Exactly the construction the checker measures -> ratio ~ 0.1.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n, m;
    scanf("%d %d", &n, &m);
    long long clock = 0;
    for (int j = 0; j < n; j++) {
        int o; long long w;
        scanf("%d %lld", &o, &w);
        vector<long long> starts(o);
        for (int k = 0; k < o; k++) {
            int c; long long d;
            scanf("%d %lld", &c, &d);
            starts[k] = clock;
            clock += d;
        }
        for (int k = 0; k < o; k++)
            printf("%lld%c", starts[k], k + 1 < o ? ' ' : '\n');
    }
    return 0;
}
