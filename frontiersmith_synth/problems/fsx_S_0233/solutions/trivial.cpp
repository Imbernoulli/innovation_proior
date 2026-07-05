// TIER: trivial
// Fully-serial schedule: every step back-to-back on one global timeline in telescope
// order, each telescope starting no earlier than its ready time. This is exactly the
// baseline B the checker measures, so it scores ratio == 0.1.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n, m;
    if (scanf("%d %d", &n, &m) != 2) return 0;
    long long cur = 0;
    for (int j = 0; j < n; j++) {
        long long r; int o;
        scanf("%lld %d", &r, &o);
        long long t = max(cur, r);
        vector<long long> st(o);
        for (int k = 0; k < o; k++) {
            int mac; long long d;
            scanf("%d %lld", &mac, &d);
            st[k] = t;
            t += d;
        }
        cur = t;
        for (int k = 0; k < o; k++)
            printf("%lld%c", st[k], k + 1 < o ? ' ' : '\n');
    }
    return 0;
}
