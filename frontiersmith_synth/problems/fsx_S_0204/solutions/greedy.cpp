// TIER: greedy
// One-pass prefix fill: from slot 1, while demand not yet met, run the cheaper-per-sack
// mode available that slot (Spot if available and cheaper per sack, else On-Demand),
// stopping once W sacks are milled. Contiguous from the start -> one run (cheap startup),
// and uses cheap spot power where present, so it beats the on-demand-only baseline.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
int main() {
    int T; ll W; int q; ll K;
    if (scanf("%d %lld %d %lld", &T, &W, &q, &K) != 4) return 0;
    vector<int> avail(T + 1), g(T + 1), sc(T + 1), dc(T + 1);
    for (int t = 1; t <= T; t++)
        scanf("%d %d %d %d", &avail[t], &g[t], &sc[t], &dc[t]);

    vector<int> a(T + 1, 0);
    ll sacks = 0;
    for (int t = 1; t <= T && sacks < W; t++) {
        // per-sack cost of each usable option
        double odps = (double)dc[t] / (double)q;
        if (avail[t]) {
            double spps = (double)sc[t] / (double)g[t];
            if (spps <= odps) { a[t] = 1; sacks += g[t]; }
            else { a[t] = 2; sacks += q; }
        } else {
            a[t] = 2; sacks += q;
        }
    }
    // (guaranteed feasible because q*T >= W)
    for (int t = 1; t <= T; t++) {
        printf("%d", a[t]);
        putchar(t == T ? '\n' : ' ');
    }
    return 0;
}
