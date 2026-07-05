// TIER: greedy
// Per-variable majority vote, ignoring clause coverage overlap:
// set station i high-power iff the total weight of clauses where it appears as a
// positive literal exceeds the total weight where it appears as a negative literal.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
int main() {
    int n, m;
    if (scanf("%d %d", &n, &m) != 2) return 0;
    vector<ll> posW(n + 1, 0), negW(n + 1, 0);
    for (int c = 0; c < m; c++) {
        ll w; int L;
        scanf("%lld %d", &w, &L);
        for (int j = 0; j < L; j++) {
            int t; scanf("%d", &t);
            int v = abs(t);
            if (t > 0) posW[v] += w; else negW[v] += w;
        }
    }
    for (int i = 1; i <= n; i++) {
        int xi = (posW[i] > negW[i]) ? 1 : 0;
        printf("%d%c", xi, i == n ? '\n' : ' ');
    }
    return 0;
}
