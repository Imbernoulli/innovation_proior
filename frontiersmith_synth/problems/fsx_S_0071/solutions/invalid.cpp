// TIER: invalid
// Deliberately infeasible: selects both endpoints of a conflict edge, which are not
// independent -> the checker rejects and the score is 0.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
int main() {
    int n, m;
    if (scanf("%d %d", &n, &m) != 2) return 0;
    for (int i = 1; i <= n; i++) { ll x; scanf("%lld", &x); }
    int u = 1, v = 1;
    if (m >= 1) { scanf("%d %d", &u, &v); }
    printf("2\n%d\n%d\n", u, v);   // two conflicting routes -> not an independent set
    return 0;
}
