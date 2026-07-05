// TIER: invalid
// Puts every venue in Cohort A -> S_A = T > U, violating the population band.
// Must score 0.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main() {
    int n, m; ll L, U;
    scanf("%d %d %lld %lld", &n, &m, &L, &U);
    vector<ll> p(n + 1);
    for (int i = 1; i <= n; i++) scanf("%lld", &p[i]);
    for (int i = 0; i < m; i++) { int a, b; ll w; scanf("%d %d %lld", &a, &b, &w); (void)a;(void)b;(void)w; }
    for (int i = 1; i <= n; i++) printf("1%c", i == n ? '\n' : ' ');
    return 0;
}
