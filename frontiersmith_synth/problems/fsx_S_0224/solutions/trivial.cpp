// TIER: trivial
// Reproduces the checker's balance-only baseline exactly -> ratio == 0.1.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main() {
    int n, m; ll L, U;
    scanf("%d %d %lld %lld", &n, &m, &L, &U);
    vector<ll> p(n + 1);
    for (int i = 1; i <= n; i++) scanf("%lld", &p[i]);
    // edges are irrelevant to the baseline construction; skip them.
    for (int i = 0; i < m; i++) { int a, b; ll w; scanf("%d %d %lld", &a, &b, &w); (void)a;(void)b;(void)w; }

    vector<int> c(n + 1, 0);
    ll sA = 0, sB = 0;
    for (int i = 1; i <= n; i++) {
        if (sA < sB) { c[i] = 1; sA += p[i]; }
        else         { c[i] = 0; sB += p[i]; }
    }
    for (int i = 1; i <= n; i++) printf("%d%c", c[i], i == n ? '\n' : ' ');
    return 0;
}
