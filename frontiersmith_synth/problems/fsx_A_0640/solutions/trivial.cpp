// TIER: trivial
// Baseline construction: bend the stations in plain geometric (path) order
// 1,2,...,m and never bother pre-compensating springback (apply exactly the
// target angle). This reproduces the checker's internal baseline B -> ~0.1.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main() {
    int m; ll L; int c, TOL; ll K;
    scanf("%d %lld %d %d %lld", &m, &L, &c, &TOL, &K);
    vector<ll> x(m + 1), theta(m + 1), delta(m + 1), w(m + 1);
    for (int i = 1; i <= m; i++)
        scanf("%lld %lld %lld %lld", &x[i], &theta[i], &delta[i], &w[i]);
    for (int i = 1; i <= m; i++)
        printf("%d %lld\n", i, theta[i]);
    return 0;
}
