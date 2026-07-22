// TIER: greedy
// The obvious approach: bend the stations in the wire's own geometric (path)
// order -- station 1, then 2, ..., then m -- since that is literally "the
// order the part is drawn in". Do apply the one clearly-correct fix any
// competent coder adds: pre-compensate springback within the +/-c budget
// (a_i = theta_i - clip(delta_i, -c, c)) so realized angle is as close to
// theta_i as the budget allows. This still swings almost the FULL remaining
// wire on every early bend (nothing beyond has been shaped yet), so it keeps
// colliding on the heavy near-base stations that the generator plants there.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main() {
    int m; ll L; int c, TOL; ll K;
    scanf("%d %lld %d %d %lld", &m, &L, &c, &TOL, &K);
    vector<ll> x(m + 1), theta(m + 1), delta(m + 1), w(m + 1);
    for (int i = 1; i <= m; i++)
        scanf("%lld %lld %lld %lld", &x[i], &theta[i], &delta[i], &w[i]);
    for (int i = 1; i <= m; i++) {
        ll comp = -delta[i];
        if (comp > c) comp = c;
        if (comp < -c) comp = -c;
        printf("%d %lld\n", i, theta[i] + comp);
    }
    return 0;
}
