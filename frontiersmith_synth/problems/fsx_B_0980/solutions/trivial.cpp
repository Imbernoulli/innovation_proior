// TIER: trivial
// Naive baseline: under the NATURAL (no-extraction) mixing profile, scan every
// segment, evaluate the electricity a single converter installed only there
// would generate (capped by its own hardware cap and by the Tsink limit), and
// install only at the single best one. This is exactly the checker's internal
// baseline B -> ratio ~= 0.1 by construction.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main() {
    int T, K, Tsink;
    scanf("%d %d %d", &T, &K, &Tsink);
    vector<ll> q(T + 1), theta(T + 1), eta(T + 1), cap(T + 1);
    for (int i = 1; i <= T; i++)
        scanf("%lld %lld %lld %lld", &q[i], &theta[i], &eta[i], &cap[i]);

    vector<double> pre(T + 1), flow(T + 1);
    double F = 0.0, Tcur = 0.0;
    for (int i = 1; i <= T; i++) {
        double Fnew = F + (double)q[i];
        double Tpre = (F == 0.0) ? (double)theta[i] : (F * Tcur + (double)q[i] * theta[i]) / Fnew;
        F = Fnew;
        pre[i] = Tpre; flow[i] = F;
        Tcur = Tpre;
    }

    double best = -1.0;
    int bi = 1;
    ll bx = 0;
    for (int i = 1; i <= T; i++) {
        double e = eta[i] / 1000.0;
        double feasMax = flow[i] * (pre[i] - Tsink);
        double xmax = min((double)cap[i], feasMax);
        if (xmax < 0) xmax = 0;
        double Tpost = pre[i] - xmax / flow[i];
        if (Tpost < Tsink) Tpost = Tsink;
        double v = (Tpost >= pre[i]) ? 0.0 : e * flow[i] * ((pre[i] - Tpost) - Tsink * log(pre[i] / Tpost));
        if (v > best) { best = v; bi = i; bx = (ll)floor(xmax); }
    }
    if (bx < 0) bx = 0;

    printf("1\n%d %lld\n", bi, bx);
    return 0;
}
