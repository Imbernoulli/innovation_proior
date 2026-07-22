// TIER: strong
// Insight: the objective charges for REVERSALS, not distance -- it is a
// monotonicity norm, and its geodesics are serpentine (boustrophedon) sweeps.
// Bucket points into ceil(sqrt(n)) y-ordered row-bands, sweep each row in
// alternating x-direction (so the only reversals happen at O(sqrt n) row
// boundaries instead of on almost every move), and additionally home immediately
// before every zero-tolerance point: homing clears both axes' direction memory,
// so the very next move can never register as a reversal, buying that point for
// one extra trip instead of leaving it permanently uncollectable. This
// co-optimizes tour geometry with the hysteresis state rather than treating the
// two separately (what the naive nearest-neighbor tour does).
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int n;
ll bxp, bxm, byp, bym, C, Lmax;
vector<ll> X, Y, V, TOL;

int main() {
    scanf("%d %lld %lld %lld %lld %lld %lld", &n, &bxp, &bxm, &byp, &bym, &C, &Lmax);
    X.assign(n + 1, 0); Y.assign(n + 1, 0); V.assign(n + 1, 0); TOL.assign(n + 1, 0);
    for (int i = 1; i <= n; i++)
        scanf("%lld %lld %lld %lld", &X[i], &Y[i], &V[i], &TOL[i]);

    vector<int> byY(n);
    for (int i = 0; i < n; i++) byY[i] = i + 1;
    sort(byY.begin(), byY.end(), [&](int a, int b) {
        if (Y[a] != Y[b]) return Y[a] < Y[b];
        return a < b;
    });

    int R = max(1, (int)llround(sqrt((double)n)));
    int base = n / R, extra = n % R, pos = 0;
    vector<vector<int>> rows(R);
    for (int r = 0; r < R; r++) {
        int sz = base + (r < extra ? 1 : 0);
        for (int k = 0; k < sz && pos < n; k++) rows[r].push_back(byY[pos++]);
    }
    for (int r = 0; r < R; r++) {
        if (r % 2 == 0)
            sort(rows[r].begin(), rows[r].end(), [&](int a, int b) {
                if (X[a] != X[b]) return X[a] < X[b];
                return a < b;
            });
        else
            sort(rows[r].begin(), rows[r].end(), [&](int a, int b) {
                if (X[a] != X[b]) return X[a] > X[b];
                return a < b;
            });
    }

    vector<int> ord;
    for (int r = 0; r < R; r++)
        for (int idx : rows[r]) ord.push_back(idx);

    vector<int> route;
    route.reserve(2 * n);
    for (int idx : ord) {
        if (TOL[idx] == 0) route.push_back(0);
        route.push_back(idx);
    }

    printf("%d\n", (int)route.size());
    for (int t : route) printf("%d ", t);
    printf("\n");
    return 0;
}
