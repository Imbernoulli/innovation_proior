// TIER: greedy
// The obvious "shortest tour" first instinct: repeatedly move to the nearest
// still-unvisited point (plain squared-Euclidean distance), never homing. This
// minimizes literal travel distance but is completely blind to the hysteresis
// mechanic -- it happily reverses an axis's direction on almost every move on a
// generic point cloud, so per-axis error only ever climbs and most (especially
// tight-tolerance) points end up uncollectable despite the short geometric path.
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

    vector<char> used(n + 1, 0);
    vector<int> route;
    route.reserve(n);
    ll cx = 0, cy = 0;
    for (int step = 0; step < n; step++) {
        int best = -1;
        ll bestd = -1;
        for (int i = 1; i <= n; i++) {
            if (used[i]) continue;
            ll dx = X[i] - cx, dy = Y[i] - cy;
            ll d = dx * dx + dy * dy;
            if (best == -1 || d < bestd) { best = i; bestd = d; }
        }
        used[best] = 1;
        route.push_back(best);
        cx = X[best]; cy = Y[best];
    }

    printf("%d\n", (int)route.size());
    for (int t : route) printf("%d ", t);
    printf("\n");
    return 0;
}
