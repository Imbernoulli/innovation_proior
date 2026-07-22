#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// -----------------------------------------------------------------------------
// Checker / scorer for "Worn-Gear Gantry: The Backlash-Aware Pick Circuit"
// family: hysteresis-reversal-tour
//
// Input:  n bxp bxm byp bym C Lmax   then n lines "x y v tol".
// Output: m   then m tokens in [0,n] (0 = home, i = visit point i).
//
// Simulation: from (0,0), no established direction, zero error on both axes.
// Visiting a point moves straight there (cost C*Chebyshev distance); for each
// axis whose coordinate changes, if that axis had already committed to the
// OPPOSITE sign since the last home, its error increases by the axis+direction
// specific penalty (bxp/bxm/byp/bym); the axis's direction memory updates
// regardless. A point is credited its value (once, first time only) iff BOTH
// axes' error are within its tolerance at the moment of arrival. "Home" (token 0)
// costs the same travel formula to (0,0) and then resets BOTH axes' error and
// direction memory to zero.
//
// Objective (MAX): F = value collected - total travel cost, floored at 0.
// Baseline B (checker-computed): fetch the TOPK=6 highest-value points, each via
// its own home->visit round trip -- home resets error first, so every one of them
// is always collected regardless of tolerance. A real, feasible, always-safe but
// unambitious construction ("grab the biggest prizes one at a time, never risk a
// reversal"). B is its F (at least 1). Score: sc = min(1000, 100*F/max(1,B));
// ratio = sc/1000.
// -----------------------------------------------------------------------------

static const int TOPK = 6;
struct Pt { ll x, y, v, tol; };

static ll simulate(const vector<int>& route, const vector<Pt>& pts, int n,
                    ll bxp, ll bxm, ll byp, ll bym, ll C) {
    ll px = 0, py = 0;
    int sx = 0, sy = 0;
    ll ex = 0, ey = 0;
    vector<char> collected(n + 1, 0);
    ll totalValue = 0, totalCost = 0;
    for (int t : route) {
        ll tx = (t == 0) ? 0 : pts[t].x;
        ll ty = (t == 0) ? 0 : pts[t].y;
        ll dx = tx - px, dy = ty - py;
        totalCost += C * max(llabs(dx), llabs(dy));
        if (t == 0) {
            px = 0; py = 0; sx = 0; sy = 0; ex = 0; ey = 0;
            continue;
        }
        if (dx != 0) {
            int nd = dx > 0 ? 1 : -1;
            if (sx != 0 && nd != sx) ex += (nd == 1 ? bxp : bxm);
            sx = nd;
        }
        if (dy != 0) {
            int nd = dy > 0 ? 1 : -1;
            if (sy != 0 && nd != sy) ey += (nd == 1 ? byp : bym);
            sy = nd;
        }
        px = tx; py = ty;
        if (!collected[t] && ex <= pts[t].tol && ey <= pts[t].tol) {
            collected[t] = 1;
            totalValue += pts[t].v;
        }
    }
    ll F = totalValue - totalCost;
    return F < 0 ? 0 : F;
}

int main(int argc, char* argv[]) {
    registerTestlibCmd(argc, argv);

    int n = inf.readInt(1, 100000, "n");
    ll bxp = inf.readLong(0, (ll)1e9, "bxp");
    ll bxm = inf.readLong(0, (ll)1e9, "bxm");
    ll byp = inf.readLong(0, (ll)1e9, "byp");
    ll bym = inf.readLong(0, (ll)1e9, "bym");
    ll C = inf.readLong(0, (ll)1e9, "C");
    ll Lmax = inf.readLong(0, (ll)1e9, "Lmax");

    vector<Pt> pts(n + 1);
    for (int i = 1; i <= n; i++) {
        ll x = inf.readLong((ll)-1e9, (ll)1e9, "x");
        ll y = inf.readLong((ll)-1e9, (ll)1e9, "y");
        ll v = inf.readLong(1, (ll)1e9, "v");
        ll tol = inf.readLong(0, (ll)1e9, "tol");
        pts[i] = {x, y, v, tol};
    }

    // ---- internal baseline: top-TOPK-by-value, each fetched via its own home ----
    int K = min(n, TOPK);
    vector<int> idx(n);
    for (int i = 0; i < n; i++) idx[i] = i + 1;
    sort(idx.begin(), idx.end(), [&](int a, int b) {
        if (pts[a].v != pts[b].v) return pts[a].v > pts[b].v;
        return a < b;
    });
    vector<int> baseRoute;
    baseRoute.reserve(2 * K);
    for (int k = 0; k < K; k++) { baseRoute.push_back(0); baseRoute.push_back(idx[k]); }
    ll B = simulate(baseRoute, pts, n, bxp, bxm, byp, bym, C);
    if (B <= 0) B = 1;

    // ---- read participant route ----
    if (Lmax < 0) Lmax = 0;
    int m = ouf.readInt(0, (int)min(Lmax, (ll)2000000000LL), "m");
    vector<int> route(m);
    for (int j = 0; j < m; j++) route[j] = ouf.readInt(0, n, "token");
    if (!ouf.seekEof()) quitf(_wa, "trailing output tokens");

    ll F = simulate(route, pts, n, bxp, bxm, byp, bym, C);

    double sc = min(1000.0, 100.0 * (double)F / (double)max(1LL, B));
    quitp(sc / 1000.0, "OK F=%lld B=%lld m=%d/%lld Ratio: %.6f", F, B, m, Lmax, sc / 1000.0);
    return 0;
}
