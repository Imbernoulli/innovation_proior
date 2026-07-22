#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// -----------------------------------------------------------------------------
// "Wire-bending robot that keeps hitting itself"  (generator)
// family: tail-swing-bend-order
//
// A straight wire of length L has m bend STATIONS at fixed arc-length positions
// x_1 < x_2 < ... < x_m (indices = the wire's own geometric left-to-right order,
// base end at 0, free tip at L). Station i wants a target turning angle theta_i
// (degrees) but the material springs back by a known delta_i after any bend, so
// the machine must choose an APPLIED angle a_i within +/- c of theta_i to
// pre-compensate; the REALIZED angle is a_i + delta_i.
//
// The machine always clamps the base-side structure and swings everything
// strictly beyond the pivot. Bending order is free (any permutation of the m
// stations). The swept "reach" for bending station i = the arc-length distance
// from x_i out to the NEAREST station beyond it that has ALREADY been bent (or
// to the free tip L if none has) -- an unshaped run has no intermediate brace and
// swings as one un-braced arm, whereas an already-shaped station lets the machine
// re-index against it. The chord swept by the farthest point of that arm is
// 2*reach*sin(|realized|/2); if it exceeds the clearance K the bend COLLIDES and
// that station is never matched (contributes 0), leaving the tail unchanged.
//
// PLANTED TRAP: a contiguous low-index zone near the base is stocked with
// large-magnitude, heavy-weight target angles. Bending strictly in geometric
// (path) order forces the very first bends to swing almost the FULL remaining
// wire (nothing beyond has been shaped yet) -- guaranteed collisions on the
// heaviest stations. Bending an interior station first shortens (braces) the
// reach for everything on both sides of it before their turn comes -- the
// balance-point recursion (process a well-chosen interior split first, recurse
// on the two halves) keeps reach small throughout instead of only at the end.
//
// Output: m L c TOL K
//         then m lines (index order i=1..m): x_i theta_i delta_i w_i
// -----------------------------------------------------------------------------

int main(int argc, char* argv[]) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);
    double f = (testId - 1) / 9.0;

    int m;
    double trapfrac;
    bool needle = false;
    if (testId == 1)      { m = 6;   trapfrac = 0.00; }
    else if (testId == 2) { m = 10;  trapfrac = 0.20; }
    else if (testId == 3) { m = 20;  trapfrac = 0.40; }
    else if (testId == 4) { m = 30;  trapfrac = 0.50; }
    else if (testId == 5) { m = 50;  trapfrac = 0.60; }
    else if (testId == 6) { m = 70;  trapfrac = 0.70; needle = true; }
    else if (testId == 7) { m = 90;  trapfrac = 0.80; }
    else if (testId == 8) { m = 110; trapfrac = 0.85; }
    else if (testId == 9) { m = 130; trapfrac = 0.90; }
    else                  { m = 150; trapfrac = 1.00; }

    ll L = 300LL * m;

    // distinct sorted positions in [1, L-1]
    vector<ll> pool;
    pool.reserve(L - 1);
    for (ll p = 1; p <= L - 1; p++) pool.push_back(p);
    for (int i = (int)pool.size() - 1; i > 0; i--) {
        int j = rnd.next(0, i);
        swap(pool[i], pool[j]);
    }
    vector<ll> xs(pool.begin(), pool.begin() + m);
    sort(xs.begin(), xs.end());

    int c = 6 + (testId % 5);          // 6..10
    int TOL = 10 + (testId % 4);       // 10..13
    // clearance: scales with the average gap so the trap is meaningful at every size
    ll K = (ll)llround((16.0 - 4.0 * f) * ((double)L / m)); // tighter as testId grows

    int kzone = max(1, m / 3);         // near-base trap-eligible zone

    vector<ll> theta(m + 1), delta(m + 1), w(m + 1);
    int needleIdx = needle ? (m / 2) : -1;

    for (int i = 1; i <= m; i++) {
        bool istrap = (i <= kzone) && (rnd.next(0.0, 1.0) < trapfrac);
        int sign = rnd.next(0, 1) ? 1 : -1;
        if (istrap) {
            theta[i] = sign * (ll)rnd.next(105, 155);
            w[i] = rnd.next(15, 20);
        } else {
            theta[i] = sign * (ll)rnd.next(20, 90);
            w[i] = rnd.next(1, 10);
        }
        if (i == needleIdx) { w[i] = 30; theta[i] = sign * (ll)rnd.next(110, 150); }
        delta[i] = rnd.next(-25, 25);
        if (delta[i] == 0) delta[i] = 1;
    }

    printf("%d %lld %d %d %lld\n", m, L, c, TOL, K);
    for (int i = 1; i <= m; i++)
        printf("%lld %lld %lld %lld\n", xs[i - 1], theta[i], delta[i], w[i]);
    return 0;
}
