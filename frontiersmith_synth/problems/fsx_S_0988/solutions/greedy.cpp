// TIER: greedy
// The "obvious" move: read the castle's own true value table V and announce
// it verbatim as the scoring rule (fit V exactly into the 2-piece family,
// WP=1, Pcap=Pmax). This NEVER looks at any guild's printed cost table, so
// it cannot see which guild is cheap in which quality range -- exactly the
// announced-vs-true-preference trap.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main() {
    int Q, n; ll Pmax;
    if (scanf("%d %d %lld", &Q, &n, &Pmax) != 3) return 0;
    vector<ll> V(Q + 1);
    for (int q = 0; q <= Q; q++) scanf("%lld", &V[q]);
    for (int j = 0; j < n; j++)
        for (int q = 0; q <= Q; q++) { ll x; scanf("%lld", &x); }

    // Detect the kink: the largest q0 such that V is still rising at its
    // initial (steepest) per-unit slope.
    ll slope1 = (Q >= 1) ? (V[1] - V[0]) : 0;
    if (slope1 <= 0) slope1 = 1;
    int q0 = Q;
    for (int q = 1; q <= Q; q++) {
        ll step = V[q] - V[q - 1];
        if (step < slope1) { q0 = q - 1; break; }
    }
    if (q0 < 0) q0 = 0;

    ll slope2 = 0;
    if (Q > q0) slope2 = (V[Q] - V[q0]) / (Q - q0);

    ll WQ = min(300LL, max(1LL, slope1));
    ll WP = 1;
    // slopeHi in announced units ~= round(slope2 / slope1), clamped to [0,8].
    ll slopeHi = 0;
    if (slope1 > 0) slopeHi = (ll)llround((double)slope2 / (double)slope1);
    slopeHi = max(0LL, min(8LL, slopeHi));

    printf("%lld %lld %d %lld %lld\n", WQ, WP, q0, slopeHi, Pmax);
    return 0;
}
