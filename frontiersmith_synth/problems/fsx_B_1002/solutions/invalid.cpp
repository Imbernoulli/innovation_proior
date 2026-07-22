// TIER: invalid
// Deliberately infeasible: pulls order 0 one tick AFTER its own deadline.
// The checker must reject this (and any output of this shape) with score 0.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main() {
    int N, Tmax, H0, FMAX;
    ll TLO, THI, THOT, PH, QH, TCOLD, PC, QC, T0, FLUSH_DROP;
    if (scanf("%d %d %lld %lld %lld %lld %lld %lld %lld %lld %lld %d %lld %d",
               &N, &Tmax, &TLO, &THI, &THOT, &PH, &QH, &TCOLD, &PC, &QC, &T0, &H0, &FLUSH_DROP, &FMAX) != 14) return 0;
    ll a0 = 0, d0 = 0;
    for (int i = 0; i < N; i++) {
        ll a, d, lo, hi, w;
        scanf("%lld %lld %lld %lld %lld", &a, &d, &lo, &hi, &w);
        if (i == 0) { a0 = a; d0 = d; }
    }
    ll t = d0 + 1;   // outside [a0,d0]: rejected either as a window violation, or (if == Tmax) as an out-of-range tick
    (void) a0;
    printf("1\n%lld 0\n", t);
    return 0;
}
