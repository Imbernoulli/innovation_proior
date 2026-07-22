// TIER: trivial
// Naive baseline: cable elements together in fixed-size runs of 6 -- each
// new element measures against the PREVIOUS element, except every 6th
// element which resets against the reference. Ignores geometry and drift
// entirely. Reproduces exactly the checker's internal baseline B. Any
// leftover time budget re-touches (0,1) -- harmless, since node 1 was
// connected earliest (t=1) and by the time leftover steps run its clock is
// already far more stale than it was at connection time, so the checker's
// min-rule never improves err_1 from this (it can only ignore a worse
// candidate).
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main() {
    int N; ll T;
    scanf("%d %lld", &N, &T);
    for (int i = 0; i < N; i++) { ll x, y, d; scanf("%lld %lld %lld", &x, &y, &d); }

    const int GROUP_SIZE = 6;
    for (int i = 1; i < N; i++) {
        int p = ((i - 1) % GROUP_SIZE == 0) ? 0 : (i - 1);
        printf("%d %d\n", p, i);
    }
    for (ll t = N; t <= T; t++) printf("0 1\n");
    return 0;
}
