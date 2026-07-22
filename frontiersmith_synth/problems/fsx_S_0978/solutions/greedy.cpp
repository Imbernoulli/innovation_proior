// TIER: greedy
// The obvious "optimize routing, not geometry" approach: keep the given input-order
// door draw exactly as handed to us (inbound block then outbound block -- never
// touch the door assignment) and just apply the one clearly-good local optimization
// anyone would spot: batch every pair into maximal K-sized trips instead of moving
// pallets one at a time. This cuts trip counts a lot, but the flow field still
// crosses the whole aisle through the same central segments, because the geometry
// (which segments each pair's trips cross) was never reconsidered.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int Tin, Tout, M, K, cap;
    scanf("%d %d %d %d %d", &Tin, &Tout, &M, &K, &cap);
    vector<int> pi_(M), pj_(M), pf_(M);
    for (int e = 0; e < M; e++) scanf("%d %d %d", &pi_[e], &pj_[e], &pf_[e]);

    int D = Tin + Tout;
    for (int k = 1; k <= D; k++) printf("%d%c", k, k == D ? '\n' : ' ');

    for (int e = 0; e < M; e++) {
        int f = pf_[e];
        int m = (f + K - 1) / K;
        printf("%d", m);
        int rem = f;
        for (int t = 0; t < m; t++) {
            int sz = min(K, rem);
            rem -= sz;
            printf(" %d", sz);
        }
        printf("\n");
    }
    return 0;
}
