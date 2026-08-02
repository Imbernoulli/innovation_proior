// TIER: greedy
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// Natural "one pass" balancer: scan the players in the GIVEN input order and
// drop each one into whichever of the K residue-groups currently has the
// least total weight (ties -> lowest group index). This DOES use the
// weights (unlike trivial), but it commits every early decision before it
// has seen the later, more informative players -- so when the big/valuable
// players finally show up (always ascending in this problem's structure) the
// light groups are already "spent" on filler and only one group can absorb
// each big item. No re-sorting, no lookahead: just a single greedy pass.

int main() {
    int N, K;
    scanf("%d %d", &N, &K);
    vector<ll> w(N + 1);
    for (int i = 1; i <= N; i++) scanf("%lld", &w[i]);

    vector<ll> bins(K, 0);
    vector<int> r(N + 1);
    for (int i = 1; i <= N; i++) {
        int best = 0;
        for (int b = 1; b < K; b++) if (bins[b] < bins[best]) best = b;
        r[i] = best;
        bins[best] += w[i];
    }

    ll tableSize = 1;
    for (int e = 0; e < N - 1; e++) tableSize *= K;

    for (int i = 1; i <= N; i++) {
        int ri = r[i];
        for (ll t = 0; t < tableSize; t++) {
            ll x = t;
            int sum = 0;
            for (int e = 0; e < N - 1; e++) { sum += (int)(x % K); x /= K; }
            int guess = ((ri - sum) % K + K) % K;
            printf("%d%c", guess, (t + 1 == tableSize) ? '\n' : ' ');
        }
    }
    return 0;
}
