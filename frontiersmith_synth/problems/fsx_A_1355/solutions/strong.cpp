// TIER: strong
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// The insight: SORT players by weight DESCENDING first, then greedily drop
// each into the currently-lightest residue group (classic Longest-Processing-
// Time / exchange-argument load balancing, repurposed as a pre-agreed
// covering code). Placing the most informative (highest-weight) guessers
// while every group is still empty means the biggest commitments are made
// with full freedom, and the leftover filler weight can always top off
// whichever group is behind -- this is provably far more balanced than
// processing players in whatever order they happen to be listed. The
// resulting worst-case guarantee (the minimum group weight) is what the
// checker's brute force will discover no matter how it is verified.

int main() {
    int N, K;
    scanf("%d %d", &N, &K);
    vector<ll> w(N + 1);
    for (int i = 1; i <= N; i++) scanf("%lld", &w[i]);

    vector<int> order(N);
    for (int i = 0; i < N; i++) order[i] = i + 1;
    sort(order.begin(), order.end(), [&](int a, int b) {
        if (w[a] != w[b]) return w[a] > w[b];
        return a < b;
    });

    vector<ll> bins(K, 0);
    vector<int> r(N + 1);
    for (int idx : order) {
        int best = 0;
        for (int b = 1; b < K; b++) if (bins[b] < bins[best]) best = b;
        r[idx] = best;
        bins[best] += w[idx];
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
