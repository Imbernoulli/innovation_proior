// TIER: trivial
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// Index round-robin: player i (1-indexed) is assigned target residue
// r_i = (i-1) mod K and guesses "my colour = r_i - (sum of colours I see) mod K".
// This is exactly the checker's internal baseline B (ignores the weights
// entirely -- it never looks at which players are informative).
//
// For any strategy of this "assumed total" form, the digit ORDER of the
// K^(N-1) seen-colour tuple is irrelevant: only the SUM of seen colours
// matters, and the sum is invariant under any relabelling of positions, so we
// may enumerate t = 0..K^(N-1)-1 and use the digit-sum of t in base K
// directly (whatever position each digit "really" represents for the
// checker), and the guess is still correct for every actual assignment.

int main() {
    int N, K;
    scanf("%d %d", &N, &K);
    vector<ll> w(N + 1);
    for (int i = 1; i <= N; i++) scanf("%lld", &w[i]);

    ll tableSize = 1;
    for (int e = 0; e < N - 1; e++) tableSize *= K;

    for (int i = 1; i <= N; i++) {
        int r = (i - 1) % K;
        for (ll t = 0; t < tableSize; t++) {
            ll x = t;
            int sum = 0;
            for (int e = 0; e < N - 1; e++) { sum += (int)(x % K); x /= K; }
            int guess = ((r - sum) % K + K) % K;
            printf("%d%c", guess, (t + 1 == tableSize) ? '\n' : ' ');
        }
    }
    return 0;
}
