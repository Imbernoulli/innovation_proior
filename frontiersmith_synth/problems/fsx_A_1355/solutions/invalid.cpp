// TIER: invalid
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// Deliberately infeasible: every printed "guess" equals K, which is one past
// the valid range [0, K-1]. The checker's bounded ouf.readInt(0, K-1, ...)
// must reject this on the very first token -> score 0.

int main() {
    int N, K;
    scanf("%d %d", &N, &K);
    vector<ll> w(N + 1);
    for (int i = 1; i <= N; i++) scanf("%lld", &w[i]);

    ll tableSize = 1;
    for (int e = 0; e < N - 1; e++) tableSize *= K;

    for (int i = 1; i <= N; i++)
        for (ll t = 0; t < tableSize; t++)
            printf("%d%c", K, (t + 1 == tableSize) ? '\n' : ' ');
    return 0;
}
