#include <bits/stdc++.h>
using namespace std;

int main() {
    long long n, K, MOD;
    if (!(cin >> n >> K >> MOD)) return 0;

    // g[i] = number of valid colored tilings of a 1 x i strip using tiles of
    // length 1 and 2, each tile one of K colors, no two ADJACENT tiles equal.
    // Extending a tiling by one tile: K choices if it is the first tile
    // (predecessor is the empty strip), else K-1 choices (avoid neighbour).
    // g[0] = 1 (empty strip). For i >= 1, a length-1 tile sits on g[i-1],
    // a length-2 tile sits on g[i-2]; the K-vs-(K-1) factor depends only on
    // whether that predecessor strip is empty (length 0) or not.
    vector<long long> g(max<long long>(n + 1, 1));
    g[0] = 1 % MOD;
    long long Km = K % MOD, Km1 = ((K - 1) % MOD + MOD) % MOD;
    for (long long i = 1; i <= n; i++) {
        long long total = 0;
        // add a length-1 tile onto a strip of length i-1
        long long c1 = (i - 1 == 0) ? Km : Km1;     // first tile -> K, else K-1
        total = (total + c1 * g[i - 1]) % MOD;
        // add a length-2 tile onto a strip of length i-2 (needs i >= 2)
        if (i >= 2) {
            long long c2 = (i - 2 == 0) ? Km : Km1;  // first tile -> K, else K-1
            total = (total + c2 * g[i - 2]) % MOD;
        }
        g[i] = total;
    }

    cout << g[n] % MOD << "\n";
    return 0;
}
