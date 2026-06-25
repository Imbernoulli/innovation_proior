#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    if (!(cin >> n)) return 0;
    vector<vector<long long>> c(n, vector<long long>(n));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            cin >> c[i][j];

    const long long INF = (long long)4e18;
    // dp[mask] = minimum cost to have assigned couriers 0..popcount(mask)-1
    // to exactly the set of zones in `mask`.
    int full = (1 << n);
    vector<long long> dp(full, INF);
    dp[0] = 0;
    for (int mask = 0; mask < full; mask++) {
        if (dp[mask] == INF) continue;
        int courier = __builtin_popcount(mask); // next courier to place
        if (courier == n) continue;
        for (int z = 0; z < n; z++) {
            if (mask & (1 << z)) continue;       // zone z already used
            int nmask = mask | (1 << z);
            long long cand = dp[mask] + c[courier][z];
            if (cand < dp[nmask]) dp[nmask] = cand;
        }
    }

    cout << dp[full - 1] << "\n";
    return 0;
}
