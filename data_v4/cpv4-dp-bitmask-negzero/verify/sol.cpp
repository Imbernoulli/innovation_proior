#include <bits/stdc++.h>
using namespace std;

int main() {
    int n, m;
    if (!(cin >> n >> m)) return 0;        // missing header -> nothing to do
    vector<vector<long long>> p(n, vector<long long>(m));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < m; j++) cin >> p[i][j];

    // dp[mask] = best total profit after deciding the first i parcels,
    // where mask is the set of slots already occupied.
    // Empty assignment is allowed, so 0 is always reachable.
    const long long NEG = LLONG_MIN / 4;   // unreachable sentinel
    int full = 1 << m;
    vector<long long> dp(full, NEG);
    dp[0] = 0;                              // before any parcel, no slot used, profit 0

    for (int i = 0; i < n; i++) {
        vector<long long> ndp(full, NEG);
        for (int mask = 0; mask < full; mask++) {
            if (dp[mask] == NEG) continue;
            long long base = dp[mask];
            // Option A: leave parcel i undelivered (slots unchanged).
            ndp[mask] = max(ndp[mask], base);
            // Option B: deliver parcel i in some free slot j.
            for (int j = 0; j < m; j++) {
                if (mask & (1 << j)) continue;
                int nmask = mask | (1 << j);
                ndp[nmask] = max(ndp[nmask], base + p[i][j]);
            }
        }
        dp.swap(ndp);
    }

    long long best = 0;                     // empty selection always allowed
    for (int mask = 0; mask < full; mask++)
        if (dp[mask] != NEG) best = max(best, dp[mask]);

    cout << best << "\n";
    return 0;
}
