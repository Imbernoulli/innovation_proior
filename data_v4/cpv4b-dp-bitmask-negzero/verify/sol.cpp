#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    if (!(cin >> n)) return 0;            // n = 0 -> empty roster -> 0
    vector<vector<long long>> p(n, vector<long long>(n));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) cin >> p[i][j];

    // We sweep stages s = 0..n-1. dp[mask] = best total reward achievable after deciding
    // stages 0..s-1, where `mask` is the set of artists already booked. Each stage may be
    // left empty (booking nobody, +0) or given one still-free artist (+p[artist][stage]).
    // Partial rosters are allowed, so the empty roster (dp all skipped) keeps the value 0.
    const long long NEG = LLONG_MIN / 4;
    int full = 1 << n;
    vector<long long> dp(full, NEG);
    dp[0] = 0;                            // before any stage: nobody booked, reward 0

    for (int s = 0; s < n; s++) {
        vector<long long> ndp(full, NEG);
        for (int mask = 0; mask < full; mask++) {
            if (dp[mask] == NEG) continue;
            // Option 1: leave stage s empty.
            if (dp[mask] > ndp[mask]) ndp[mask] = dp[mask];
            // Option 2: book one free artist a on stage s.
            for (int a = 0; a < n; a++) {
                if (mask & (1 << a)) continue;
                int nm = mask | (1 << a);
                long long cand = dp[mask] + p[a][s];
                if (cand > ndp[nm]) ndp[nm] = cand;
            }
        }
        dp.swap(ndp);
    }

    long long ans = 0;                    // empty roster is always allowed
    for (int mask = 0; mask < full; mask++)
        if (dp[mask] != NEG && dp[mask] > ans) ans = dp[mask];

    cout << ans << "\n";
    return 0;
}
