#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    if (!(cin >> n)) return 0;             // empty input -> nothing to do
    vector<vector<long long>> s(n, vector<long long>(n));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            cin >> s[i][j];

    // dp[mask] = best total synergy achievable using exactly the runners in `mask`,
    // assigned to legs 0 .. popcount(mask)-1 (legs are filled in index order).
    // We assign the runner chosen for leg = popcount(mask) next.
    const long long NEG = LLONG_MIN / 4;   // sentinel for "unreachable"
    vector<long long> dp(1 << n, NEG);
    dp[0] = 0;                              // no runners placed, no legs filled, score 0

    for (int mask = 0; mask < (1 << n); mask++) {
        if (dp[mask] == NEG) continue;      // unreachable state
        int leg = __builtin_popcount((unsigned)mask); // next leg to fill
        if (leg == n) continue;             // all legs filled
        for (int i = 0; i < n; i++) {
            if (mask & (1 << i)) continue;  // runner i already used
            int nmask = mask | (1 << i);
            long long cand = dp[mask] + s[i][leg];
            if (cand > dp[nmask]) dp[nmask] = cand;
        }
    }

    cout << dp[(1 << n) - 1] << "\n";
    return 0;
}
