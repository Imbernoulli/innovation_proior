#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    if (!(cin >> n)) return 0;             // n = 0 (no matrices) -> answer 0
    vector<long long> p(n + 1);
    for (auto &x : p) cin >> x;            // p[0..n]: matrix i is p[i-1] x p[i]

    if (n <= 1) { cout << 0 << "\n"; return 0; } // 0 or 1 matrix: nothing to multiply

    // dp[i][j] = min scalar multiplications to multiply matrices i..j (1-indexed, inclusive).
    // dp[i][i] = 0; for i < j, split at k in [i, j-1]:
    //   dp[i][j] = min over k of dp[i][k] + dp[k+1][j] + p[i-1]*p[k]*p[j].
    const long long INF = LLONG_MAX / 4;
    vector<vector<long long>> dp(n + 1, vector<long long>(n + 1, 0));

    for (int len = 2; len <= n; len++) {        // chain length
        for (int i = 1; i + len - 1 <= n; i++) {
            int j = i + len - 1;
            long long best = INF;
            for (int k = i; k < j; k++) {
                long long cost = dp[i][k] + dp[k + 1][j]
                                 + p[i - 1] * p[k] * p[j];
                if (cost < best) best = cost;
            }
            dp[i][j] = best;
        }
    }

    cout << dp[1][n] << "\n";
    return 0;
}
