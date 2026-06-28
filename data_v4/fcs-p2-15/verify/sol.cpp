#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    if (!(cin >> n)) return 0;              // n = 0 (or empty input) -> cost 0
    vector<long long> f(n + 1);
    for (int i = 1; i <= n; i++) cin >> f[i];

    // prefix[i] = f[1] + ... + f[i]; weight of interval [i..j] is prefix[j]-prefix[i-1].
    vector<long long> prefix(n + 1, 0);
    for (int i = 1; i <= n; i++) prefix[i] = prefix[i - 1] + f[i];

    // dp[i][j] = minimum expected cost (sum over depths*freq, root at depth 1)
    //            for an optimal BST built from keys i..j (1-indexed, inclusive).
    // dp[i][i-1] = 0 represents an empty range.
    // Recurrence: dp[i][j] = (prefix[j]-prefix[i-1])
    //                        + min over root r in [i..j] of dp[i][r-1] + dp[r+1][j].
    // The added interval weight accounts for every key in [i..j] sinking one level
    // deeper when we hang the two subtrees under the chosen root.
    vector<vector<long long>> dp(n + 2, vector<long long>(n + 2, 0));

    // len = number of keys in the interval, from 1 up to n.
    for (int len = 1; len <= n; len++) {
        for (int i = 1; i + len - 1 <= n; i++) {
            int j = i + len - 1;
            long long w = prefix[j] - prefix[i - 1];
            long long best = LLONG_MAX;
            for (int r = i; r <= j; r++) {
                long long left = dp[i][r - 1];      // r==i => empty left, dp[i][i-1]=0
                long long right = dp[r + 1][j];     // r==j => empty right, dp[j+1][j]=0
                long long cand = left + right;
                if (cand < best) best = cand;
            }
            dp[i][j] = best + w;
        }
    }

    cout << dp[1][n] << "\n";                // dp[1][0] = 0 when n == 0
    return 0;
}
