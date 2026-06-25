#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    if (!(cin >> n)) return 0;             // empty input -> nothing to do
    vector<long long> w(n);
    for (auto &x : w) cin >> x;
    if (n <= 1) { cout << 0 << "\n"; return 0; }  // 0 or 1 stack: no merges

    // prefix[i] = w[0] + ... + w[i-1]; sum of [l..r] = prefix[r+1]-prefix[l]
    vector<long long> prefix(n + 1, 0);
    for (int i = 0; i < n; i++) prefix[i + 1] = prefix[i] + w[i];

    // dp[l][r] = minimum total effort to merge stacks l..r into one.
    const long long INF = LLONG_MAX / 4;
    vector<vector<long long>> dp(n, vector<long long>(n, 0));
    // base dp[i][i] = 0 already.

    for (int len = 2; len <= n; len++) {
        for (int l = 0; l + len - 1 < n; l++) {
            int r = l + len - 1;
            long long seg = prefix[r + 1] - prefix[l]; // crates in [l..r]
            long long best = INF;
            for (int k = l; k < r; k++) {
                long long cand = dp[l][k] + dp[k + 1][r] + seg;
                if (cand < best) best = cand;
            }
            dp[l][r] = best;
        }
    }

    cout << dp[0][n - 1] << "\n";
    return 0;
}
