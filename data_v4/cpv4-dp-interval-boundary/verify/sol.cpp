#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    if (!(cin >> n)) return 0;
    vector<long long> w(n);
    for (auto &x : w) cin >> x;

    if (n <= 1) { cout << 0 << "\n"; return 0; }

    // prefix[i] = w[0] + ... + w[i-1], a half-open prefix sum over [0, i).
    vector<long long> prefix(n + 1, 0);
    for (int i = 0; i < n; i++) prefix[i + 1] = prefix[i] + w[i];

    // dp[i][j] = minimum total merge cost to combine the slabs whose indices
    // lie in the CLOSED interval [i, j] into one slab. The cost of the final
    // merge that unites the two children [i,k] and [k+1,j] is the combined
    // width prefix[j+1]-prefix[i] (sum of all w over the closed range).
    const long long INF = LLONG_MAX / 4;
    vector<vector<long long>> dp(n, vector<long long>(n, 0));
    // length-1 intervals already cost 0 (initialized above).

    for (int len = 2; len <= n; len++) {
        for (int i = 0; i + len - 1 < n; i++) {
            int j = i + len - 1;          // closed interval [i, j]
            long long best = INF;
            // split between k and k+1, so k ranges over i .. j-1 (inclusive).
            for (int k = i; k < j; k++) {
                long long cur = dp[i][k] + dp[k + 1][j];
                if (cur < best) best = cur;
            }
            // width of the closed range [i, j] is prefix[j+1] - prefix[i].
            dp[i][j] = best + (prefix[j + 1] - prefix[i]);
        }
    }

    cout << dp[0][n - 1] << "\n";
    return 0;
}
