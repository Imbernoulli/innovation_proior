#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    if (!(cin >> n)) return 0;             // no input -> n = 0 -> answer 0
    vector<long long> a(n);
    for (auto &x : a) cin >> x;

    if (n <= 1) {                          // 0 or 1 token: no merge happens, score 0
        cout << 0 << "\n";
        return 0;
    }

    // prefix sums: pre[i] = a[0] + ... + a[i-1]; S(i..j) = pre[j+1] - pre[i].
    vector<long long> pre(n + 1, 0);
    for (int i = 0; i < n; i++) pre[i + 1] = pre[i] + a[i];
    auto S = [&](int i, int j) -> long long { return pre[j + 1] - pre[i]; };

    // dp[i][j] = max total score to collapse the inclusive interval [i..j] into one token.
    // dp[i][i] = 0 (a single token needs no merge).
    // dp[i][j] = max over split k in [i, j-1] of
    //            dp[i][k] + dp[k+1][j] + S(i,k) * S(k+1,j).
    vector<vector<long long>> dp(n, vector<long long>(n, 0));

    for (int len = 2; len <= n; len++) {
        for (int i = 0; i + len - 1 < n; i++) {
            int j = i + len - 1;
            long long best = LLONG_MIN;     // forced to merge -> may be negative; base must be -inf
            for (int k = i; k < j; k++) {
                long long cand = dp[i][k] + dp[k + 1][j] + S(i, k) * S(k + 1, j);
                if (cand > best) best = cand;
            }
            dp[i][j] = best;
        }
    }

    cout << dp[0][n - 1] << "\n";
    return 0;
}
