#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    if (!(cin >> n)) return 0;             // n = 0 (or empty input) -> 0 coins
    vector<long long> v(n + 2);
    v[0] = 1;                              // virtual balloon "1" on the left
    v[n + 1] = 1;                          // virtual balloon "1" on the right
    for (int i = 1; i <= n; i++) cin >> v[i];

    // dp[l][r] = max coins from bursting every balloon strictly between
    // padded indices l and r (exclusive), with l and r still present.
    // Choose k in (l, r) as the LAST balloon burst in that open interval;
    // when k is burst its neighbours are exactly l and r, contributing
    // v[l] * v[k] * v[r], plus the two already-solved sub-intervals.
    vector<vector<long long>> dp(n + 2, vector<long long>(n + 2, 0));
    for (int len = 2; len <= n + 1; len++) {          // gap length r - l
        for (int l = 0; l + len <= n + 1; l++) {
            int r = l + len;
            long long best = 0;
            for (int k = l + 1; k < r; k++) {
                long long cur = dp[l][k] + v[l] * v[k] * v[r] + dp[k][r];
                if (cur > best) best = cur;
            }
            dp[l][r] = best;
        }
    }

    cout << dp[0][n + 1] << "\n";
    return 0;
}
