#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    if (!(cin >> n)) return 0;             // n = 0 (or empty input) -> answer 0
    vector<long long> w(n);
    for (auto &x : w) cin >> x;
    if (n <= 1) { cout << 0 << "\n"; return 0; } // 0 or 1 pile: no merge needed

    // prefix[i] = w[0] + ... + w[i-1]; sum of piles in interval [i..j] is prefix[j+1]-prefix[i].
    vector<long long> prefix(n + 1, 0);
    for (int i = 0; i < n; i++) prefix[i + 1] = prefix[i] + w[i];
    auto range = [&](int i, int j) -> long long { return prefix[j + 1] - prefix[i]; };

    // dp[i][j] = min cost to merge piles i..j into one. opt[i][j] = a splitting index k
    // (i <= k < j) that attains the minimum, used to drive Knuth's monotonicity bounds.
    vector<vector<long long>> dp(n, vector<long long>(n, 0));
    vector<vector<int>> opt(n, vector<int>(n, 0));

    // Length-1 intervals cost 0; their "optimal split" is the index itself (boundary anchor).
    for (int i = 0; i < n; i++) { dp[i][i] = 0; opt[i][i] = i; }

    // Build by increasing interval length. Knuth: opt[i][j-1] <= opt[i][j] <= opt[i+1][j].
    for (int len = 2; len <= n; len++) {
        for (int i = 0; i + len - 1 < n; i++) {
            int j = i + len - 1;
            long long best = LLONG_MAX;
            int lo = opt[i][j - 1];      // Knuth lower bound on the optimal split
            int hi = opt[i + 1][j];      // Knuth upper bound on the optimal split
            if (lo < i) lo = i;          // a split index k lives in [i, j-1]
            if (hi > j - 1) hi = j - 1;
            int bestk = lo;
            for (int k = lo; k <= hi; k++) {
                long long cand = dp[i][k] + dp[k + 1][j];
                if (cand < best) { best = cand; bestk = k; }
            }
            dp[i][j] = best + range(i, j);     // every merge in [i..j] re-touches all its weight
            opt[i][j] = bestk;
        }
    }

    cout << dp[0][n - 1] << "\n";
    return 0;
}
