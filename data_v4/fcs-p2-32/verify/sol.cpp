#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, m;
    if (!(cin >> n >> m)) return 0;
    vector<long long> A(n), B(m);
    for (auto &x : A) cin >> x;
    for (auto &x : B) cin >> x;

    // dp[i][j] = best dot product of a NON-EMPTY equal-length pairing using
    // a subsequence of A[0..i-1] and of B[0..j-1]. NEG sentinel = "no non-empty
    // pairing exists yet for this prefix pair".
    const long long NEG = LLONG_MIN / 4;
    vector<vector<long long>> dp(n + 1, vector<long long>(m + 1, NEG));

    for (int i = 1; i <= n; i++) {
        for (int j = 1; j <= m; j++) {
            long long prod = A[i - 1] * B[j - 1];
            // pair A[i-1] with B[j-1], either as the only pair or extending dp[i-1][j-1]
            long long best = prod;                       // start a brand-new pairing here
            if (dp[i - 1][j - 1] != NEG)
                best = max(best, dp[i - 1][j - 1] + prod); // extend an existing pairing
            // or drop A[i-1] / drop B[j-1]
            best = max(best, dp[i - 1][j]);
            best = max(best, dp[i][j - 1]);
            dp[i][j] = best;
        }
    }

    cout << dp[n][m] << "\n";
    return 0;
}
