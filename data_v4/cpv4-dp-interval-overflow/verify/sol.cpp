#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    if (!(cin >> n)) return 0;
    vector<long long> v(n);
    for (auto &x : v) cin >> x;

    if (n < 3) { cout << 0 << "\n"; return 0; }

    // dp[i][j] = min total cost to triangulate the sub-polygon whose boundary
    // runs i, i+1, ..., j (a contiguous arc plus the closing chord i-j).
    // Triangle cost for vertices (i, k, j) is v[i]*v[k]*v[j].
    vector<vector<long long>> dp(n, vector<long long>(n, 0));

    for (int len = 2; len < n; ++len) {            // j - i = len, need len>=2 for a triangle
        for (int i = 0; i + len < n; ++i) {
            int j = i + len;
            long long best = LLONG_MAX;
            for (int k = i + 1; k < j; ++k) {
                long long cost = dp[i][k] + dp[k][j] + v[i] * v[k] * v[j];
                if (cost < best) best = cost;
            }
            dp[i][j] = best;
        }
    }

    cout << dp[0][n - 1] << "\n";
    return 0;
}
