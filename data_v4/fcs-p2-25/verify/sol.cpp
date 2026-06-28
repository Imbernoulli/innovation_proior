#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    if (!(cin >> n)) return 0;             // no input -> nothing to do
    if (n == 0) { cout << 0 << "\n"; return 0; }

    vector<vector<long long>> a(n, vector<long long>(n));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            cin >> a[i][j];

    // dp[j] = minimum sum of a falling path that ends at cell (current row, j).
    // Row 0: a path ending at (0, j) is just the single cell a[0][j].
    vector<long long> dp(a[0].begin(), a[0].end());

    for (int i = 1; i < n; i++) {
        vector<long long> ndp(n);
        for (int j = 0; j < n; j++) {
            long long best = dp[j];                       // came from (i-1, j)
            if (j > 0)     best = min(best, dp[j - 1]);    // came from (i-1, j-1)
            if (j + 1 < n) best = min(best, dp[j + 1]);    // came from (i-1, j+1)
            ndp[j] = best + a[i][j];
        }
        dp = move(ndp);
    }

    long long answer = *min_element(dp.begin(), dp.end());
    cout << answer << "\n";
    return 0;
}
