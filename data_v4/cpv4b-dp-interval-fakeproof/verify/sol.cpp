#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    if (!(cin >> n)) return 0;
    vector<long long> a(n);
    for (auto &x : a) cin >> x;
    if (n == 0) { cout << 0 << "\n"; return 0; }

    // px[i] = XOR of a[0..i-1]; XOR of segment [l..r] = px[r+1] ^ px[l].
    vector<long long> px(n + 1, 0);
    for (int i = 0; i < n; i++) px[i + 1] = px[i] ^ a[i];

    auto segxor = [&](int l, int r) -> long long { return px[r + 1] ^ px[l]; };

    const long long INF = LLONG_MAX / 4;
    // dp[l][r] = minimal total cost to merge segment [l..r] into one stone.
    vector<vector<long long>> dp(n, vector<long long>(n, 0));
    for (int len = 2; len <= n; len++) {
        for (int l = 0; l + len - 1 < n; l++) {
            int r = l + len - 1;
            long long best = INF;
            for (int k = l; k < r; k++) {
                // left part [l..k] -> value segxor(l,k); right part [k+1..r] -> value segxor(k+1,r)
                long long cost = dp[l][k] + dp[k + 1][r]
                               + (segxor(l, k) | segxor(k + 1, r));
                best = min(best, cost);
            }
            dp[l][r] = best;
        }
    }

    cout << dp[0][n - 1] << "\n";
    return 0;
}
