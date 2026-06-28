#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    long long T;
    if (!(cin >> n >> T)) return 0;          // empty input -> nothing to do

    const long long MOD = 1000000007LL;

    // dp[s] = number of subsets of the items seen so far whose sum is exactly s, mod MOD.
    // Start with the empty subset, which has sum 0.
    vector<long long> dp(T + 1, 0);
    dp[0] = 1;

    for (int i = 0; i < n; i++) {
        long long v;
        cin >> v;                            // a[i], a non-negative integer
        // 0/1 knapsack count: each item is used at most once.
        // Iterate s from high to low so dp[s - v] still refers to subsets
        // that do NOT yet include item i (prevents reusing one item twice).
        // This descending order is correct even when v == 0.
        for (long long s = T; s >= v; s--) {
            dp[s] = (dp[s] + dp[s - v]) % MOD;
        }
    }

    cout << dp[T] % MOD << "\n";
    return 0;
}
