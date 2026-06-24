#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    long long W;
    if (!(cin >> n >> W)) return 0;

    vector<long long> wt(n), val(n);
    for (int i = 0; i < n; i++) cin >> wt[i] >> val[i];

    // dp[c] = maximum total value achievable using a capacity budget of exactly-at-most c.
    // 0/1 knapsack: each item used at most once -> iterate capacity downward.
    vector<long long> dp(W + 1, 0);
    for (int i = 0; i < n; i++) {
        long long w = wt[i], v = val[i];
        if (w > W) continue;                       // item never fits
        for (long long c = W; c >= w; c--) {
            long long cand = dp[c - w] + v;        // both operands long long -> no overflow
            if (cand > dp[c]) dp[c] = cand;
        }
    }

    cout << dp[W] << "\n";
    return 0;
}
