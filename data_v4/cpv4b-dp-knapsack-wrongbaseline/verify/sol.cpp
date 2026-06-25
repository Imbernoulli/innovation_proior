#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    long long W;
    if (!(cin >> n >> W)) return 0;          // no data -> nothing scheduled
    vector<long long> e(n), v(n);            // e[i] = energy cost, v[i] = science value
    for (int i = 0; i < n; i++) cin >> e[i] >> v[i];

    // 0/1 knapsack: each experiment scheduled at most once, total energy <= W.
    // dp[c] = best total value using energy budget exactly-bounded-by c.
    vector<long long> dp(W + 1, 0);
    for (int i = 0; i < n; i++) {
        if (e[i] > W) continue;              // never fits, skip to avoid touching dp out of range
        // DESCENDING capacity: each item contributes to a strictly smaller earlier state,
        // so within this i the item is used at most once.
        for (long long c = W; c >= e[i]; c--) {
            long long cand = dp[c - e[i]] + v[i];
            if (cand > dp[c]) dp[c] = cand;
        }
    }

    cout << dp[W] << "\n";
    return 0;
}
