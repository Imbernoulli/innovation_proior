#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    long long C;
    if (!(cin >> n >> C)) return 0;
    vector<long long> w(n), v(n);
    for (int i = 0; i < n; i++) cin >> w[i] >> v[i];

    // dp[c] = best total value using a chosen subset with total weight EXACTLY c,
    // or -1 if c is unreachable. dp[0] = 0 (empty subset). 0/1 knapsack: iterate
    // capacity downward so each crate is used at most once.
    const long long NEG = -1;
    vector<long long> dp(C + 1, NEG);
    dp[0] = 0;
    for (int i = 0; i < n; i++) {
        if (w[i] > C) continue;                 // crate alone exceeds the hold
        for (long long c = C; c >= w[i]; c--) {
            if (dp[c - w[i]] != NEG) {
                long long cand = dp[c - w[i]] + v[i];
                if (cand > dp[c]) dp[c] = cand;
            }
        }
    }

    long long best = 0;                         // empty subset has value 0
    for (long long c = 0; c <= C; c++)
        if (dp[c] != NEG && dp[c] > best) best = dp[c];

    cout << best << "\n";
    return 0;
}
