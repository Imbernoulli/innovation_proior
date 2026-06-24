#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    long long L;
    if (!(cin >> n >> L)) return 0;
    vector<long long> a(n);
    for (auto &x : a) cin >> x;

    // Prefix sums: P[i] = a[0] + ... + a[i-1], so sum(a[j..i-1]) = P[i] - P[j].
    vector<long long> P(n + 1, 0);
    for (int i = 0; i < n; i++) P[i + 1] = P[i] + a[i];

    // dp[i] = best total over the first i days (a[0..i-1]) choosing non-overlapping
    // intervals each of length >= L; we always allow choosing nothing, so dp[i] >= 0.
    // Transition at day boundary i:
    //   - leave day i-1 uncovered: dp[i] = dp[i-1]
    //   - end an interval [j, i-1] of length (i - j) >= L: dp[i] = dp[j] + (P[i] - P[j])
    //       = (P[i]) + max over valid j of (dp[j] - P[j]).
    // Maintain best = max over j (0 <= j <= i - L) of (dp[j] - P[j]) incrementally.
    const long long NEG = LLONG_MIN / 4;
    vector<long long> dp(n + 1, 0);
    long long best = NEG; // best of (dp[j] - P[j]) for j allowed at current i
    for (int i = 1; i <= n; i++) {
        // A new candidate j = i - L becomes available the moment i reaches that j + L.
        int j = i - (int)L;
        if (j >= 0) best = max(best, dp[j] - P[j]);
        dp[i] = dp[i - 1];                       // skip day i-1
        if (best > NEG / 2) dp[i] = max(dp[i], P[i] + best); // close an interval here
    }

    cout << dp[n] << "\n";
    return 0;
}
