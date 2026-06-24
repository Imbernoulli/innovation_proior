#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    long long n, W;
    if (!(cin >> n >> W)) return 0;
    vector<long long> w(n);
    for (auto &x : w) cin >> x;

    // prefix[i] = sum of widths of beads 0..i-1
    vector<long long> pre(n + 1, 0);
    for (int i = 0; i < n; i++) pre[i + 1] = pre[i] + w[i];

    const long long INF = LLONG_MAX / 4;

    // dp[i] = minimum total penalty to pack the first i beads (0..i-1) into lines.
    // A line covering beads [j..i-1] is allowed only if its used width
    //   used = sum w[j..i-1] + (i-1-j)   (one unit gap between adjacent beads)
    // does not exceed W. Penalty of that line is (W - used)^2, EXCEPT the last
    // line (the one ending at i == n) has penalty 0 (no trailing-slack penalty).
    vector<long long> dp(n + 1, INF);
    dp[0] = 0;
    for (int i = 1; i <= n; i++) {
        // line is [j .. i-1], 0-based beads, j from i-1 down to 0
        for (int j = i; j >= 1; j--) {
            // beads j-1 .. i-1  (1-based prefix indexing: beads with indices j-1..i-1)
            long long cnt = i - (j - 1);             // number of beads on the line
            long long widthSum = pre[i] - pre[j - 1];
            long long used = widthSum + (cnt - 1);   // gaps between beads
            if (used > W) break;                     // adding earlier beads only grows used
            if (dp[j - 1] == INF) continue;
            long long slack = W - used;
            long long pen = (i == n) ? 0 : slack * slack; // last line: no slack penalty
            dp[i] = min(dp[i], dp[j - 1] + pen);
        }
    }

    cout << dp[n] << "\n";
    return 0;
}
