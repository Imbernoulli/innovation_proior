#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    if (!(cin >> n)) return 0;
    if (n == 0) { cout << 0 << "\n"; return 0; }
    vector<long long> w(n);
    for (auto &x : w) cin >> x;
    if (n == 1) { cout << 0 << "\n"; return 0; }

    // Circular merge of adjacent reels. Unroll the circle into a line of length 2n
    // so any contiguous arc of length L (over the original n reels) appears as a
    // contiguous segment [i, i+L-1] for some 0 <= i < n.
    int m = 2 * n;
    vector<long long> a(m + 1);
    for (int i = 0; i < m; i++) a[i] = w[i % n];
    // prefix sums of the doubled array
    vector<long long> pre(m + 1, 0);
    for (int i = 0; i < m; i++) pre[i + 1] = pre[i] + a[i];
    auto sum = [&](int l, int r) { return pre[r + 1] - pre[l]; }; // inclusive [l,r]

    const long long INF = LLONG_MAX / 4;
    // dp over segments of the doubled array; we only need segments of length <= n.
    // dp[l][len] = min cost to merge reels a[l..l+len-1] into one.
    // Use 2D vectors indexed by left endpoint l (0..m-1) and length len (1..n).
    vector<vector<long long>> dp(m, vector<long long>(n + 1, INF));
    for (int l = 0; l < m; l++) dp[l][1] = 0;
    for (int len = 2; len <= n; len++) {
        for (int l = 0; l + len <= m; l++) {
            int r = l + len - 1;
            long long best = INF;
            long long s = sum(l, r);
            for (int k = l; k < r; k++) {
                int leftLen = k - l + 1;
                int rightLen = r - k;
                long long cand = dp[l][leftLen] + dp[k + 1][rightLen] + s;
                if (cand < best) best = cand;
            }
            dp[l][len] = best;
        }
    }
    // Answer: best over all starting reels i of merging the whole circle into one,
    // i.e. the arc of length n starting at i.
    long long ans = INF;
    for (int i = 0; i < n; i++) ans = min(ans, dp[i][n]);
    cout << ans << "\n";
    return 0;
}
