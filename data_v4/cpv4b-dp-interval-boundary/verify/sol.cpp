#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    long long K;
    int L;
    if (!(cin >> n >> K >> L)) return 0;      // empty input
    vector<long long> a(n + 1);               // 1-indexed heights a[1..n]
    for (int i = 1; i <= n; i++) cin >> a[i];

    // dp[i] = minimum total cost to tile the first i panels (panels 1..i).
    // dp[0] = 0 (no panels, no rugs). The last rug covers an INCLUSIVE
    // interval [j+1, i] of length (i - (j+1) + 1) = i - j panels, which must
    // be between 1 and L. Its cost is K + max(a[j+1..i]).
    const long long INF = LLONG_MAX / 4;
    vector<long long> dp(n + 1, INF);
    dp[0] = 0;

    for (int i = 1; i <= n; i++) {
        long long curMax = 0;                 // max over the growing suffix
        // j is the index BEFORE the last rug; last rug = [j+1, i].
        // length = i - j must satisfy 1 <= i - j <= L, so j ranges in
        // [max(0, i - L), i - 1]. We extend the rug leftward from j = i-1.
        int lo = max(0, i - L);
        for (int j = i - 1; j >= lo; j--) {
            curMax = max(curMax, a[j + 1]);   // include panel (j+1) inclusively
            if (dp[j] != INF) {
                long long cand = dp[j] + K + curMax;
                if (cand < dp[i]) dp[i] = cand;
            }
        }
    }

    cout << dp[n] << "\n";
    return 0;
}
