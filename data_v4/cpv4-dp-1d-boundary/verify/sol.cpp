#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    long long K;
    int L, R;
    if (!(cin >> n >> K >> L >> R)) return 0;
    vector<long long> v(n);
    for (auto &x : v) cin >> x;

    // prefix sums S[0..n], S[i] = v[0] + ... + v[i-1]
    vector<long long> S(n + 1, 0);
    for (int i = 0; i < n; i++) S[i + 1] = S[i] + v[i];

    const long long INF = (long long)4e18;
    // dp[i] = min cost to partition the first i unit-segments, i.e. the half-open
    // range [0, i), into valid billets. dp[0] = 0 (nothing cut yet).
    // A billet covering segments [j, i) has length (i - j), which must satisfy
    // L <= i - j <= R. Solving for j: j in [i - R, i - L], and also j >= 0.
    // The billet's cost is K + |S[i] - S[j]| (a setup fee plus the imbalance).
    vector<long long> dp(n + 1, INF);
    dp[0] = 0;
    for (int i = 1; i <= n; i++) {
        int jlo = max(0, i - R);   // longest allowed billet, length R
        int jhi = i - L;           // shortest allowed billet, length L
        for (int j = jlo; j <= jhi; j++) {
            if (dp[j] >= INF) continue;
            long long seg = S[i] - S[j];
            long long cost = K + llabs(seg);
            if (dp[j] + cost < dp[i]) dp[i] = dp[j] + cost;
        }
    }

    if (dp[n] >= INF) cout << -1 << "\n";
    else cout << dp[n] << "\n";
    return 0;
}
