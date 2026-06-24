#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    long long A;
    if (!(cin >> n >> A)) return 0;          // empty input -> nothing to do
    vector<long long> d(n);
    for (auto &x : d) cin >> x;

    // Unbounded "fewest stamps to make exactly A" by DP over amounts 0..A.
    // dp[v] = minimum number of stamps summing to exactly v, or INF if impossible.
    const long long INF = (long long)4e18;
    vector<long long> dp(A + 1, INF);
    dp[0] = 0;
    for (long long v = 1; v <= A; v++) {
        for (int i = 0; i < n; i++) {
            if (d[i] <= v && dp[v - d[i]] != INF) {
                long long cand = dp[v - d[i]] + 1;
                if (cand < dp[v]) dp[v] = cand;
            }
        }
    }

    if (dp[A] == INF) cout << -1 << "\n";
    else cout << dp[A] << "\n";
    return 0;
}
