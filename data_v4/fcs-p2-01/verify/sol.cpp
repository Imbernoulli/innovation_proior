#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    long long S;
    if (!(cin >> n >> S)) return 0;        // empty input -> nothing to do
    vector<long long> c(n);
    for (auto &x : c) cin >> x;

    // dp[s] = minimum number of coins to make sum s exactly; INF if unreachable.
    // Unbounded supply of each denomination, so each coin relaxes every larger sum.
    const long long INF = (long long)4e18;
    vector<long long> dp((size_t)S + 1, INF);
    dp[0] = 0;                              // zero coins make sum 0
    for (long long s = 1; s <= S; s++) {
        for (int i = 0; i < n; i++) {
            long long v = c[i];
            if (v <= s && dp[s - v] != INF && dp[s - v] + 1 < dp[s]) {
                dp[s] = dp[s - v] + 1;
            }
        }
    }

    cout << (dp[S] == INF ? -1 : dp[S]) << "\n";
    return 0;
}
