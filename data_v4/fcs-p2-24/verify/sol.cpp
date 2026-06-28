#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    if (!(cin >> n)) return 0;             // no input -> nothing to do

    // dp[v] = fewest perfect squares (1,4,9,...) that sum to exactly v.
    // dp[0] = 0; dp[v] = 1 + min over squares s*s <= v of dp[v - s*s].
    vector<int> dp(n + 1, INT_MAX);
    dp[0] = 0;
    for (int v = 1; v <= n; v++) {
        for (int s = 1; (long long)s * s <= v; s++) {
            int prev = dp[v - s * s];      // always finite: dp[0]=0 reachable
            if (prev + 1 < dp[v]) dp[v] = prev + 1;
        }
    }

    cout << dp[n] << "\n";
    return 0;
}
