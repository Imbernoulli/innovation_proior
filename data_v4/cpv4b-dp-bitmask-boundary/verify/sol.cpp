#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int m, n;
    if (!(cin >> m >> n)) return 0;            // m vents (0..m-1), n sealant strips

    const long long INF = (long long)4e18;
    // Each strip seals a contiguous span of vents given as [a, b) (half-open: a..b-1) at cost c.
    // Build the bitmask of vents a strip covers, then run a set-cover DP over 2^m subsets.
    vector<int> mask(n);
    vector<long long> cost(n);
    for (int i = 0; i < n; i++) {
        int a, b; long long c;
        cin >> a >> b >> c;                    // strip seals vents a..b-1, cost c
        int bits = 0;
        for (int h = a; h < b; h++) bits |= (1 << h);   // inclusive a, exclusive b
        mask[i] = bits;
        cost[i] = c;
    }

    int FULL = (1 << m) - 1;
    // dp[S] = minimum total cost of a multiset of strips whose union of sealed vents is exactly S
    //         reachable by adding strips one at a time. We only need to reach the FULL set.
    vector<long long> dp(1 << m, INF);
    dp[0] = 0;
    for (int S = 0; S < (1 << m); S++) {
        if (dp[S] == INF) continue;
        for (int i = 0; i < n; i++) {
            int nS = S | mask[i];
            if (dp[S] + cost[i] < dp[nS]) dp[nS] = dp[S] + cost[i];
        }
    }

    if (dp[FULL] >= INF) cout << -1 << "\n";
    else cout << dp[FULL] << "\n";
    return 0;
}
