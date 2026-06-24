#include <bits/stdc++.h>
using namespace std;

int main() {
    int m, k;
    if (!(cin >> m >> k)) return 0;          // empty input -> nothing to do
    vector<long long> cost(k);
    vector<int> mask(k);
    for (int i = 0; i < k; i++) {
        cin >> cost[i] >> mask[i];           // cost, then the skill bitmask (0 .. 2^m - 1)
    }

    const int FULL = (1 << m) - 1;
    const long long INF = LLONG_MAX / 4;
    vector<long long> dp(1 << m, INF);
    dp[0] = 0;                               // covering the empty set costs nothing

    // Forward DP over coverage states: from a reachable state, try hiring each contractor.
    for (int s = 0; s <= FULL; s++) {
        if (dp[s] == INF) continue;
        for (int i = 0; i < k; i++) {
            int ns = s | mask[i];
            long long nc = dp[s] + cost[i];
            if (nc < dp[ns]) dp[ns] = nc;
        }
    }

    if (dp[FULL] >= INF) cout << -1 << "\n";
    else cout << dp[FULL] << "\n";
    return 0;
}
