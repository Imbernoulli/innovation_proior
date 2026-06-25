#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int m, k;
    if (!(cin >> m >> k)) return 0;

    vector<int> mask(k, 0);
    vector<long long> cost(k, 0);
    for (int j = 0; j < k; j++) {
        long long c;
        int t;
        cin >> c >> t;
        cost[j] = c;
        int mk = 0;
        for (int s = 0; s < t; s++) {
            int ch;
            cin >> ch;
            mk |= (1 << ch);
        }
        mask[j] = mk;
    }

    const long long INF = LLONG_MAX / 4;
    int full = (1 << m) - 1;
    vector<long long> dp(1 << m, INF);
    dp[0] = 0;

    // dp[S] = minimum total cost of a set of bursts whose union of delivered
    // channels is exactly S' >= S, reached by accumulating bursts. We process
    // states in increasing order; from state S we may fire any burst j, moving
    // to S | mask[j] at additional cost cost[j]. Firing the same burst twice is
    // never beneficial, so monotone forward relaxation suffices.
    for (int S = 0; S <= full; S++) {
        if (dp[S] == INF) continue;
        long long base = dp[S];
        for (int j = 0; j < k; j++) {
            int ns = S | mask[j];
            if (ns == S) continue;            // adds nothing new
            long long nc = base + cost[j];
            if (nc < dp[ns]) dp[ns] = nc;
        }
    }

    if (dp[full] >= INF) cout << -1 << "\n";
    else cout << dp[full] << "\n";
    return 0;
}
