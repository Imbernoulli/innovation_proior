#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    if (!(cin >> n)) return 0;             // n = 0 (or empty input) -> answer 0

    // cost[i][j] = cost of giving worker i task j.
    vector<vector<long long>> cost(n, vector<long long>(n));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            cin >> cost[i][j];

    if (n == 0) { cout << 0 << "\n"; return 0; }

    // dp[mask] = minimum total cost of assigning workers 0..k-1 to exactly the
    // tasks in `mask`, where k = popcount(mask). Worker index is implied by how
    // many bits are already set, so each worker is used exactly once and each
    // task at most once. Unreachable states stay at INF.
    const long long INF = LLONG_MAX / 4;
    vector<long long> dp(1 << n, INF);
    dp[0] = 0;                              // no workers placed, no tasks used

    for (int mask = 0; mask < (1 << n); mask++) {
        if (dp[mask] == INF) continue;
        int i = __builtin_popcount((unsigned)mask); // next worker to place
        if (i >= n) continue;                        // all workers already placed
        for (int j = 0; j < n; j++) {
            if (mask & (1 << j)) continue;           // task j already taken
            int nmask = mask | (1 << j);
            long long cand = dp[mask] + cost[i][j];
            if (cand < dp[nmask]) dp[nmask] = cand;
        }
    }

    cout << dp[(1 << n) - 1] << "\n";       // all tasks assigned
    return 0;
}
