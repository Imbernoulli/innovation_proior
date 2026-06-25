#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    long long L, R;
    if (!(cin >> n >> L >> R)) return 0;

    vector<long long> p(n), v(n);
    for (int i = 0; i < n; i++) cin >> p[i] >> v[i];

    const long long NEG = LLONG_MIN / 4;
    // dp[s] = best total joy using a subset whose total price is EXACTLY s,
    // for s in [0, R] (indices 0..R inclusive, so size R+1).
    vector<long long> dp(R + 1, NEG);
    dp[0] = 0;                                   // empty subset: price 0, joy 0

    for (int i = 0; i < n; i++) {
        long long pi = p[i], vi = v[i];
        if (pi > R) continue;                    // cannot ever fit within the cap
        for (long long s = R; s >= pi; s--) {    // 0/1 knapsack: descend so each item used once
            if (dp[s - pi] != NEG)
                dp[s] = max(dp[s], dp[s - pi] + vi);
        }
    }

    long long best = NEG;
    for (long long s = L; s <= R; s++)           // window [L, R] INCLUSIVE on both ends
        best = max(best, dp[s]);

    if (best == NEG) cout << "IMPOSSIBLE\n";
    else cout << best << "\n";
    return 0;
}
