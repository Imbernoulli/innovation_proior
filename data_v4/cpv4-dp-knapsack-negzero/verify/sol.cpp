#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    long long C;
    if (!(cin >> n >> C)) return 0;
    vector<long long> w(n), v(n);
    for (int i = 0; i < n; i++) cin >> w[i] >> v[i];

    // dp[c] = best total trim score achievable using a subset of weight exactly c.
    // Sentinel UNREACH marks "no subset reaches this weight". Base: dp[0] = 0 (empty subset).
    const long long UNREACH = LLONG_MIN / 4;
    vector<long long> dp(C + 1, UNREACH);
    dp[0] = 0;

    for (int i = 0; i < n; i++) {
        long long wi = w[i], vi = v[i];
        if (wi > C) continue;                 // can never fit into an exact total of C
        for (long long c = C; c >= wi; c--) { // 0/1: iterate capacity downward
            if (dp[c - wi] != UNREACH) {
                long long cand = dp[c - wi] + vi;
                if (cand > dp[c]) dp[c] = cand;
            }
        }
    }

    if (dp[C] == UNREACH) cout << "IMPOSSIBLE" << "\n";
    else cout << dp[C] << "\n";
    return 0;
}
