#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    long long K, C;
    if (!(cin >> n >> K >> C)) return 0;
    vector<long long> w(n), v(n);
    for (int i = 0; i < n; i++) cin >> w[i] >> v[i];

    // If the required count is impossible up front, no feasible load exists.
    if (K < 0 || K > n) { cout << "INFEASIBLE" << "\n"; return 0; }

    const long long NEG = LLONG_MIN / 4;       // "no subset with this (count,weight) exists"
    int Kc = (int)K;
    int Cc = (int)C;

    // dp[k][c] = best total profit using EXACTLY k parcels of total weight EXACTLY c (c in 0..Cc).
    vector<vector<long long>> dp(Kc + 1, vector<long long>(Cc + 1, NEG));
    dp[0][0] = 0;                              // exactly 0 parcels, weight 0, profit 0 (empty load)

    for (int i = 0; i < n; i++) {
        long long wi = w[i], vi = v[i];
        if (wi > (long long)Cc) continue;      // parcel alone exceeds capacity: never choosable
        int wint = (int)wi;
        for (int k = Kc - 1; k >= 0; k--) {    // 0/1: counts downward so each parcel used once
            for (int c = Cc - wint; c >= 0; c--) {
                if (dp[k][c] == NEG) continue;
                long long cand = dp[k][c] + vi;
                if (cand > dp[k + 1][c + wint]) dp[k + 1][c + wint] = cand;
            }
        }
    }

    long long ans = NEG;
    for (int c = 0; c <= Cc; c++)
        if (dp[Kc][c] != NEG && dp[Kc][c] > ans) ans = dp[Kc][c];

    if (ans == NEG) cout << "INFEASIBLE" << "\n";
    else cout << ans << "\n";
    return 0;
}
