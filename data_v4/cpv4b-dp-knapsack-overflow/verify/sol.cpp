#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    long long C;
    if (!(cin >> n >> C)) return 0;

    vector<long long> w(n), b(n);
    for (int i = 0; i < n; i++) cin >> w[i] >> b[i];

    // dp[c] = maximum total brightness using a subset whose total wattage is exactly <= c.
    // Brightness sums can reach n * 1e9 = 2e12, so dp MUST be 64-bit.
    vector<long long> dp(C + 1, 0);
    for (int i = 0; i < n; i++) {
        long long wi = w[i], bi = b[i];
        if (wi > C) continue;                 // module can never fit
        for (long long c = C; c >= wi; c--) { // 0/1 knapsack: iterate capacity downward
            long long cand = dp[c - wi] + bi;
            if (cand > dp[c]) dp[c] = cand;
        }
    }

    cout << dp[C] << "\n";
    return 0;
}
