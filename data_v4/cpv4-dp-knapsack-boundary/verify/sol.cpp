#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    long long K, g;
    if (!(cin >> n >> K >> g)) return 0;
    vector<long long> s(n), v(n);
    for (int i = 0; i < n; i++) cin >> s[i] >> v[i];

    // Usable space: the locker has capacity K, but the last g units must stay
    // empty (fire-safety buffer). Positions are 1..K; reserving the last g of
    // them leaves usable positions 1..(K-g), i.e. U = K - g usable units.
    // If K - g < 0 there is no usable space at all -> treat U = 0.
    long long U = K - g;
    if (U < 0) U = 0;               // no usable space at all

    // 0/1 knapsack over capacity 0..U inclusive.
    // dp[c] = best value achievable using total occupied space exactly <= c.
    vector<long long> dp((size_t)U + 1, 0);
    for (int i = 0; i < n; i++) {
        long long si = s[i], vi = v[i];
        if (si > U) continue;                 // item cannot fit at all
        for (long long c = U; c >= si; c--) { // 0/1: descending
            long long cand = dp[c - si] + vi;
            if (cand > dp[c]) dp[c] = cand;
        }
    }

    cout << dp[U] << "\n";
    return 0;
}
