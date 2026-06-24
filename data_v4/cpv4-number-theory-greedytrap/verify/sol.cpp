#include <bits/stdc++.h>
using namespace std;

int main() {
    long long k, n;
    if (!(cin >> k >> n)) return 0;        // empty input -> nothing to do

    // n == 0 needs zero pulses; handle before allocating the DP table.
    if (n == 0) { cout << 0 << "\n"; return 0; }

    // Enumerate every perfect k-th power in [1, n]: 1^k, 2^k, 3^k, ...
    // Multiply step by step, bailing out the instant the running product
    // exceeds n so the product never overflows.
    vector<long long> powers;
    for (long long b = 1;; b++) {
        long long p = 1;
        bool exceed = false;
        for (long long e = 0; e < k; e++) {
            if (p > n / b) { exceed = true; break; }  // p*b would exceed n
            p *= b;
        }
        if (exceed || p > n) break;
        powers.push_back(p);
    }
    // powers is ascending; powers[0] == 1 guarantees every n is representable.

    const int INF = 1e9;
    vector<int> dp(n + 1, INF);
    dp[0] = 0;
    for (long long v = 1; v <= n; v++) {
        int best = INF;
        for (long long p : powers) {
            if (p > v) break;                 // ascending: all later powers exceed v too
            int cand = dp[v - p] + 1;
            if (cand < best) best = cand;
        }
        dp[v] = best;
    }

    cout << dp[n] << "\n";
    return 0;
}
