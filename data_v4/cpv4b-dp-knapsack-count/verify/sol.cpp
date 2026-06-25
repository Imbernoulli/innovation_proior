#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    long long S;
    long long MOD;
    if (!(cin >> n >> S >> MOD)) return 0;

    vector<long long> v(n), c(n);
    for (int i = 0; i < n; i++) cin >> v[i] >> c[i];

    // dp[s] = number of distinct multisets (combinations) of stamps, drawn from
    // the denominations processed so far, whose values sum to exactly s, mod MOD.
    // Denomination in the OUTER loop, capacity in the INNER loop: each multiset
    // is counted exactly once (unordered).
    vector<long long> dp(S + 1, 0);
    dp[0] = 1 % MOD;

    for (int i = 0; i < n; i++) {
        long long val = v[i];
        long long lim = c[i];               // max copies of denomination i

        // ndp[s] = sum_{k=0..lim} dp[s - k*val]  (k copies of this denomination,
        // remainder a combination over the PREVIOUS denominations). We compute it
        // with a sliding window of width (lim+1) along each residue class mod val,
        // so the transition is O(S) rather than O(S * lim).
        vector<long long> ndp(S + 1, 0);
        for (long long r = 0; r < val && r <= S; r++) {
            long long window = 0;           // sum of dp at the last (lim+1) terms
            long long s = r;
            long long count = 0;            // how many terms are currently in window
            for (; s <= S; s += val) {
                window += dp[s];
                if (window >= MOD) window -= MOD;
                count++;
                if (count > lim + 1) {      // window holds more than lim+1 terms: drop oldest
                    long long old = s - (lim + 1) * val;
                    window -= dp[old];
                    if (window < 0) window += MOD;
                    count--;
                }
                ndp[s] = window;
            }
        }
        dp.swap(ndp);
    }

    cout << dp[S] % MOD << "\n";
    return 0;
}
