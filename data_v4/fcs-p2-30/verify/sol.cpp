#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    long long p;
    string s;
    if (!(cin >> p >> s)) return 0;        // missing input -> nothing to do
    int n = (int)s.size();

    // dp[i] = number of decodings of the prefix s[0..i-1], reduced mod p.
    // dp[0] = 1: the empty prefix has exactly one decoding (the empty string).
    // dp[i] = (s[i-1] is '1'..'9'        ? dp[i-1] : 0)
    //       + (s[i-2]s[i-1] in 10..26    ? dp[i-2] : 0)
    // We keep only the two most recent values in a rolling window.
    // prev1 = dp[i-1], prev2 = dp[i-2]. Before the loop (i==1): prev1 = dp[0] = 1,
    // prev2 = dp[-1] = 0 (a phantom; the i>=2 guard makes sure it is never used as dp[-1]).
    long long prev1 = 1 % p;               // dp[0]
    long long prev2 = 0;                   // dp[-1], unused while i < 2

    for (int i = 1; i <= n; i++) {
        char c1 = s[i - 1];                // i-th character (1-indexed): the one-digit group
        long long cur = 0;

        // One-digit group: s[i-1] alone decodes iff it is '1'..'9' (a leading '0' is invalid).
        if (c1 != '0') {
            cur += prev1;                  // extend each dp[i-1] decoding by this single digit
        }

        // Two-digit group: s[i-2]s[i-1] must form a value in 10..26.
        if (i >= 2) {
            int two = (s[i - 2] - '0') * 10 + (c1 - '0');
            if (two >= 10 && two <= 26) {
                cur += prev2;              // extend each dp[i-2] decoding by this two-digit group
            }
        }

        cur %= p;
        prev2 = prev1;                     // slide window: dp[i-2] <- dp[i-1]
        prev1 = cur;                       // dp[i-1] <- dp[i]
    }

    cout << (prev1 % p) << "\n";           // prev1 holds dp[n]
    return 0;
}
