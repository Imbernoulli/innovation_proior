#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    if (!(cin >> n)) return 0;             // empty input -> 0 (no platforms, no tolls)
    vector<long long> t(n);
    for (auto &x : t) cin >> x;

    if (n == 0) { cout << 0 << "\n"; return 0; }
    if (n == 1) { cout << t[0] << "\n"; return 0; } // must land on the only platform

    // dp[i] = minimum total toll paid for a legal route that LANDS on platform i,
    // starting from platform 0 (always landed on) and using jumps of +1 or +2.
    // dp[0] = t[0]. To reach i you arrive from i-1 (a +1 jump) or i-2 (a +2 jump):
    //   dp[i] = t[i] + min(dp[i-1], dp[i-2]).
    // We only keep the previous two values (O(1) memory).
    long long prev2 = t[0];                 // dp[0]
    long long prev1 = t[1] + t[0];          // dp[1]: only reachable from 0 via +1
    for (int i = 2; i < n; i++) {
        long long cur = t[i] + min(prev1, prev2);
        prev2 = prev1;
        prev1 = cur;
    }

    cout << prev1 << "\n";                   // dp[n-1]: must finish on the last platform
    return 0;
}
