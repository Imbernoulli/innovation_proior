#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    if (!(cin >> n)) return 0;             // n = 0 (or empty input) -> answer 0
    vector<long long> a(n);
    for (auto &x : a) cin >> x;

    // dp[i] = max sum of a STRICTLY increasing subsequence that ends exactly at i.
    // A subsequence ending at i either starts at i (sum a[i]) or extends some j<i
    // with a[j] < a[i] (sum dp[j] + a[i]); we keep the best such predecessor.
    long long answer = 0;                  // empty subsequence is always allowed
    vector<long long> dp(n);
    for (int i = 0; i < n; i++) {
        long long best = 0;                // 0 = "no predecessor", start fresh at i
        for (int j = 0; j < i; j++) {
            if (a[j] < a[i] && dp[j] > best) best = dp[j];
        }
        dp[i] = best + a[i];
        if (dp[i] > answer) answer = dp[i];
    }

    cout << answer << "\n";
    return 0;
}
