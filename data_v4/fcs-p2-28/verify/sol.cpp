#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    string s;
    if (!(cin >> s)) {           // empty input -> empty string -> answer 0
        cout << 0 << "\n";
        return 0;
    }
    int n = (int)s.size();

    // dp[i][j] = length of the longest palindromic subsequence of s[i..j].
    // Base: dp[i][i] = 1. Recurrence over increasing length L = j - i + 1:
    //   if s[i] == s[j]: dp[i][j] = dp[i+1][j-1] + 2   (the empty inner interval contributes 0)
    //   else:            dp[i][j] = max(dp[i+1][j], dp[i][j-1])
    // Two rolling rows suffice: 'cur' indexed by i holds dp for the current length,
    // 'prev' holds dp for length-1 intervals, 'prev2' for length-2 intervals.
    vector<int> prev2(n, 0), prev(n, 0), cur(n, 0);

    // Length 1.
    for (int i = 0; i < n; i++) prev[i] = 1;
    if (n == 1) { cout << 1 << "\n"; return 0; }

    int answer = 1;
    for (int L = 2; L <= n; L++) {
        for (int i = 0; i + L - 1 < n; i++) {
            int j = i + L - 1;
            int val;
            if (s[i] == s[j]) {
                // Inner interval s[i+1..j-1] has length L-2.
                // For L == 2 the inner interval is empty, contributing 0.
                int inner = (L == 2) ? 0 : prev2[i + 1];
                val = inner + 2;
            } else {
                // dp[i+1][j] and dp[i][j-1] are both length-(L-1) intervals: 'prev'.
                val = max(prev[i + 1], prev[i]);
            }
            cur[i] = val;
            if (val > answer) answer = val;
        }
        // Shift rolling rows.
        prev2 = prev;
        prev = cur;
    }

    cout << answer << "\n";
    return 0;
}
