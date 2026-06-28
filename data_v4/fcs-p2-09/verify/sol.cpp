#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    string s, t;
    if (!(cin >> s)) return 0;             // empty input -> LCS length 0
    if (!(cin >> t)) { cout << 0 << "\n"; return 0; }

    int n = (int)s.size(), m = (int)t.size();

    // dp[j] = LCS length of s[0..i-1] and t[0..j-1], rolled over rows of s.
    // Two rolling rows keep memory at O(m) while the recurrence stays O(n*m).
    vector<int> prev(m + 1, 0), cur(m + 1, 0);
    for (int i = 1; i <= n; i++) {
        cur[0] = 0;
        char si = s[i - 1];
        for (int j = 1; j <= m; j++) {
            if (si == t[j - 1])
                cur[j] = prev[j - 1] + 1;          // extend the diagonal match
            else
                cur[j] = max(prev[j], cur[j - 1]);  // drop one char from s or from t
        }
        swap(prev, cur);
    }

    cout << prev[m] << "\n";
    return 0;
}
