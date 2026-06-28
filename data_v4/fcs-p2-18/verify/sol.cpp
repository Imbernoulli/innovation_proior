#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    string s;
    if (!(cin >> s)) {            // empty input -> empty string is already a palindrome
        cout << 0 << "\n";
        return 0;
    }
    int n = (int)s.size();

    // dp[i][j] = minimum insertions to turn the substring s[i..j] into a palindrome.
    // Base: every length-0 or length-1 substring needs 0 insertions.
    // Transition (i < j):
    //   if s[i] == s[j]: dp[i][j] = dp[i+1][j-1]      (matched ends, recurse inward)
    //   else:            dp[i][j] = 1 + min(dp[i+1][j], dp[i][j-1])
    // We only ever need the previous row (i+1), so keep two rolling rows of size n.
    vector<int> prev(n, 0), cur(n, 0);   // prev plays the role of dp[i+1][*]
    for (int i = n - 1; i >= 0; --i) {
        cur[i] = 0;                       // dp[i][i] = 0
        for (int j = i + 1; j < n; ++j) {
            if (s[i] == s[j]) {
                // dp[i+1][j-1]: row i+1 is in prev, column j-1
                cur[j] = (j - 1 >= i + 1) ? prev[j - 1] : 0; // length-2 match -> 0
            } else {
                // 1 + min(dp[i+1][j], dp[i][j-1])
                cur[j] = 1 + min(prev[j], cur[j - 1]);
            }
        }
        swap(prev, cur);                  // current row becomes "prev" for the next (smaller) i
    }

    // After the final swap, the row for i = 0 lives in prev.
    cout << prev[n - 1] << "\n";
    return 0;
}
