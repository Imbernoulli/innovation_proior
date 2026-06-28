#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    string p, s;
    if (!(cin >> p)) return 0;            // empty input -> nothing to do
    if (!(cin >> s)) s = "";             // pattern present but string empty

    if (p == "-") p = "";                 // "-" denotes the empty pattern
    if (s == "-") s = "";                 // "-" denotes the empty string

    int n = (int)p.size();                // pattern length
    int m = (int)s.size();                // string length

    // dp[i][j] = does p[0..i-1] match s[0..j-1] ?
    // We keep two rolling rows of length (m+1) to stay within memory.
    vector<char> prev(m + 1, 0), cur(m + 1, 0);

    // Empty pattern matches empty string only.
    prev[0] = 1;
    for (int j = 1; j <= m; j++) prev[j] = 0;

    for (int i = 1; i <= n; i++) {
        char pc = p[i - 1];
        if (pc == '*') {
            // '*' matches an empty sequence: dp[i][0] = dp[i-1][0].
            cur[0] = prev[0];
        } else {
            // Any non-'*' pattern char cannot match the empty string.
            cur[0] = 0;
        }
        for (int j = 1; j <= m; j++) {
            char sc = s[j - 1];
            if (pc == '*') {
                // '*' = match empty (drop '*' : prev[j]) OR consume one more char (cur[j-1]).
                cur[j] = (prev[j] || cur[j - 1]) ? 1 : 0;
            } else if (pc == '?' || pc == sc) {
                // single-char match: consume one char from both.
                cur[j] = prev[j - 1];
            } else {
                cur[j] = 0;
            }
        }
        swap(prev, cur);
    }

    cout << (prev[m] ? "YES" : "NO") << "\n";
    return 0;
}
