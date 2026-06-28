#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    if (!(cin >> n)) return 0;              // n = 0 dictionary words possible
    vector<string> dict(n);
    int maxLen = 0;
    for (auto &w : dict) {
        cin >> w;
        maxLen = max(maxLen, (int)w.size());
    }
    string s;
    cin >> s;                               // the string to segment
    int m = (int)s.size();

    // Put the dictionary in a hash set for O(1) average membership tests.
    unordered_set<string> words(dict.begin(), dict.end());
    words.reserve(dict.size() * 2 + 1);

    // dp[i] = true if the prefix s[0..i-1] (length i) can be segmented.
    // dp[0] = true (empty prefix). For each end position i, look back at every
    // possible last-word start j and check dp[j] && s[j..i-1] in dictionary.
    // Bounded word length keeps the inner loop to O(maxLen) instead of O(n).
    vector<char> dp(m + 1, 0);
    dp[0] = 1;
    for (int i = 1; i <= m; ++i) {
        int lo = max(0, i - maxLen);        // last word can be at most maxLen long
        for (int j = i - 1; j >= lo; --j) {
            if (dp[j] && words.count(s.substr(j, i - j))) {
                dp[i] = 1;
                break;
            }
        }
    }

    cout << (dp[m] ? "YES" : "NO") << "\n";
    return 0;
}
