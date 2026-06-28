#include <bits/stdc++.h>
using namespace std;

int main() {
    string s;
    if (!(cin >> s)) {            // empty input -> empty string -> 0 cuts
        cout << 0 << "\n";
        return 0;
    }
    int n = (int)s.size();

    // pal[i][j] = true iff s[i..j] (inclusive) is a palindrome.
    // Fill by increasing substring length so the inner [i+1..j-1] is ready.
    vector<vector<char>> pal(n, vector<char>(n, 0));
    for (int i = 0; i < n; i++) pal[i][i] = 1;                 // length 1
    for (int len = 2; len <= n; len++) {
        for (int i = 0; i + len - 1 < n; i++) {
            int j = i + len - 1;
            if (s[i] == s[j] && (len == 2 || pal[i + 1][j - 1]))
                pal[i][j] = 1;
        }
    }

    // cut[j] = minimum number of cuts so that s[0..j] splits into palindromes.
    // cut[j] = 0 if s[0..j] is itself a palindrome; otherwise
    // cut[j] = min over k in [1..j] with s[k..j] palindrome of cut[k-1] + 1.
    const int INF = INT_MAX / 2;
    vector<int> cut(n, INF);
    for (int j = 0; j < n; j++) {
        if (pal[0][j]) {
            cut[j] = 0;
            continue;
        }
        for (int k = 1; k <= j; k++) {
            if (pal[k][j] && cut[k - 1] + 1 < cut[j])
                cut[j] = cut[k - 1] + 1;
        }
    }

    cout << cut[n - 1] << "\n";
    return 0;
}
