#include <bits/stdc++.h>
using namespace std;

// Booth's algorithm (1980): least starting index of the lexicographically
// minimal rotation of s. O(n) time, O(n) space. It runs a failure-function /
// KMP-style scan over the conceptual doubled string s+s; on a mismatch it slides
// the candidate start k forward past an entire already-matched block at once,
// which is what turns the naive O(n^2) compare-all-rotations into O(n).
int leastRotation(const string &s) {
    int n = (int)s.size();
    if (n == 0) return 0;
    string ss = s + s;                 // conceptual doubled string
    vector<int> f(2 * n, -1);          // failure function over ss; -1 = no match
    int k = 0;                         // current best rotation start
    for (int j = 1; j < 2 * n; j++) {
        char c = ss[j];
        int i = f[j - k - 1];
        while (i != -1 && c != ss[k + i + 1]) {
            if (c < ss[k + i + 1]) k = j - i - 1;
            i = f[i];
        }
        if (c != ss[k + i + 1]) {      // here i == -1
            if (c < ss[k + i + 1]) k = j;   // since i == -1, j - i - 1 == j
            f[j - k] = -1;
        } else {
            f[j - k] = i + 1;
        }
    }
    return k;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    string s;
    if (!(cin >> s)) {
        // Empty input: treat as the empty string; least-rotation index is 0.
        cout << 0 << "\n";
        return 0;
    }
    cout << leastRotation(s) << "\n";
    return 0;
}
