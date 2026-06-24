#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    string s;
    if (!(cin >> s)) return 0;
    int n = (int)s.size();

    // KMP failure function: pi[i] = length of the longest proper prefix of
    // s[0..i-1] that is also a suffix of s[0..i-1]. pi has size n+1, indexed by
    // PREFIX LENGTH L in [0, n]; pi[0] = pi[1] = 0 by definition. We build it on
    // 1-indexed lengths so the "period" arithmetic L - pi[L] is exact.
    vector<int> pi(n + 1, 0);
    int k = 0;                       // current matched border length
    for (int i = 2; i <= n; i++) {   // i = current prefix LENGTH
        // character being added is s[i-1] (0-indexed); compare against s[k]
        while (k > 0 && s[i - 1] != s[k]) k = pi[k];
        if (s[i - 1] == s[k]) k++;
        pi[i] = k;
    }

    // A prefix of length L (1 <= L <= n) is "tiled" iff its shortest period
    // d = L - pi[L] satisfies d < L (at least two copies) AND L % d == 0
    // (the period tiles the whole length exactly).
    long long count = 0;
    long long sumTile = 0;           // sum of minimal tile length d over tiled L
    for (int L = 1; L <= n; L++) {
        int d = L - pi[L];
        if (d < L && L % d == 0) {
            count++;
            sumTile += d;
        }
    }

    cout << count << " " << sumTile << "\n";
    return 0;
}
