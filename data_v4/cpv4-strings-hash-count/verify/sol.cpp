#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    long long n, k;
    if (!(cin >> n >> k)) return 0;
    string s;
    if (n > 0) cin >> s;            // when n == 0 there is no string token

    // Number of length-k windows. If k is out of [1, n] there are none.
    if (k < 1 || k > n) { cout << 0 << "\n"; return 0; }
    long long W = n - k + 1;        // window count: starts 0..n-k inclusive

    // Two independent polynomial rolling hashes, packed into one 64-bit key.
    const unsigned long long M1 = 1000000007ULL, M2 = 998244353ULL;
    const unsigned long long B1 = 131ULL, B2 = 137ULL;

    // Precompute B^(k-1) mod M: the weight of the leading char we remove when rolling.
    unsigned long long p1 = 1, p2 = 1;
    for (long long i = 0; i < k - 1; i++) { p1 = (p1 * B1) % M1; p2 = (p2 * B2) % M2; }

    vector<unsigned long long> keys;
    keys.reserve(W);

    unsigned long long h1 = 0, h2 = 0;
    // first window s[0..k-1]
    for (long long i = 0; i < k; i++) {
        unsigned long long c = (unsigned long long)(s[i] - 'a' + 1);
        h1 = (h1 * B1 + c) % M1;
        h2 = (h2 * B2 + c) % M2;
    }
    keys.push_back((h1 << 32) ^ h2);

    // roll: window starting at i uses removing s[i-1], appending s[i+k-1]
    for (long long i = 1; i < W; i++) {
        unsigned long long out = (unsigned long long)(s[i - 1] - 'a' + 1);
        unsigned long long in  = (unsigned long long)(s[i + k - 1] - 'a' + 1);
        // remove leading char (weight B^(k-1)), shift left by one, append new char
        h1 = (h1 + M1 - (out * p1) % M1) % M1;
        h1 = (h1 * B1 + in) % M1;
        h2 = (h2 + M2 - (out * p2) % M2) % M2;
        h2 = (h2 * B2 + in) % M2;
        keys.push_back((h1 << 32) ^ h2);
    }

    sort(keys.begin(), keys.end());

    // Count DISTINCT substrings whose group size is >= 2 (appears at >= 2 positions).
    long long ans = 0;
    long long i = 0, m = (long long)keys.size();
    while (i < m) {
        long long j = i;
        while (j < m && keys[j] == keys[i]) j++;
        if (j - i >= 2) ans++;     // one distinct substring, counted once
        i = j;
    }

    cout << ans << "\n";
    return 0;
}
