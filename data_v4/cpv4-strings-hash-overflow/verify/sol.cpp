#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, L;
    if (!(cin >> n >> L)) return 0;
    string s;
    cin >> s;

    // If L > n there are no windows -> echo score 0.
    if (L > n || L <= 0) {
        cout << 0 << "\n";
        return 0;
    }

    int m = n - L + 1; // number of windows

    // Double polynomial hashing to make collisions astronomically unlikely.
    const unsigned long long MOD1 = 1000000007ULL;
    const unsigned long long MOD2 = 998244353ULL;
    const unsigned long long B1 = 131ULL;
    const unsigned long long B2 = 137ULL;

    // Precompute prefix hashes.
    vector<unsigned long long> h1(n + 1, 0), h2(n + 1, 0), p1(n + 1, 1), p2(n + 1, 1);
    for (int i = 0; i < n; i++) {
        unsigned long long c = (unsigned long long)(unsigned char)s[i] + 1ULL;
        h1[i + 1] = (h1[i] * B1 + c) % MOD1;
        h2[i + 1] = (h2[i] * B2 + c) % MOD2;
        p1[i + 1] = (p1[i] * B1) % MOD1;
        p2[i + 1] = (p2[i] * B2) % MOD2;
    }

    auto getHash = [&](int l) -> unsigned long long {
        // hash of window [l, l+L)
        unsigned long long x1 = (h1[l + L] + MOD1 - (h1[l] * p1[L]) % MOD1) % MOD1;
        unsigned long long x2 = (h2[l + L] + MOD2 - (h2[l] * p2[L]) % MOD2) % MOD2;
        return (x1 << 32) ^ x2; // combine two 32-bit-ish hashes into one key
    };

    // Group windows by combined hash, count multiplicities.
    vector<unsigned long long> keys(m);
    for (int i = 0; i < m; i++) keys[i] = getHash(i);
    sort(keys.begin(), keys.end());

    // Echo score = sum over distinct contents of c*(c-1)/2.
    // c can be ~2e5, so c*(c-1)/2 ~ 2e10 and the total can exceed 32-bit range.
    long long answer = 0;
    int i = 0;
    while (i < m) {
        int j = i;
        while (j < m && keys[j] == keys[i]) j++;
        long long c = j - i;
        answer += c * (c - 1) / 2;
        i = j;
    }

    cout << answer << "\n";
    return 0;
}
