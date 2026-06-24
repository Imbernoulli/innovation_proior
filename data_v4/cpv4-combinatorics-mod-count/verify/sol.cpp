#include <bits/stdc++.h>
using namespace std;

const long long MOD = 1000000007LL;

long long power_mod(long long b, long long e, long long m) {
    b %= m; if (b < 0) b += m;
    long long r = 1 % m;
    while (e > 0) {
        if (e & 1) r = (__int128)r * b % m;
        b = (__int128)b * b % m;
        e >>= 1;
    }
    return r;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    long long n, k, c;
    if (!(cin >> n >> k >> c)) return 0;

    // Count multisets of size k drawn from n colors, each color used at most c times, mod p.
    // Inclusion-exclusion over the set of colors forced to exceed the cap:
    //   answer = sum_{j>=0} (-1)^j * C(n, j) * C(n + (k - j*(c+1)) - 1, n - 1).
    // The term for index j is real only while r_j = k - j*(c+1) >= 0; once r_j < 0 it and
    // every larger j contribute 0 (the binomial top n+r_j-1 drops below n-1).

    // Largest binomial top ever requested is n + k - 1, so factorials up to n + k suffice.
    long long N = n + k + 5;
    vector<long long> fact(N + 1), inv_fact(N + 1);
    fact[0] = 1;
    for (long long i = 1; i <= N; i++) fact[i] = (__int128)fact[i - 1] * i % MOD;
    inv_fact[N] = power_mod(fact[N], MOD - 2, MOD);
    for (long long i = N; i >= 1; i--) inv_fact[i - 1] = (__int128)inv_fact[i] * i % MOD;

    auto C = [&](long long a, long long b) -> long long {
        if (b < 0 || a < 0 || b > a) return 0;
        return (__int128)fact[a] % MOD * inv_fact[b] % MOD * inv_fact[a - b] % MOD;
    };

    // Number of ways to put r indistinguishable balls into m distinguishable bins (each >= 0).
    // Stars and bars = C(m + r - 1, m - 1) for m >= 1; for m = 0 it is 1 iff r = 0, else 0.
    auto bars = [&](long long m, long long r) -> long long {
        if (m == 0) return (r == 0) ? 1 : 0;
        return C(m + r - 1, m - 1);
    };

    long long ans = 0;
    for (long long j = 0; ; j++) {
        if (j > n) break;                 // cannot force more than n colors over the cap
        long long r = k - j * (c + 1);    // size remaining after over-filling j colors
        if (r < 0) break;                 // r only decreases as j grows, so stop
        long long term = (__int128)C(n, j) % MOD * bars(n, r) % MOD;
        if (j & 1) ans = (ans - term % MOD + MOD) % MOD;
        else ans = (ans + term) % MOD;
    }

    cout << ans % MOD << "\n";
    return 0;
}
