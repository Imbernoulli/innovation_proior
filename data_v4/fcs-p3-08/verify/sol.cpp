#include <bits/stdc++.h>
using namespace std;

const long long MOD = 1000000007LL;

long long power_mod(long long base, long long exp, long long mod) {
    long long result = 1 % mod;
    base %= mod;
    if (base < 0) base += mod;
    while (exp > 0) {
        if (exp & 1) result = (__int128)result * base % mod;
        base = (__int128)base * base % mod;
        exp >>= 1;
    }
    return result;
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int q;
    if (!(cin >> q)) return 0;

    vector<int> as(q), bs(q);
    int maxN = 0;
    for (int i = 0; i < q; i++) {
        cin >> as[i] >> bs[i];
        maxN = max(maxN, as[i] + bs[i]);
    }

    // Precompute factorials and inverse factorials up to maxN (= a+b, up to 2*10^6).
    // C(a+b, a) mod p via fact[a+b] * invfact[a] * invfact[b] mod p, O(a+b) total.
    vector<long long> fact(maxN + 1), invfact(maxN + 1);
    fact[0] = 1 % MOD;
    for (int i = 1; i <= maxN; i++) fact[i] = (__int128)fact[i - 1] * i % MOD;
    invfact[maxN] = power_mod(fact[maxN], MOD - 2, MOD);
    for (int i = maxN; i >= 1; i--) invfact[i - 1] = (__int128)invfact[i] * i % MOD;

    string out;
    out.reserve(q * 12);
    for (int i = 0; i < q; i++) {
        int a = as[i], b = bs[i];
        long long c = (__int128)fact[a + b] * invfact[a] % MOD * invfact[b] % MOD;
        out += to_string(c);
        out += '\n';
    }
    cout << out;
    return 0;
}
