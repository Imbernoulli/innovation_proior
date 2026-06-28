#include <bits/stdc++.h>
using namespace std;

const long long MOD = 998244353;        // NTT-friendly prime: 998244353 = 119*2^23 + 1
const long long G = 3;                   // primitive root of 998244353

long long power_mod(long long b, long long e, long long m) {
    long long r = 1 % m; b %= m; if (b < 0) b += m;
    while (e > 0) { if (e & 1) r = (__int128)r * b % m; b = (__int128)b * b % m; e >>= 1; }
    return r;
}

// In-place iterative NTT. n must be a power of two. invert=false: forward, true: inverse.
void ntt(vector<long long>& a, bool invert) {
    int n = (int)a.size();
    for (int i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) swap(a[i], a[j]);
    }
    for (int len = 2; len <= n; len <<= 1) {
        long long w = invert ? power_mod(G, MOD - 1 - (MOD - 1) / len, MOD)
                             : power_mod(G, (MOD - 1) / len, MOD);
        for (int i = 0; i < n; i += len) {
            long long wn = 1;
            for (int k = 0; k < len / 2; k++) {
                long long u = a[i + k];
                long long v = (__int128)a[i + k + len / 2] * wn % MOD;
                a[i + k] = u + v < MOD ? u + v : u + v - MOD;
                a[i + k + len / 2] = u - v >= 0 ? u - v : u - v + MOD;
                wn = (__int128)wn * w % MOD;
            }
        }
    }
    if (invert) {
        long long n_inv = power_mod(n, MOD - 2, MOD);
        for (long long& x : a) x = (__int128)x * n_inv % MOD;
    }
}

// Convolution of A and B modulo MOD, returned as a polynomial of length |A|+|B|-1.
vector<long long> convolve(vector<long long> A, vector<long long> B) {
    if (A.empty() || B.empty()) return {};
    int result_size = (int)A.size() + (int)B.size() - 1;
    int sz = 1;
    while (sz < result_size) sz <<= 1;
    A.resize(sz, 0);
    B.resize(sz, 0);
    ntt(A, false);
    ntt(B, false);
    for (int i = 0; i < sz; i++) A[i] = (__int128)A[i] * B[i] % MOD;
    ntt(A, true);
    A.resize(result_size);
    return A;
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n; long long V;
    if (!(cin >> n >> V)) return 0;

    // f[v] = number of array elements equal to v  (the frequency polynomial coefficients)
    vector<long long> f(V + 1, 0);
    for (int i = 0; i < n; i++) {
        long long x; cin >> x;
        f[x] = (f[x] + 1) % MOD;
    }

    // The count of ordered triples (i,j,k) with a_i + a_j + a_k = s is [x^s] f(x)^3.
    // Compute f^2 then (f^2)*f via NTT; degree grows to at most 3V.
    vector<long long> f2 = convolve(f, f);     // length 2V+1, [x^s] = #ordered pairs summing to s
    vector<long long> f3 = convolve(f2, f);    // length 3V+1, [x^s] = #ordered triples summing to s

    int q; cin >> q;
    string out;
    out.reserve((size_t)q * 7);
    for (int j = 0; j < q; j++) {
        long long s; cin >> s;
        long long ans = 0;
        if (s >= 0 && s < (long long)f3.size()) ans = f3[s];
        out += to_string(ans);
        out += '\n';
    }
    cout << out;
    return 0;
}
