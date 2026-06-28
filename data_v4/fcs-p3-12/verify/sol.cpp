#include <bits/stdc++.h>
using namespace std;

typedef long long ll;
typedef unsigned long long ull;
typedef __int128 lll;

// Multiply two polynomials a, b (coefficients mod m, degree < k each) and reduce
// modulo the characteristic polynomial of the recurrence
//   f(n) = f(n-1) + f(n-2) + ... + f(n-k),
// i.e. x^k = x^{k-1} + x^{k-2} + ... + 1.
// Returns the product reduced to degree < k, all coefficients in [0, m).
static vector<ll> mulmod(const vector<ll>& a, const vector<ll>& b, int k, ll m) {
    // raw convolution into degree < 2k-1
    vector<ll> c(2 * k - 1, 0);
    for (int i = 0; i < k; i++) {
        if (a[i] == 0) continue;
        for (int j = 0; j < k; j++) {
            if (b[j] == 0) continue;
            // (a[i]*b[j]) mod m via __int128 to avoid overflow
            ll add = (ll)((lll)a[i] * b[j] % m);
            c[i + j] += add;
            if (c[i + j] >= m) c[i + j] -= m;
        }
    }
    // reduce high terms x^d for d from 2k-2 down to k using
    //   x^k = x^{k-1} + ... + x^0, so x^{d} = x^{d-1} + x^{d-2} + ... + x^{d-k}
    for (int d = 2 * k - 2; d >= k; d--) {
        ll coef = c[d];
        if (coef != 0) {
            for (int t = 1; t <= k; t++) {
                int idx = d - t;
                c[idx] += coef;
                if (c[idx] >= m) c[idx] -= m;
            }
            c[d] = 0;
        }
    }
    c.resize(k);
    return c;
}

// Compute x^N mod (characteristic polynomial), return coefficient vector of length k.
static vector<ll> polypow(ll N, int k, ll m) {
    // result = 1 (the polynomial "1"), base = x (if k > 1) else reduce
    vector<ll> result(k, 0), base(k, 0);
    result[0] = 1 % m;
    if (k == 1) {
        // characteristic poly: x = 1, so x reduces to the constant 1; x^N = 1.
        base[0] = 1 % m;
    } else {
        base[1] = 1 % m; // x
    }
    while (N > 0) {
        if (N & 1) result = mulmod(result, base, k, m);
        base = mulmod(base, base, k, m);
        N >>= 1;
    }
    return result;
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int T;
    if (!(cin >> T)) return 0;
    while (T--) {
        ll N, k, p;
        cin >> N >> k >> p;
        // Number of binary strings of length N with no run of k or more
        // consecutive ones, modulo p.
        //
        // Let f(n) = that count. Then f(n) = 2^n for 0 <= n <= k-1, and for
        // n >= k the order-k linear recurrence
        //   f(n) = f(n-1) + f(n-2) + ... + f(n-k)
        // holds. We evaluate f(N) via Kitamasa: compute x^N modulo the
        // characteristic polynomial x^k - x^{k-1} - ... - 1, then dot with the
        // initial values f(0..k-1).

        ll m = p; // modulus (not necessarily prime; Kitamasa needs no inverses)

        // initial values f(0..k-1) = 2^i mod p
        int kk = (int)k;
        vector<ll> init(kk, 0);
        ll cur = 1 % m;
        for (int i = 0; i < kk; i++) {
            init[i] = cur;
            cur = (cur * 2) % m;
        }

        ll ans;
        if (N < k) {
            // f(N) = 2^N directly (still reduce mod p)
            ans = init[(int)N];
        } else {
            // x^N mod char poly, then sum coef[i] * f(i)
            vector<ll> coef = polypow(N, kk, m);
            lll acc = 0;
            for (int i = 0; i < kk; i++) {
                acc += (lll)coef[i] * init[i] % m;
            }
            ans = (ll)(acc % m);
        }
        cout << ans << "\n";
    }
    return 0;
}
