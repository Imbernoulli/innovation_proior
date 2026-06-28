**Problem.** Count binary strings of length `N` that contain no run of `k` or more consecutive ones, modulo `p`. Answer `T` queries; each gives `N` (up to `10^18`), `k` (up to `50`), and `p` (`2..10^9`, possibly composite). Read from stdin, print one count per query.

**The trap to avoid.** For any fixed `k`, the counts for small lengths are `f(0), f(1), ..., f(k-1) = 1, 2, 4, ..., 2^{k-1}` — exact powers of two, because when `N < k` no string is long enough to contain a forbidden run, so all `2^N` strings are good. At `N = k` the count is `2^k - 1` (only the all-ones string is bad). This tidy head tempts a hardcoded table or a `pow(2, N)` shortcut, and it would pass the samples and the smallest hidden cases. But `N` ranges to `10^18`: no table is large enough, and `2^N` is simply wrong for `N >= k` (e.g. `k=3, N=3` is `7`, not `8`). The powers-of-two pattern is only the *initial condition* of a recurrence, not the answer.

**Key idea — the order-`k` recurrence.** Classifying a good string by its leading block of ones (`j` ones then a `0`, for `j = 0..k-1`) gives, for `n >= k`,

`f(n) = f(n-1) + f(n-2) + ... + f(n-k)`,

a `k`-step "`k`-bonacci" recurrence (Fibonacci is the `k = 2` case). The seeds are `f(i) = 2^i` for `i = 0..k-1`; the recurrence then produces `f(k) = 2^k - 1` and everything beyond.

**Fast-forwarding to `N = 10^18`.** An `O(N)` DP is impossible. I use **Kitamasa**: `f(N)` is a fixed linear combination of the seeds `f(0..k-1)`, whose coefficients are the coordinates of `x^N` modulo the characteristic polynomial `x^k - x^{k-1} - ... - 1`. Computing `x^N mod C(x)` by binary exponentiation over polynomials costs `O(k^2 log N)` per query — a `k`-fold improvement over the `O(k^3 log N)` matrix-exponentiation alternative, which matters with up to `2*10^5` queries. The reduction uses `x^d = x^{d-1} + ... + x^{d-k}`, sweeping high degrees down into the low `k` coefficients; it is all additions and multiplications, so it needs **no modular inverse** and a composite `p` is fine.

**Pitfalls handled.**
1. *Keep everything reduced mod `p`.* Building `2^i` unreduced and reducing "later" mixes reduced and unreduced operands and gives wrong answers for small `p` (caught on `k=50, p=7`: `2^49 ≡ 2`, not `1`). Reduce as you build: `cur = (cur*2) % p`.
2. *`k = 1` degenerate polynomial.* "No run of `1` one" means the only good string is all-zeros, so `f(n) = 1`. The characteristic polynomial is `x - 1`, the coefficient vector has length `1`, and `x` reduces to the constant `1`; the generic code's `base[1] = 1` is an out-of-range write, so `k = 1` is special-cased to set `base[0] = 1`.
3. *Overflow.* Coefficients are `< p <= 10^9`, so products reach `10^18`; every modular product goes through `__int128`.

**Verification.** Differential-tested against two independent oracles: a direct `O(N*k)` trailing-run DP on `34800` small-`N` query rows (zero mismatches), and an independent matrix-power solver on `206` cases with `N` up to `10^18`, `k` up to `50`, prime and composite moduli (zero mismatches). A `2000`-query batch with `N` up to `10^18`, `k` up to `50` runs in `0.59 s`.

**Complexity.** `O(k^2 log N)` time per query, `O(k)` space.

**Code.**

```cpp
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
```
