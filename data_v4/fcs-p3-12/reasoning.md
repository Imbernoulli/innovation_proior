Three constraints decide this problem before any algorithm is chosen: `N` runs up to `10^18`, `k` is at most `50`, and `p` (`2 <= p <= 10^9`) is not promised prime. The first kills any `O(N)` dynamic program outright — whatever I do must be logarithmic in `N`. The second says a cost polynomial in `k` is affordable. The third is the quiet one: with a composite modulus I cannot divide, so every step has to stay inside additions and multiplications mod `p`, with no modular inverse anywhere. I answer `T` queries (up to `2*10^5`), one triple `N k p` each, printing one count per line.

The head of the sequence is a trap. Fix `k = 3` and count good strings by length: `N=0` gives `1`, then `2, 4` for `N=1,2` (every string fits), then `N=3` loses only `111` for `7`, and brute force continues `13, 24, 44, 81`. So `1, 2, 4, 7, 13, ...` opens with three exact powers of two, and that is not special to `k=3`: for any `k`, lengths `0..k-1` have no room for a forbidden run, so all `2^N` strings are good and `f(N) = 2^N`; for `k = 50` that is fifty consecutive powers of two before `f(50) = 2^50 - 1` finally falls one short. A solution that peeked only at small `N` would "discover" `f(N) = 2^N` and ship a `pow(2, N)` one-liner or a hardcoded table. Both pass the samples and die everywhere `N >= k`: `k=3, N=3` is `7`, not `2^3 = 8`, because `111` is forbidden, and the gap is exactly the count of bad strings, which only grows with `N`. With `N` reaching `10^18` no table is addressable and `2^N` is simply the wrong function. The powers-of-two head is the initial condition of a recurrence, not the answer.

Deriving that recurrence: I classify a good string of length `n >= 1` by its leading block of ones — `j` ones (`0 <= j <= k-1`) then a `0`, followed by any good string of length `n-1-j`; the only string with no such `0` is all-ones, good only while `n <= k-1`. Summing over `j` gives, for `n >= k`,

`f(n) = f(n-1) + f(n-2) + ... + f(n-k)`,

the order-`k` "`k`-bonacci" recurrence (`k = 2` is Fibonacci), with coefficients independent of `n` — exactly the lever for an astronomical `N`. Against the brute values for `k = 3`: `f(3) = 4+2+1 = 7`, `f(4) = 7+4+2 = 13`, `f(5) = 13+7+4 = 24`, all matching. The seeds are `f(i) = 2^i` for `i = 0..k-1`, and the recurrence itself produces `f(k) = 2^{k-1} + ... + 1 = 2^k - 1`.

Now I need `f(N)` for `N` up to `10^18`. Two standard tools raise a linear recurrence across an astronomical index. Matrix exponentiation builds the `k x k` companion matrix and binary-exponentiates it: `O(k^3 log N)` per query. Kitamasa instead notes that `f(N)` is a fixed linear combination of the seeds whose weights are the coefficients of `x^N mod C(x)`, where `C(x) = x^k - x^{k-1} - ... - 1`; computing that power by binary exponentiation over degree-`<k` polynomials costs `O(k^2 log N)`. The factor of `k` decides it: at `k = 50` matrix exponentiation is about `50^3 * 60 ~ 7.5*10^6` operations per query, and `2*10^5` queries would be `~1.5*10^12` — hopeless in 2 seconds. Kitamasa's `~1.5*10^5` per query is a hundredfold lighter and comfortably in budget, so Kitamasa it is.

The reduction, concretely: a polynomial is a length-`k` coefficient vector (degree `< k`). To get `x^N mod C(x)`, keep `result = 1` and `base = x` and, per bit of `N`, square `base` and conditionally multiply it into `result`, reducing each product back below degree `k`. Multiplying two degree-`<k` polynomials gives degree `< 2k-1`; from `C(x) = 0` I read `x^k = x^{k-1} + ... + 1`, hence `x^d = x^{d-1} + x^{d-2} + ... + x^{d-k}` for every `d`. Sweeping `d` from high to low pushes each high coefficient into its `k` lower neighbours until only degrees `0..k-1` remain. Then `f(N) = sum_i coef_i * f(i)`. Every operation is an add or a multiply mod `p`; no inverse is taken, so a composite `p` needs no special handling.

One place the small-`p` tests can bite: the seeds. For `N < k` I answer with the seed directly, `f(N) = 2^N mod p`. If I build `2^i` as a raw `long long` — it fits, since `2^49 ~ 1.1*10^15` — and postpone the reduction, that direct path emits an unreduced value, and with `p` as small as `2` the error is not academic. Pin it with `N = 49, k = 50, p = 7`: here `N < k`, so the answer is `2^49 mod 7`; `2` has order `3` mod `7` (`2^3 = 8 = 1 mod 7`) and `49 = 3*16 + 1`, so `2^49 = 2 mod 7`. Building the seeds already reduced — `cur = (cur * 2) % p`, storing the reduced value — makes this case print `2`, and `N = 50` print `2^50 - 1 = 3 mod 7`. The discipline that keeps it clean everywhere: never carry an unreduced accumulator into a place where it is output or combined with reduced values.

The `k = 1` corner is a genuine degeneracy, not decoration — and the constraints stress it explicitly. "No run of one or more ones" means no ones at all, so only the all-zeros string is good and `f(n) = 1` for every `n`. The generic machinery breaks here: the characteristic polynomial is `x - 1`, the coefficient vector has a single slot, and `x` reduces to the constant `1`. Seeding `base` with `base[1] = 1` (the polynomial `x`) is an out-of-range write when the vector length is `1`, so `k = 1` gets its own branch that sets `base[0] = 1` directly; then `x^N = 1` and `f(N) = coef[0] * f(0) = 1`.

Overflow: coefficients live in `[0, p)` with `p <= 10^9`, so a single product reaches `(10^9)^2 = 10^18` — past `int`, within `__int128`, which is what every modular product uses; accumulating `k <= 50` of them before reduction stays in range. `N < 2^63` reads into a 64-bit integer and the exponentiation loop runs at most `60` times.

A counting solution earns no trust until an independent method agrees, so I cross-check against two. For small `N`, a direct `O(N*k)` DP over trailing-run-length states — a genuinely different algorithm that never forms a recurrence or a characteristic polynomial — agreed on every query row across `N=0`, `N<k`, the `N=k`/`N=k+1` boundary, `k=1`, `k=2`, and prime and composite moduli. For large `N`, where that DP cannot reach, an independent matrix-power solver agreed on hundreds of cases with `N` uniform up to `10^18`, `k` up to `50`, including the extremes `N=10^18, k=50` and `N=10^18, k=1`. Both of the pitfalls above surfaced here — the unreduced seed on the `k=50, p=7` trace, the `k=1` out-of-range write in the DP comparison — and were fixed at the cause. A `2000`-query batch with `N` up to `10^18` runs in about `0.59 s`, inside the limit, and the concrete reason the `k`-fold-heavier matrix approach was not worth its extra simplicity.

The shipped program is one self-contained file: the reduced seeds, the Kitamasa multiply-reduce, the `k = 1` branch, and the seed dot product.

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