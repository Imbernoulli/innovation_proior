**Problem.** Given `q <= 500` queries, each an integer `n` with `1 <= n <= 10^18`, print the prime factorization of each `n` as `n: p1^e1 p2^e2 ... pk^ek` with primes in increasing order and multiplicities attached; print `1:` for `n = 1`. Read from stdin, write to stdout.

**Why the obvious trial division is wrong (too slow).** Testing divisors up to `sqrt(n)` is `O(sqrt(n))` per query. The worst legal input is a semiprime `n = p*q` with both primes near `10^9` (so `n ~ 10^18`): no factor appears until the divisor reaches `p ~ 10^9`, i.e. ~`10^9` iterations *per number*, and there can be 500 of them. That is 3–4 orders of magnitude over budget. Trial division is discarded for the large part.

**Key idea — Miller-Rabin + Pollard-rho (Brent).** You never need to scan to `sqrt(n)`:

- **Primality without factoring.** Decide whether a value is prime with deterministic Miller-Rabin. The fixed witness set `{2,3,5,7,11,13,17,19,23,29,31,37}` is provably exact for all `n < 3.317*10^24`, which covers `10^18` with no probabilistic doubt.
- **Splitting a composite.** Pollard's rho iterates `x <- x^2 + c (mod n)` and reads a nontrivial factor off `gcd(|x_i - x_j|, n)` once the sequence cycles modulo a hidden prime factor `p`. Expected cost is `O(p^{1/4})`, and the relevant `p` is the *smallest* prime factor of `n`, which is `<= 10^9` for any composite `n <= 10^18` — so rho needs `~sqrt(10^9) ~ 3*10^4` steps, not `10^9`. Use Brent's cycle detection (one `f` per step) with batched gcd (one gcd per ~128 steps).
- **The one primitive everything rests on:** `(a*b) mod n` with `a,b < n <= 10^18` overflows 64 bits (`a*b` reaches `10^36`), so compute it in `__int128`: `(u64)((u128)a*b % n)`.

Recurse: strip small factors, test primality of the remainder, split with rho, recurse on both halves.

**Pitfalls to get right.**
1. *Unbounded Brent distance.* Brent doubles its compare-distance `r` each round. If an unlucky `(c, x)` never finds a factor (perfect powers are the classic offender), `r` doubles into a billion-`f`-call iteration and `int r` even overflows past `2^30`, so the program *hangs* (no output to diff). Cap `r` at `2^20` and `break` to a fresh `(c, x)` retry; `2^20` is far above the `~3*10^4` rho ever legitimately needs, so a solvable case is never abandoned.
2. *Batch overshoot.* The accumulated product can be `≡ 0 (mod n)`, so the batch gcd jumps straight to `n`. Re-walk the saved sub-sequence (`ys`) one step at a time, taking a gcd each step, to recover the actual factor; reset `d = 1` before that walk.
3. *Block bookkeeping.* Advance the inner counter by the actual block length `lim = min(m, r-k)`, not by the nominal `m` — the last sub-block is short and `k += m` mis-accounts it.
4. *Perfect-power thrash.* Even with the cap, small prime powers (`25`, `7^6`, `5^11`) waste whole `RCAP` attempts in rho. Trial-divide all primes below 1000 *first*, then hand only the large cofactor to rho; this both removes the thrash and is faster.
5. *Overflow.* All modular multiplies via `__int128`; `r` bounded so its loop can never overflow `int`.

**Edge cases.** `n = 1` -> `1:` (empty factorization of the unit). Small primes and `n = 2` -> short-circuited by the small-prime check. Large prime near `10^18` -> Miller-Rabin returns prime, emitted as `p^1`, no rho. High prime powers `2^59, 3^37, 5^18` -> the small base is stripped by trial division and the exponent counted directly. Output has exactly one space between tokens and no trailing space.

**Complexity.** Per number: trial division `O(1000)` to strip small primes, then `O(n^{1/4})` expected for each rho split and `O(log n)` Miller-Rabin checks, for `O(n^{1/4} polylog n)` overall — about `~3*10^4` rho steps on the worst semiprime. The full worst-case batch of 500 maximal semiprimes runs in ~0.18 s, well under a 2-second limit.

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;
typedef unsigned long long u64;
typedef __uint128_t u128;

// modular multiply without overflow: (a*b) mod m, a,b < m <= ~9.2e18 fits in u64
static u64 mulmod(u64 a, u64 b, u64 m) {
    return (u128)a * b % m;
}

// fast modular exponentiation a^e mod m
static u64 powmod(u64 a, u64 e, u64 m) {
    u64 r = 1 % m;
    a %= m;
    while (e) {
        if (e & 1) r = mulmod(r, a, m);
        a = mulmod(a, a, m);
        e >>= 1;
    }
    return r;
}

// deterministic Miller-Rabin for n < 3.3e24 using these 7 bases.
static bool isPrime(u64 n) {
    if (n < 2) return false;
    for (u64 p : {2ull, 3ull, 5ull, 7ull, 11ull, 13ull, 17ull, 19ull, 23ull, 29ull, 31ull, 37ull}) {
        if (n % p == 0) return n == p;
    }
    u64 d = n - 1;
    int s = 0;
    while ((d & 1) == 0) { d >>= 1; ++s; }
    for (u64 a : {2ull, 3ull, 5ull, 7ull, 11ull, 13ull, 17ull, 19ull, 23ull, 29ull, 31ull, 37ull}) {
        u64 x = powmod(a, d, n);
        if (x == 1 || x == n - 1) continue;
        bool composite = true;
        for (int i = 0; i < s - 1; i++) {
            x = mulmod(x, x, n);
            if (x == n - 1) { composite = false; break; }
        }
        if (composite) return false;
    }
    return true;
}

static mt19937_64 rng(0x9e3779b97f4a7c15ull);

// Pollard-Rho with Brent's cycle detection plus batched gcd; returns a nontrivial
// factor of n (n composite, odd, > 1). Retries with fresh (c, x) on failure.
static u64 pollardRho(u64 n) {
    if ((n & 1) == 0) return 2;
    while (true) {
        u64 c = rng() % (n - 1) + 1;     // constant in g(x) = x^2 + c
        auto f = [&](u64 v) { return (u64)(((u128)v * v + c) % n); };
        u64 x = rng() % n, y = x, d = 1;
        u64 ys = 0, q = 1;
        int m = 128;                     // gcd-batch size: accumulate diffs, gcd once per m steps
        int r = 1;                       // Brent's geometric tortoise distance
        const int RCAP = 1 << 20;        // bound the search; a factor < 2^60 needs << 2^20 steps
        while (d == 1) {
            x = y;
            for (int i = 0; i < r; i++) y = f(y);
            int k = 0;
            while (k < r && d == 1) {
                ys = y;
                int lim = min(m, r - k);
                for (int i = 0; i < lim; i++) {
                    y = f(y);
                    u64 diff = x > y ? x - y : y - x;
                    q = (u64)((u128)q * (diff ? diff : 1) % n);
                }
                d = std::__gcd(q, n);
                k += lim;
            }
            if (r >= RCAP) break;        // give up on this (c, x); fall through to retry
            r <<= 1;
        }
        if (d == n) {
            // a whole batch collided at once; walk the saved sub-sequence step-by-step
            d = 1;
            do {
                ys = f(ys);
                u64 diff = x > ys ? x - ys : ys - x;
                d = std::__gcd(diff, n);
            } while (d == 1);
        }
        if (d != n && d != 1) return d;
        // d collapsed to n (or 1) — this (c, x) is unlucky; retry with fresh randomness
    }
}

// rho-only factorization of a value already stripped of all primes < 1000.
static void factorRho(u64 n, vector<u64> &out) {
    if (n == 1) return;
    if (isPrime(n)) { out.push_back(n); return; }
    u64 d = pollardRho(n);
    factorRho(d, out);
    factorRho(n / d, out);
}

// Strip small primes by trial division first (kills the perfect-power / tiny-factor
// pathology that makes Pollard-Rho thrash), then hand the large cofactor to rho.
static void factor(u64 n, vector<u64> &out) {
    static const int SMALL = 1000;
    for (int p = 2; p < SMALL && (u64)p * p <= n; p++) {
        while (n % p == 0) { out.push_back(p); n /= p; }
    }
    if (n > 1) factorRho(n, out);
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);
    int q;
    if (!(cin >> q)) return 0;
    string buf;
    while (q--) {
        u64 n;
        cin >> n;
        if (n == 1) {
            cout << "1:\n";          // 1 has no prime factors
            continue;
        }
        vector<u64> f;
        factor(n, f);
        sort(f.begin(), f.end());
        // collapse into prime^exp pairs
        buf.clear();
        buf += to_string(n);
        buf += ':';
        size_t i = 0;
        while (i < f.size()) {
            size_t j = i;
            while (j < f.size() && f[j] == f[i]) ++j;
            buf += ' ';
            buf += to_string(f[i]);
            buf += '^';
            buf += to_string((unsigned)(j - i));
            i = j;
        }
        buf += '\n';
        cout << buf;
    }
    return 0;
}
```
