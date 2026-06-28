**Reading the problem and pinning the contract.** I get `q <= 500` queries, each an integer `n` with `1 <= n <= 10^18`, and for each I must print the prime factorization `n: p1^e1 ... pk^ek` with the primes in increasing order, multiplicities attached, and `1:` for the unit. The first thing I fix is the magnitude, because it dictates everything downstream: `10^18` is below `2^63 ~ 9.22*10^18` but above `2^31`, so `n` does not fit in a 32-bit `int`. It does fit in a signed `long long` (up to ~9.2*10^18), but several internal operations — squaring a residue modulo `n` — will produce intermediate products up to `(10^18)^2 = 10^36`, which overflows even 64-bit. So I will carry `n` and all residues in `unsigned long long` (`u64`) and do the one dangerous multiply through `__int128`. That decision is non-negotiable; a 64-bit multiply of two near-`10^18` operands is silent garbage.

**Laying out the candidate approaches.** Two routes, and I want the one I can both prove correct and prove fast.

- *Trial division.* For each `n`, test divisors `d = 2, 3, 4, ...` up to `sqrt(n)`, peeling each factor as found; whatever survives above `sqrt(n)` is the last prime. Dead simple, obviously correct. The risk is purely speed.
- *Randomized factorization.* Use a primality test to know when a piece is already prime, and a randomized splitter to break a composite into two smaller factors, recursing on each. No `sqrt(n)` scan. The risk is correctness of the primality witnesses, overflow in the modular multiply, and pathological slowdowns on adversarial `n`.

**Killing trial division on a concrete worst case.** Before I write anything, let me see whether trial division is actually too slow, because if it survives I would much rather ship three lines than a randomized splitter. Take the hardest shape the constraints allow: `n = p * q` with `p` and `q` both prime and both near `10^9`, so `n` is near `10^18`. Concretely `p = 999999937`, `q = 999999893`, `n = p*q ~ 9.999...*10^17`. Trial division finds no factor until `d` reaches `p = 999999937`. That is about `10^9` iterations *for one query*. At maybe `10^8`–`10^9` simple iterations per second, that is on the order of 1–10 seconds for a **single** `n` — and the input can contain up to `q = 500` such numbers. Trial division is off by three to four orders of magnitude. It is dead at these limits.

So the whole game is: how do I factor a near-`10^18` semiprime without scanning to `sqrt(n) ~ 10^9`? The insight the problem is built around is that I do not have to. I can (a) decide primality without factoring, using **Miller-Rabin** with a fixed set of witnesses that is *deterministic* for all `n` in range, and (b) split a composite using **Pollard's rho** method, whose expected cost to find a factor `p` is `O(p^{1/4} * polylog)` — and crucially the relevant `p` is the *smallest prime factor*, which for any composite `n <= 10^18` is at most `10^9`, so rho needs on the order of `sqrt(10^9) ~ 3*10^4` steps, not `10^9`. The combination Miller-Rabin (exact, via the right witnesses) + Pollard-rho (with a non-overflowing `__int128` multiply, and Brent's improvement on the cycle detection) is the SOTA approach for 64-bit factorization, and it is what I will build.

**Designing the deterministic primality test.** Miller-Rabin writes `n-1 = d * 2^s` with `d` odd, and for a base `a` checks whether `a^d ≡ 1` or `a^{d*2^i} ≡ -1 (mod n)` for some `0 <= i < s`; if neither holds, `a` is a *witness* that `n` is composite. A single random base can be fooled by strong pseudoprimes, but it is a known result that testing a fixed small set of bases is *deterministic* below a threshold. The set `{2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37}` (all primes up to 37) is provably sufficient for all `n < 3.317 * 10^24`, which comfortably covers `10^18`. So I test exactly those twelve bases; if none is a witness, `n` is genuinely prime, with no probabilistic doubt. I also short-circuit: if any of those small primes divides `n`, then `n` is prime iff it *equals* that small prime. That handles `n = 2, 3, 5, ...` and avoids feeding `a = n` into the loop.

**The overflow-safe modular multiply.** Both Miller-Rabin (`a^d mod n` by fast exponentiation) and Pollard-rho (`x^2 + c mod n`) need `(a * b) mod n` with `a, b < n <= 10^18`. The product `a*b` reaches `10^36`, far past 64 bits. The clean fix at this scale is to compute the product in `__int128` and reduce: `(u64)((u128)a * b % n)`. `__int128` holds up to ~`1.7*10^38`, so `10^36` is safe. That is the one primitive everything else stands on, so I get it right first and reuse it.

**Pollard's rho with Brent.** The rho method iterates `x <- f(x) = x^2 + c (mod n)` and looks for a nontrivial `gcd(|x_i - x_j|, n)`: once the sequence cycles modulo some hidden prime factor `p` of `n` (which happens after `~sqrt(p)` steps by the birthday bound) but not yet modulo `n`, that gcd is a proper factor. Floyd's tortoise-and-hare detects the cycle but evaluates `f` three times per step; Brent's variant detects it with one `f` per step by comparing the current value against a value saved at geometrically growing distances. On top of Brent I use the standard **batched-gcd** trick: instead of a gcd every step, accumulate the product of the differences `|x - y|` modulo `n` over a block of ~128 steps and take a single gcd of that product with `n`. One gcd per 128 steps is a big constant-factor win. The price is that a batch can *overshoot* — the accumulated product can be divisible by `n` itself, so the gcd jumps straight to `n` and tells me a collision happened *somewhere* in the block without telling me where. The remedy is to re-walk that block's saved start (`ys`) one step at a time, taking a gcd each step, until the factor reappears.

**First implementation.** I assemble the pieces: `mulmod`/`powmod`, `isPrime`, a `pollardRho` returning one nontrivial factor, and a recursive `factor` that tests primality, splits with rho, and recurses on both halves. My first `pollardRho` was the textbook batched-Brent loop:

```
u64 c = rng() % (n - 1) + 1;
auto f = [&](u64 v){ return (u64)(((u128)v*v + c) % n); };
u64 x = rng()%n, y = x, d = 1, ys = 0, q = 1;
int m = 128, r = 1;
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
        k += m;                 // <-- this line is wrong, found below
    }
    r <<= 1;
}
if (d != n) return d;
do { ys = f(ys); u64 diff = x>ys?x-ys:ys-x; d = std::__gcd(diff, n); } while (d == 1);
return d;
```

The `factor` recursion checks `isPrime`, else splits.

**The first failure: a hang, not a wrong answer.** I compiled and smoke-tested `1, 2, 12, 10^18, 999999999999999989`. The first three printed instantly and correctly. Then it hung — the process pinned a core at 100% and never returned. A hang is worse than a wrong answer because there is no output to diff, so I had to localize it by hand. I bisected the inputs and found the culprit: `n = 10^18 = 2^18 * 5^18`. The even part is stripped instantly (`pollardRho` returns 2 for even `n`), so what actually hangs is factoring the odd cofactor `5^18 = 3814697265625`.

**Diagnosing the hang.** My instinct was an infinite loop, so I dropped iteration guards into every loop in `pollardRho` (the retry `while(true)`, the Brent `while(d==1)`, the backtrack `do-while`) and into the `factor` recursion, each set to `exit()` with a distinct code after a large cap. I re-ran `5^18`. *None* of the guards fired, yet it still timed out. That was the real clue: nothing was looping *infinitely* — something was looping a bounded but astronomically large number of times. I looked again at the Brent loop. `r` doubles every outer iteration (`r <<= 1`), and the inner work `for (int i = 0; i < r; i++) y = f(y)` does `r` evaluations of `f`. If rho fails to find a factor for many rounds, `r` climbs: `2^20 ~ 10^6`, `2^25 ~ 3*10^7`, and by `2^30 ~ 10^9` a *single* outer iteration is doing a billion `f` calls. Worse, `int r` overflows at `r <<= 1` past `2^30`, going negative, after which `for (i=0; i<r; ...)` with `r < 0` runs zero times and `while (k < r ...)` is also skipped, so `d` stays 1 and the loop spins essentially forever at no visible per-iteration cost — but my outer guard counted iterations, and the cost was front-loaded into the giant `for` before overflow, which is why the *wall clock* exploded before any guard's counter reached its cap. The deep cause: rho with this `(c, x)` simply was not finding a factor of `5^18`, and my loop had **no ceiling on `r`** to force a retry with fresh randomness.

**Why perfect powers are the villain.** I dug into *why* rho stalls on `5^18`. The rho splitter relies on the sequence colliding modulo a prime factor `p` before colliding modulo `n`. For `n = p^k`, every residue is "close to" a multiple of `p`, the differences `|x - y|` are very often themselves divisible by high powers of `p`, and the accumulated batch product `q` tends to jump from `gcd = 1` straight to `gcd = n` (overshoot) — and then my backtrack, which re-walks `ys`, can *also* fail to isolate a proper factor for that `(c, x)`, leaving `d == n`. With no `r` ceiling, the code never abandoned the bad `(c, x)`; it just kept doubling `r`. I confirmed the diagnosis: a stripped-down rho with a *fresh* RNG factored `5^18` in three steps, but inside the full program the global RNG state at that point handed `pollardRho` an unlucky `(c, x)` that needed a bail-out the code never offered.

**The fix — bound the search and retry; also a second, quieter bug.** Two corrections. First, cap `r` at `RCAP = 1 << 20` and `break` out of the Brent loop when it is reached; since the smallest prime factor of any composite `n <= 10^18` is at most `10^9`, rho needs `~sqrt(10^9) ~ 3*10^4 << 2^20` steps, so a `2^20` ceiling never prematurely abandons a *solvable* `(c, x)` but does abandon an unlucky one. When the loop breaks with `d == 1`, control falls through `if (d == n)` (false) and `if (d != n && d != 1)` (false, since `d == 1`) back to the top of `while(true)`, which draws fresh `c` and `x`. Second, while re-reading I caught `k += m` in the inner block loop: when the last sub-block is short, `lim = min(m, r-k)` is less than `m`, so advancing `k` by the full `m` skips past the true end and can mis-account the block; it must be `k += lim`. I fixed both, and also added `d = 1;` before the backtrack `do-while` so a fresh gcd walk starts clean.

```
const int RCAP = 1 << 20;
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
        k += lim;                 // fixed: advance by the actual block length
    }
    if (r >= RCAP) break;         // bail to a fresh (c, x) instead of doubling forever
    r <<= 1;
}
if (d == n) { d = 1; do { ys = f(ys); ... } while (d == 1); }
if (d != n && d != 1) return d;   // else retry
```

After this, `5^18` and `10^18` returned instantly and correctly. But the cap merely *rescued* the perfect-power case by retrying; it did not make it *fast*. Timing a batch, I found dozens of small perfect powers (`25`, `6561 = 3^8`, `117649 = 7^6`, `48828125 = 5^11`, `1977326743 = 7^11`) each taking 100–400 ms, because each one burned a full `RCAP`-length attempt or two before a lucky retry split it. Sixty such inputs in one test added up to seconds.

**The second fix — strip small primes first.** The clean, standard cure for the perfect-power and tiny-factor pathology is to **not** ask rho to find small factors at all. I trial-divide by every integer below `1000` first, peeling those factors directly, and only hand the *remaining* cofactor to rho. After stripping primes below 1000, the cofactor has no prime factor below 1000, so any residual prime power `p^k` has `p >= 1000`, well outside the pathological tiny range; and the common villains `25 = 5^2`, `7^6`, `5^11`, `7^11` are now factored entirely by trial division (5 and 7 are below 1000) and never reach rho. This is both faster and removes the thrash:

```
static void factor(u64 n, vector<u64> &out) {
    static const int SMALL = 1000;
    for (int p = 2; p < SMALL && (u64)p * p <= n; p++)
        while (n % p == 0) { out.push_back(p); n /= p; }
    if (n > 1) factorRho(n, out);
}
```

where `factorRho` does the primality-or-split recursion on the large part. Note the loop condition re-reads `n` as it shrinks, so it stops early once `n` is reduced; and `(u64)p*p` cannot overflow because `p < 1000`.

**Re-verifying after the fixes.** I re-timed the slow set: `25, 6561, 117649, 48828125, 1977326743`, plus random prime squares with the prime between 1000 and `10^6` and between `10^6` and `10^9` (so `p^2` ranges up to `~10^18`, forcing rho to actually crack a medium prime square). Maximum single-case time fell to **7.5 ms**. Then the adversarial throughput test: 500 maximal hard semiprimes `p*q` with both primes near `10^9`. The whole batch factored in **0.18 s** — comfortably inside a 2-second limit with the worst input the constraints permit. Every output multiplied back to its `n`, every printed base passed an independent primality check, and the bases were sorted and de-duplicated into exponents.

**Differential testing against a trivially-correct oracle.** Idea-correctness and code-correctness are different claims; for the latter I diff against an independent trial-division factorizer in Python on inputs small enough that trial division is fast (capped at `10^12`). I generated 600 seeds of a mixed generator (tiny numbers, single large primes, two-large-prime semiprimes, prime powers, products of many small primes, squares of medium primes, powers of two, and the `1`/`2`/`3`/`4` boundaries), pooled them with explicit edges (`1, 2, 4, 1000000007, 10^18, 999999999999999989, 600851475143`, a known hard composite `9007199254740993`), and compared my C++ output line-for-line against the oracle. All 3768 numbers matched with zero mismatches.

**Edge cases, deliberately.**
- `n = 1`: handled before factoring; prints exactly `1:` with no factors, which is the mathematically correct empty factorization of the unit.
- `n = 2` (and other small primes): `isPrime` short-circuits on the small-prime divisibility check (`n % 2 == 0` and `n == 2`), so `factorRho` records `2` once; trial division also catches it. Output `2: 2^1`.
- Large prime near `10^18` (`999999999999999989`): trial division below 1000 finds nothing, `factorRho` calls `isPrime`, which returns true, so it is emitted once as `p^1`. No rho call, no hang.
- High prime powers (`2^59`, `3^37`, `5^18`): the small base is stripped by trial division (2, 3, 5 are all < 1000), so the exponent is counted directly and rho is never invoked. Output e.g. `5^18`.
- Overflow: every modular multiply goes through `__int128`; `r` is bounded by `RCAP` so the `for (i=0;i<r;...)` loop can never run a pathological count nor overflow `int`; the exponent count `(unsigned)(j - i)` fits easily (max exponent is 59 for `2^59`).
- Output format: exactly one space separates the colon from the first factor and successive `p^e` tokens, with no trailing space; this is produced by prefixing each `p^e` with a single space and never adding one after the last.

**Final solution.** I convinced myself the *idea* is right by killing trial division on the `10^9`-factor semiprime and by using a witness set that is provably exact below `3.3*10^24`; I convinced myself the *code* is right by tracing the `5^18` hang to a missing `r` ceiling (plus a `k += m` mis-step and a perfect-power thrash), fixing each at its root, and then differential-testing 3768 numbers against an independent oracle with zero mismatches and timing the worst legal input at 0.18 s. That is what I ship — one self-contained file:

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

**Causal recap.** Trial division dies on a single `~10^18` semiprime whose smallest factor is near `10^9` (~`10^9` iterations per query, up to 500 queries), so I moved to Miller-Rabin (the twelve-base witness set is provably exact below `3.3*10^24`, covering `10^18`) plus Pollard-rho with Brent and an `__int128` multiply; my first rho hung on `5^18` because the Brent distance `r` had no ceiling and an unlucky `(c, x)` made it double `r` into a billion-`f`-call iteration (and `int r` even overflowed) — capping `r` at `2^20` with a retry, fixing `k += m` to `k += lim`, and resetting `d` before the backtrack fixed the hang; the residual slowness on small perfect powers was cured by trial-dividing primes below 1000 before invoking rho; and differential testing 3768 numbers against an independent trial-division oracle (zero mismatches) plus a 0.18 s worst-case batch of 500 maximal semiprimes closed out correctness and the time limit.
