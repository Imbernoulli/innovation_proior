Every constraint here funnels to one number: `n` can be as large as `10^18`. That sits below `2^63 ~ 9.22*10^18`, so `n` itself fits in a signed 64-bit integer — but the instant I square a residue modulo `n`, as any nontrivial factoring routine must, the product reaches `(10^18)^2 = 10^36`, which overflows even 64 bits by seventeen orders of magnitude. So `n` and every residue live in `unsigned long long`, and the one dangerous multiply — `(a*b) mod n` — goes through `__int128`, whose `~1.7*10^38` range swallows `10^36` comfortably. That single primitive is the floor everything stands on: a 64-bit product of two near-`10^18` operands is silent garbage. The other constraint is throughput — with `q` up to 500 queries I have to factor the worst legal `n` five hundred times inside two seconds.

The obvious method is trial division: test `d = 2, 3, 4, ...` up to `sqrt(n)`, peeling each factor as found, and whatever survives above `sqrt(n)` is the last prime. It is trivially correct, so the only open question is whether it survives the limits. The hardest shape the constraints allow is a semiprime `n = p*q` with both primes near `10^9` and `n` near `10^18` — say `p = 999999937`, `q = 999999893`. Trial division finds nothing until `d` reaches `p`, about `10^9` iterations for a single query; at `~10^8`–`10^9` simple iterations per second that is one to ten seconds per number, and there can be 500 of them. Three to four orders of magnitude over budget. Trial division is dead for the large part.

So I need to factor a near-`10^18` semiprime without ever scanning to `sqrt(n) ~ 10^9`. Two tools do exactly that. Miller-Rabin decides primality *without* factoring: writing `n-1 = d*2^s` with `d` odd, a base `a` witnesses compositeness unless `a^d ≡ 1` or some `a^{d*2^i} ≡ -1 (mod n)`. A single random base can be fooled by strong pseudoprimes, but a fixed set of bases is provably deterministic below a threshold, and `{2,3,5,7,11,13,17,19,23,29,31,37}` (the primes up to 37) is exact for all `n < 3.317*10^24`, far above `10^18`. So twelve bases give me primality with no probabilistic doubt. I also short-circuit: if one of those small primes divides `n`, then `n` is prime iff it *equals* that prime — this settles the small cases and keeps `a = n` out of the witness loop. To split a composite, Pollard's rho iterates `x <- x^2 + c (mod n)` and reads a nontrivial factor off `gcd(|x_i - x_j|, n)` once the sequence cycles modulo a hidden prime factor `p` but not yet modulo `n`. Its expected cost is `O(p^{1/4})` where `p` is the *smallest* prime factor — at most `10^9` for any composite `n <= 10^18`, so rho needs `~sqrt(10^9) ~ 3*10^4` steps, not `10^9`.

For the cycle detection I use Brent's variant — one evaluation of `f` per step, comparing the current value against one saved at geometrically growing distances — rather than Floyd's three-per-step, and on top of it the batched-gcd trick: accumulate the product of the differences `|x - y|` modulo `n` across a block of `~128` steps and take a single gcd per block instead of one gcd per step. The batch can overshoot: the accumulated product can itself be divisible by `n`, so the gcd jumps straight to `n` and reports that a collision happened somewhere in the block without saying where. The remedy is to re-walk that block's saved start `ys` one step at a time, taking a gcd each step, until the factor reappears.

Assembling `mulmod`/`powmod`, `isPrime`, a `pollardRho`, and a recursive splitter, I smoke-test `1, 2, 12, 10^18, 999999999999999989`. The first three print instantly and correctly; then the process pins a core at 100% and never returns. A hang is worse than a wrong answer — there is no output to diff — so I bisect the inputs and land on `n = 10^18 = 2^18 * 5^18`. The even part strips instantly, so what actually hangs is the odd cofactor `5^18`.

Nothing is looping infinitely; something is looping a bounded but astronomical number of times. The culprit is Brent's compare-distance `r`, which doubles every outer round while `for (int i = 0; i < r; i++) y = f(y)` does `r` evaluations of `f`:

```
while (d == 1) {
    x = y;
    for (int i = 0; i < r; i++) y = f(y);    // r doubles unboundedly
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
        k += lim;                            // advance by the actual block length
    }
    r <<= 1;                                 // no ceiling: this is the hang
}
```

When an unlucky `(c, x)` never yields a factor, `r` climbs — `2^25 ~ 3*10^7`, `2^30 ~ 10^9` — until one outer iteration is doing a billion `f` calls; and past `2^30` the `int r` even overflows negative, after which both loops become no-ops and it spins forever at no visible per-iteration cost. The root cause is that the loop has no ceiling on `r` to force it to abandon a bad `(c, x)`. The fix is exactly that ceiling: cap `r` at `RCAP = 1 << 20` and break out of the Brent loop when it is reached. Since the smallest prime factor of any composite `n <= 10^18` is at most `10^9`, rho legitimately needs `~3*10^4 << 2^20` steps, so a `2^20` cap never abandons a *solvable* `(c, x)` but does abandon an unlucky one — on break with `d == 1` control falls back to the top of the retry loop, which draws a fresh `c` and `x`.

That rescues the perfect-power case by retrying, but it does not make it fast. Timing a batch, small perfect powers — `25`, `3^8`, `7^6`, `5^11 = 48828125` — each burn 100–400 ms, a full `RCAP`-length attempt or two before a lucky retry splits them. The clean cure is to never ask rho to find small factors at all: I trial-divide by every integer below 1000 first, peeling those factors directly, and hand only the remaining cofactor to rho. After that strip the cofactor has no prime factor below 1000, so the tiny-power villains — and `5^18` itself, since 5 < 1000 — are factored entirely by trial division and never reach rho; the `r`-cap survives only as a safety net for a genuinely unlucky medium-factor draw. The strip loop re-reads `n` as it shrinks and its guard `(u64)p*p <= n` cannot overflow because `p < 1000`.

Re-timing the slow set plus random prime squares with the prime between 1000 and `10^9` (so `p^2` forces rho to crack a real medium prime square), the maximum single-case time falls to 7.5 ms. The adversarial throughput test — 500 maximal semiprimes `p*q` with both primes near `10^9` — factors in 0.18 s, well inside two seconds with the worst input the constraints permit. For code-correctness, a separate claim from idea-correctness, I diff against an independent trial-division oracle in Python on inputs capped at `10^12` where it stays fast: 600 seeds of a mixed generator (tiny numbers, single large primes, two-large-prime semiprimes, prime powers, products of many small primes, medium prime squares, powers of two, and the `1/2/3/4` boundaries) plus explicit edges, 3768 numbers in all, matched the C++ output line-for-line with zero mismatches, and every printed base passed an independent primality check.

The edge shapes fall out of machinery already in place: `n = 1` is special-cased before factoring and prints exactly `1:`; small primes and near-`10^18` primes exit through the small-prime check and Miller-Rabin without a rho call; high prime powers `2^59, 3^37, 5^18` are stripped by trial division and their exponents counted directly. The one output subtlety worth pinning is that each `p^e` token is *prefixed* by a space rather than followed by one, so there is exactly one space after the colon and never a trailing one.

That assembles the whole solver: `mulmod`/`powmod`, the twelve-base `isPrime`, the retrying `pollardRho`, and the trial-strip `factor` feeding the per-query output buffer.
