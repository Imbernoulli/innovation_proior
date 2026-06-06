# Context: deciding primality of a large integer quickly and reliably

## Research question

Given an odd integer `n` with hundreds or thousands of bits, decide whether `n` is prime — fast, and with a controllable, vanishingly small chance of being wrong. The pain point is sharp. Cryptographic key generation (RSA, Diffie-Hellman) needs to draw random primes of, say, 1024 or 2048 bits; a candidate is sampled and must be tested for primality before anything else can proceed, and this happens constantly. The naive test — trial division by all integers up to `sqrt(n)` — costs on the order of `sqrt(n)` operations, which is exponential in the bit-length `log n`. For a 1024-bit `n` that is `2^512` divisions: utterly hopeless. A usable test must run in time polynomial in `log n`.

Two further constraints shape what "a solution" must look like. First, it must not require factoring `n` — factoring is believed hard, and primality should be decidable without it. Second, because the test will be run on adversarially or randomly chosen inputs billions of times, any failure mode that a whole *class* of composites can exploit is unacceptable: a test that is fooled by a structured family of composite numbers is not reliable, even if it is fast.

## Background

The starting point is **Fermat's little theorem (FLT)**: if `n` is prime and `gcd(a, n) = 1`, then `a^(n-1) ≡ 1 (mod n)`. The left side is computable cheaply: **binary (square-and-multiply) modular exponentiation** computes `a^(n-1) mod n` in `O(log n)` modular multiplications by scanning the bits of the exponent, squaring the running base each step and multiplying in `a` when the bit is set. So FLT gives a fast *necessary* condition for primality, and its contrapositive gives a compositeness proof: if `a^(n-1) != 1 (mod n)` for some `a`, then `n` is composite. Such an `a` is called a **Fermat witness**; an `a` with `a^(n-1) ≡ 1` is a **Fermat nonwitness** (a "Fermat liar" when `n` is in fact composite).

The structural fact that governs how good this test is: for fixed `n`, the Fermat nonwitnesses `{a : a^(n-1) ≡ 1 (mod n)}` form a subgroup of the multiplicative group of units `(Z/nZ)*`. A subgroup is either the whole group or at most half of it. So if the nonwitness subgroup is *proper*, at least half of all `a` are witnesses, and a few random trials catch a composite with high probability.

The catastrophe is that the nonwitness subgroup is **not always proper**. A **Carmichael number** is a composite `n` for which `a^(n-1) ≡ 1 (mod n)` for *every* `a` coprime to `n`. The smallest is `561 = 3·11·17`; `1105` and `1729` follow. For a Carmichael number the Fermat nonwitnesses are the *entire* unit group, so the only Fermat witnesses are the `a` sharing a factor with `n` — a vanishing fraction. Worse, Carmichael numbers are not rare curiosities: there are infinitely many of them. On a Carmichael `n` the Fermat test, run with random coprime bases, reports "probably prime" essentially always. The test has a class of composites it cannot reliably detect.

Two pieces of number theory point at the fix. First, the **structure of square roots of 1**. Modulo a prime `n`, the equation `x^2 ≡ 1` has only the solutions `x ≡ ±1`: from `x^2 - 1 = (x-1)(x+1) ≡ 0 (mod n)` and `n` prime, `n` must divide one of the factors. Modulo a composite `n` with at least two distinct prime factors, the Chinese remainder theorem (CRT) manufactures *extra* square roots of 1 — for instance modulo `15`, the values `1, 4, 11, 14` all square to `1`. A square root of 1 other than `±1` is therefore a *certificate of compositeness*. Second, the **decomposition `n - 1 = 2^e · k`** with `k` odd (and `e ≥ 1` because `n` is odd). This lets `a^(n-1)` be written as `(a^k)` raised to `2^e`, i.e. as the last term of the chain `a^k, a^(2k), a^(4k), …, a^(2^(e-1)k), a^(2^e k) = a^(n-1)`, where each term is the square of the one before. The polynomial identity `x^(2^e k) - 1 = (x^k - 1)(x^k + 1)(x^(2k) + 1)(x^(4k) + 1) \cdots (x^(2^(e-1)k) + 1)` makes this chain the natural place to look for an unexpected square root of 1.

The group-theoretic background needed to *quantify* reliability: `(Z/nZ)*` has order `ϕ(n)`, and for composite `n` we have `ϕ(n) < n - 1` (some residues are not coprime to `n`). The standard lever for a >50% guarantee is to exhibit the set of nonwitnesses inside a *proper* subgroup; sharpening to >75% requires showing the index of that subgroup is at least 4. CRT, the structure of `(Z/p^α Z)*` (cyclic, of order `p^(α-1)(p-1)`), and the fact that a Carmichael number has at least three distinct prime factors are the tools that carry that sharpening.

## Baselines

**Trial division.** Test divisibility of `n` by `2, 3, 5, …, ⌊sqrt(n)⌋`. Correct and deterministic, but `Θ(sqrt(n))` = exponential in `log n`. Useful only as a cheap pre-filter (strip out small prime factors) before a real test. Gap: not polynomial-time.

**Fermat test.** Pick random bases `a`, check `a^(n-1) ≡ 1 (mod n)`; declare composite on the first failure, else "probably prime." Fast — `O(k log n)` modular multiplications for `k` rounds. For *non-Carmichael* composite `n` the Fermat nonwitnesses are a proper subgroup, so at least half the bases are witnesses and `k` rounds give error `≤ 2^(-k)`. Gap: on **Carmichael numbers** the nonwitnesses are the whole unit group, so there is no error bound at all — the test silently passes infinitely many composites. This single failure mode is what a reliable test must remove.

**Euler / Solovay–Strassen test.** Strengthen Fermat using the **Euler criterion**: for prime `n` and `gcd(a,n)=1`, `a^((n-1)/2) ≡ (a/n) (mod n)`, where `(a/n)` is the Jacobi symbol. An `a` violating this (or with `gcd(a,n)>1`) is an **Euler witness**. The decisive improvement over Fermat: for *every* composite `n`, the Euler nonwitnesses form a *proper* subgroup of `(Z/nZ)*` — there is no Carmichael-style escape — so at least 50% of bases are witnesses and `k` rounds give error `≤ 2^(-k)` unconditionally. Gaps: it needs Jacobi-symbol machinery (more to implement and reason about), and its guarantee is only the 1/2 bound.

**Deterministic-under-a-hypothesis test (Miller 1976).** If one assumes the Generalized Riemann Hypothesis (GRH), a compositeness witness of the relevant kind is guaranteed to exist among the small bases `a ≤ c·(log n)^2` (Bach's explicit form: `a ≤ 2(log n)^2`). Checking all of them then gives a *deterministic* polynomial-time primality test. Gap: correctness rests on an unproven conjecture; without GRH there is no proof that the small-base search suffices.

## Evaluation settings

The natural yardstick is success on inputs where the previous tests are known to be weak, and cost measured in modular multiplications as a function of bit-length. Concrete probe inputs that exist before any new method: the Carmichael numbers `561, 1105, 1729, 2465, …` (where Fermat fails); small odd composites and primes for exhaustive correctness checks; and the explicit hard composites that determine deterministic base sets, e.g. `1373653` (smallest odd composite that the strengthened test fails to expose with bases `2, 3`) and `3215031751` (smallest failing bases `2, 3, 5, 7`). Cost is reported as the number of `O(log n)`-mult modular exponentiations per round times the number of rounds `k`, and as the resulting error probability as a function of `k`. The application setting is random-prime generation for cryptographic key sizes (hundreds to thousands of bits), and 64-bit integers as the regime where a *fixed* finite base set can make the test deterministic.

## Code framework

Pre-method primitives that already exist — fast modular arithmetic and a generic randomized "probably prime" harness with a single empty slot where the per-base test will go:

```python
import random

def modpow(base, exp, mod):
    # square-and-multiply: base^exp mod mod in O(log exp) modular mults
    result = 1
    base %= mod
    while exp > 0:
        if exp & 1:
            result = (result * base) % mod
        base = (base * base) % mod
        exp >>= 1
    return result

def small_factor_prefilter(n):
    # strip the obvious cases with a few small primes (cheap)
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n == p:
            return True            # prime
        if n % p == 0:
            return False           # composite
    return None                    # undecided -> need the real test

def single_base_test(n, a):
    # TODO: the per-base compositeness check we are about to design.
    # Returns False if a proves n composite, True if a fails to (n still suspect).
    pass

def is_probably_prime(n, k):
    pre = small_factor_prefilter(n)
    if pre is not None:
        return pre
    # TODO: decompose n-1 to expose the structure single_base_test will exploit
    for _ in range(k):
        a = random.randrange(2, n - 1)   # 1 and n-1 will be useless bases
        if not single_base_test(n, a):
            return False                  # composite, with certainty
    return True                           # probably prime; error controlled by k
```
