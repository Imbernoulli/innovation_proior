## Research question

Given an odd integer `n` with hundreds or thousands of bits, decide whether `n` is prime, in time polynomial in `log n`. The application context is public-key cryptography, which requires a steady supply of large primes: a candidate is sampled and must be tested for primality before anything else can proceed. The naive test — trial division by all integers up to `sqrt(n)` — costs on the order of `sqrt(n)` operations, exponential in the bit-length `log n`. A usable test must run in time polynomial in `log n`, without requiring factoring `n`, and with a controllable probability of error.

## Background

The starting point is **Fermat's little theorem (FLT)**: if `n` is prime and `gcd(a, n) = 1`, then `a^(n-1) ≡ 1 (mod n)`. The left side is computable cheaply: **binary (square-and-multiply) modular exponentiation** computes `a^(n-1) mod n` in `O(log n)` modular multiplications by scanning the bits of the exponent, squaring the running base each step and multiplying in `a` when the bit is set. So FLT gives a fast *necessary* condition for primality, and its contrapositive gives a compositeness proof: if `a^(n-1) != 1 (mod n)` for some `a`, then `n` is composite. Such an `a` is called a **Fermat witness**; an `a` with `a^(n-1) ≡ 1` is a **Fermat nonwitness** (a "Fermat liar" when `n` is in fact composite).

The structural fact that governs how good this test is: for fixed `n`, the Fermat nonwitnesses `{a : a^(n-1) ≡ 1 (mod n)}` form a subgroup of the multiplicative group of units `(Z/nZ)*`. A subgroup is either the whole group or at most half of it. So if the nonwitness subgroup is *proper*, at least half of the coprime bases are witnesses, and the non-coprime bases also prove compositeness by sharing a factor with `n`.

A **Carmichael number** is a composite `n` for which `a^(n-1) ≡ 1 (mod n)` for *every* `a` coprime to `n`. The smallest is `561 = 3·11·17`; `1105` and `1729` follow. For a Carmichael number the Fermat nonwitnesses are the *entire* unit group. It is known that Carmichael numbers are squarefree and have at least three distinct prime factors.

One established fact about primes concerns the **structure of square roots of 1**. Modulo a prime `n`, the equation `x^2 ≡ 1` has only the solutions `x ≡ ±1`: from `x^2 - 1 = (x-1)(x+1) ≡ 0 (mod n)` and `n` prime, `n` must divide one of the factors. Modulo a composite `n` with at least two distinct prime factors, the Chinese remainder theorem (CRT) manufactures *extra* square roots of 1 — for instance modulo `15`, the values `1, 4, 11, 14` all square to `1`. A square root of 1 other than `±1` is therefore a *certificate of compositeness*. Also available, since `n` is odd, is the **decomposition `n - 1 = 2^s · d`** with `d` odd (and `s ≥ 1`).

The group-theoretic vocabulary available for reasoning about how many bases expose a composite: `(Z/nZ)*` has order `ϕ(n)`, and for composite `n` we have `ϕ(n) < n - 1` (some residues are not coprime to `n`). CRT, the structure of `(Z/p^α Z)*` (cyclic, of order `p^(α-1)(p-1)`), and the fact that a Carmichael number is squarefree with at least three distinct prime factors are all standard.

## Baselines

**Trial division.** Test divisibility of `n` by `2, 3, 5, …, ⌊sqrt(n)⌋`. Correct and deterministic, with cost `Θ(sqrt(n))` = exponential in `log n`. Useful as a cheap pre-filter (strip out small prime factors) before a real test.

**Fermat test.** Pick random bases `a`, check `a^(n-1) ≡ 1 (mod n)`; declare composite on the first failure, else "probably prime." Fast — `O(r log n)` modular multiplications for `r` rounds. For *non-Carmichael* composite `n` the Fermat nonwitnesses are a proper subgroup, so at least half the bases are witnesses and `r` rounds give error `≤ 2^(-r)`.

**Euler / Solovay–Strassen test.** Strengthen Fermat using the **Euler criterion**: for prime `n` and `gcd(a,n)=1`, `a^((n-1)/2) ≡ (a/n) (mod n)`, where `(a/n)` is the Jacobi symbol. An `a` violating this (or with `gcd(a,n)>1`) is an **Euler witness**. For *every* composite `n`, the Euler nonwitnesses form a *proper* subgroup of `(Z/nZ)*`, so at least 50% of bases are witnesses and `r` rounds give error `≤ 2^(-r)`.

**Deterministic-under-a-hypothesis test (Miller 1976).** Miller decomposes `n-1 = 2^s d` with `d` odd and checks a structured congruence at each base. Assuming the Generalized/extended Riemann Hypothesis (GRH/ERH), a base exposing any composite is guaranteed to exist among bases bounded by a constant times `(log n)^2`; checking all of them gives a *deterministic* polynomial-time primality test.

## Evaluation settings

The natural yardstick is success across all inputs and cost measured in modular multiplications as a function of bit-length. Concrete probe inputs: Carmichael numbers such as `561, 1105, 1729, 2465, …`; small odd composites and primes for exhaustive correctness checks; prime powers, where the square-root intuition alone is not enough; and products of several odd prime factors, where CRT creates many square roots of `1`. Cost is reported as the number of modular exponentiations and squarings per random trial, the number of trials, and the resulting one-sided error probability as a function of the trial count. The application setting is random-prime generation for large public-key parameters.

## Code framework

A generic randomized primality harness has fast modular arithmetic, a small-factor prefilter, one candidate-preparation slot, and one per-base compositeness slot.

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

def prepare_candidate(n):
    # TODO: compute any per-candidate data needed by the test.
    pass

def single_base_test(n, a, state):
    # TODO: the per-base compositeness check.
    # Returns False if a proves n composite, True if a fails to (n still suspect).
    pass

def is_probably_prime(n, trials=40):
    if n < 2:
        return False
    pre = small_factor_prefilter(n)
    if pre is not None:
        return pre
    state = prepare_candidate(n)
    for _ in range(trials):
        a = random.randrange(2, n - 1)   # skip the two trivial bases
        if not single_base_test(n, a, state):
            return False                  # composite, with certainty
    return True                           # probably prime; error controlled by trials
```
