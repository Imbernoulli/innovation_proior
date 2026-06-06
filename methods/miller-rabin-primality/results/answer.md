# The Miller–Rabin Primality Test

## Problem

Decide whether a large odd integer `n` is prime, in time polynomial in `log n`, with a controllable and vanishingly small probability of error — and without the failure mode that lets a structured family of composites (the Carmichael numbers) pass undetected.

## Key idea

Fermat's little theorem (`a^(n-1) ≡ 1 (mod n)` for prime `n`, `gcd(a,n)=1`) gives a cheap necessary condition, but it inspects only the top value `a^(n-1)` and is fooled by Carmichael numbers (composite `n` with `a^(n-1) ≡ 1` for all coprime `a`). Miller–Rabin keeps the *path* to `a^(n-1)`. Write `n - 1 = 2^e k` with `k` odd, and look at the chain

`a^k, a^(2k), a^(4k), …, a^(2^(e-1)k), a^(2^e k) = a^(n-1)`,

each term the square of the previous. Modulo a prime, the only square roots of `1` are `±1`. So for prime `n` the chain must either start at `1` (`a^k ≡ 1`) or pass through `-1` before reaching `1` (`a^(2^i k) ≡ -1` for some `i ∈ {0,…,e-1}`). A base `a` for which **neither** holds has exposed a *nontrivial square root of 1* and thereby proves `n` composite — and this works on Carmichael numbers too, since their multiple prime factors guarantee such square roots exist.

## Witness, and the algorithm

For odd `n` with `n - 1 = 2^e k` (`k` odd) and base `a`:

- `a` is a **nonwitness** if `a^k ≡ 1 (mod n)`, or `a^(2^i k) ≡ -1 (mod n)` for some `i ∈ {0,…,e-1}`.
- otherwise `a` is a **witness**, and `n` is composite.

The test:

1. Strip small prime factors as a pre-filter.
2. Decompose `n - 1 = 2^e k`.
3. For each of `t` rounds: pick `a ∈ {2,…,n-2}`, compute `x = a^k mod n`; if `x ∈ {1, n-1}` the round passes; else square up to `e-1` times — if any square equals `n-1` the round passes, if a square equals `1` (from `x ≠ ±1`) `a` is a witness and `n` is composite; if the loop ends without `n-1`, `a` is a witness.
4. If all rounds pass, report "probably prime."

## Reliability

For every odd composite `n`, the nonwitnesses lie in a proper subgroup of `(Z/nZ)*` of index at least 4 — so **more than 3/4** of bases in `{2,…,n-2}` are witnesses. The argument: let `i0` be the largest index with some `a0^(2^i0 k) ≡ -1 (mod n)`, and `G_n = {a : a^(2^i0 k) ≡ ±1 (mod n)}`, a subgroup containing all nonwitnesses. `G_n` is proper (CRT-built unit lies outside it), giving the 1/2 bound; index ≥ 4 follows by inserting the Fermat group `F_n = {a : a^(n-1) ≡ 1}` strictly between (non-Carmichael `n`), or, for Carmichael `n` (which have ≥ 3 distinct prime factors), via the surjective sign homomorphism `H_n → ∏_l {±1 mod p_l^{α_l}}` whose kernel makes `|H_n|/|G_n| = 2^{r-1} ≥ 4`. Prime powers `p^α` give nonwitness proportion `(p-1)/(p^α-1) ≤ 1/4`.

Consequences: the test never rejects a prime (one-sided error). For composite `n`, `t` independent rounds give false-"probably prime" probability `< 4^{-t}`. Cost is `O(t · log n)` modular multiplications. With a fixed base set the test is deterministic on bounded ranges: `{2,3}` decides all `n < 1{,}373{,}653`; `{2,3,5,7}` all `n < 3{,}215{,}031{,}751`; and `{2,3,5,7,11,13,17,19,23,29,31,37}` decides **all `n < 2^64`** exactly.

## Code

```python
import random

def modpow(base, exp, mod):
    result = 1
    base %= mod
    while exp > 0:
        if exp & 1:
            result = (result * base) % mod
        base = (base * base) % mod
        exp >>= 1
    return result

def _decompose(n):
    # n - 1 = 2^e * k, k odd
    k, e = n - 1, 0
    while k % 2 == 0:
        k //= 2
        e += 1
    return e, k

def is_composite_witness(n, a, e, k):
    # True iff a is a Miller-Rabin witness (proves n composite)
    x = modpow(a, k, n)
    if x == 1 or x == n - 1:
        return False
    for _ in range(e - 1):
        x = (x * x) % n
        if x == n - 1:
            return False
        if x == 1:               # nontrivial square root of 1 -> composite
            return True
    return True

def is_probably_prime(n, t=40):
    # randomized: never rejects a prime; error < 4^(-t) on composites
    if n < 2:
        return False
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n == p:
            return True
        if n % p == 0:
            return n == p
    e, k = _decompose(n)
    for _ in range(t):
        a = random.randrange(2, n - 1)
        if is_composite_witness(n, a, e, k):
            return False
    return True

def is_prime_deterministic_64(n):
    # exact for all n < 2^64
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    e, k = _decompose(n)
    for a in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n == a:
            return True
        if n % a == 0:
            return False
        if is_composite_witness(n, a, e, k):
            return False
    return True
```
