# The Miller–Rabin Primality Test

## Problem

Decide whether a large odd integer `n` is prime, in time polynomial in `log n`, with a controllable and vanishingly small probability of error — and crucially with a guarantee that rests on proven mathematics. A fast deterministic test (Miller's) already exists, but its correctness is conditional on the unproven extended Riemann hypothesis: a fixed short list of small bases is only guaranteed to expose every composite if that conjectural input is available. The aim is to keep the speed yet drop that dependence — and along the way also kill the failure mode that lets a structured family of composites (the Carmichael numbers) pass undetected.

## Key idea

Randomize the choice of base. Miller's deterministic test needs the Riemann hypothesis only to guarantee that a *fixed short list* of bases contains an exposer of every composite; if instead one proves that a *large fraction* of all bases expose any composite, random sampling drives the error down geometrically with no hypothesis at all. So the task reduces to: find a compositeness condition on a base `a`, and prove its exposers are dense for every composite `n`. Fermat's little theorem (`a^(n-1) ≡ 1 (mod n)` for prime `n`, `gcd(a,n)=1`) gives a cheap necessary condition, but it inspects only the top value `a^(n-1)` and is fooled by Carmichael numbers (composite `n` with `a^(n-1) ≡ 1` for all coprime `a`). Miller–Rabin keeps the *path* to `a^(n-1)`. Write `n - 1 = 2^s d` with `d` odd, and look at the chain

`a^d, a^(2d), a^(4d), …, a^(2^(s-1)d), a^(2^s d) = a^(n-1)`,

each term the square of the previous. Modulo a prime, the only square roots of `1` are `±1`. So for prime `n` the chain must either start at `1` (`a^d ≡ 1`) or pass through `-1` before reaching `1` (`a^(2^i d) ≡ -1` for some `i ∈ {0,…,s-1}`). A base `a` for which neither holds proves `n` composite: either the top Fermat value fails to be `1`, or the path reaches `1` from a value that is neither `1` nor `-1`, exposing a nontrivial square root of `1`.

## Witness, and the algorithm

For odd `n` with `n - 1 = 2^s d` (`d` odd) and base `a`:

- `a` is a **nonwitness** if `a^d ≡ 1 (mod n)`, or `a^(2^i d) ≡ -1 (mod n)` for some `i ∈ {0,…,s-1}`.
- otherwise `a` is a **witness**, and `n` is composite.

The test:

1. Strip small prime factors as a pre-filter.
2. Decompose `n - 1 = 2^s d`.
3. For each of `t` rounds: pick `a ∈ {2,…,n-2}`, compute `x = a^d mod n`; if `x ∈ {1, n-1}` the round passes; else square up to `s-1` times — if any square equals `n-1` the round passes, if a square equals `1` (from `x ≠ ±1`) `a` is a witness and `n` is composite; if the loop ends without `n-1`, `a` is a witness.
4. If all rounds pass, report "probably prime."

## Reliability

For every odd composite `n`, more than `3/4` of bases in `{2,…,n-2}` are witnesses. The proof first bounds nonwitnesses in `{1,…,n-1}` by one quarter, then uses the fact that `1` and `n-1` are always nonwitnesses, so removing them makes the sampled range strictly better. For non-prime-powers, let `i0` be the largest index with some unit `a0` satisfying `a0^(2^i0 d) ≡ -1 (mod n)`, and set `G_n = {a : a^(2^i0 d) ≡ ±1 (mod n)}`. This subgroup contains all nonwitnesses. Off Carmichael numbers, the Fermat group `F_n = {a : a^(n-1) ≡ 1}` is a proper subgroup of the unit group; the CRT element used to put a unit outside `G_n` also lies in `F_n`, so `G_n` is strictly inside `F_n`, giving two strict subgroup steps and index at least `4`. On Carmichael numbers, `n` is squarefree with at least three prime factors; allowing each prime factor its own sign gives a group `H_n`, and the sign map `H_n → ∏_l {±1 mod p_l}` is onto while `G_n` maps only to the two diagonal sign patterns, so `|H_n|/|G_n| = 2^{r-1} ≥ 4`. Prime powers are counted directly: nonwitnesses are exactly the `p-1` solutions of `a^(p-1) ≡ 1 (mod p^α)`, so their proportion in `{1,…,p^α-1}` is `(p-1)/(p^α-1) ≤ 1/4`, again strictly better after removing `1` and `n-1`.

Consequences: the test never rejects a prime (one-sided error). For composite `n`, `t` independent random rounds give false-"probably prime" probability `< 4^{-t}` — an unconditional bound, proven with no appeal to the Riemann hypothesis. Cost is `O(t · log n)` modular multiplications at the exponentiation level.

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

def small_factor_prefilter(n):
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n == p:
            return True
        if n % p == 0:
            return False
    return None

def prepare_candidate(n):
    # n - 1 = 2^s * d, d odd
    d = n - 1
    s = 0
    while d % 2 == 0:
        d //= 2
        s += 1
    return s, d

def single_base_test(n, a, state):
    # False means a proves n composite; True means this base passes.
    s, d = state
    x = modpow(a, d, n)
    if x == 1 or x == n - 1:
        return True
    for _ in range(s - 1):
        x = (x * x) % n
        if x == n - 1:
            return True
        if x == 1:               # nontrivial square root of 1 -> composite
            return False
    return False

def is_probably_prime(n, trials=40):
    # randomized over 2..n-2: for trials > 0, composite error < 4^(-trials)
    if n < 2:
        return False
    pre = small_factor_prefilter(n)
    if pre is not None:
        return pre
    state = prepare_candidate(n)
    for _ in range(trials):
        a = random.randrange(2, n - 1)
        if not single_base_test(n, a, state):
            return False
    return True
```
