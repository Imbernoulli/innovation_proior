# Low-Autocorrelation Binary Sequences and the Merit Factor

## Problem

For a binary sequence `A = (a_0, …, a_{n-1})`, `a_i ∈ {+1,-1}`, with aperiodic
autocorrelation `C_A(u) = sum_{i=0}^{n-1-u} a_i a_{i+u}`, maximize the merit factor

```
F(A) = n^2 / ( 2 * sum_{u=1}^{n-1} C_A(u)^2 ).
```

A random sequence gives `F ≈ 1`; the goal is an explicit infinite family with a *provable*
large asymptotic merit factor. Brute-force / stochastic search over the `2^n` cube
plateaus below `6` for large `n` and cannot resolve the asymptotic question, so the answer
is a *construction* plus a *character-sum analysis*, not a search record.

## Key idea

1. **Use periodic structure as the lever.** The periodic autocorrelation
   `R_A(u) = C_A(u) + C_A(n-u)` is a cyclic-group object, hence analyzable. A sequence with
   constant `R_A(u)` at all nonzero shifts is a cyclic difference set. The quadratic
   residues mod a prime `p ≡ 3 (mod 4)` (the Paley difference set) give the **Legendre
   sequence** `x_i = (i | p)` (Legendre symbol; `x_0 := +1`), whose nonzero periodic
   autocorrelations are constant: `R(u) = -1`.

2. **Rotation is the free parameter.** Flat periodic autocorrelation is rotation-invariant,
   but it only pins the *pair sums* `C(u) + C(n-u) = -1`; the bare sequence has `F → 3/2`.
   The individual aperiodic energies *do* change under rotation, so optimize over the
   rotation `r` (cyclic shift by `⌊rn⌋`).

3. **Character-sum analysis.** `C_{X_r}(u)` is a windowed sum of the quadratic character of
   a quadratic in `i`. In the Fourier domain, `X_p(ζ_k) - 1` is a quadratic Gauss sum of
   modulus exactly `p^{1/2}` for `k ≠ 0`; the extra `1` from `x_0 := +1` is carried in the
   error term. Packaging the merit factor through the fourth-moment quantity
   `L_A(a,b,c) = n^{-3} sum_k A(ζ_k)A(ζ_{k+a})\overline{A(ζ_{k+b})}\overline{A(ζ_{k+c})}`, multiplicativity
   collapses it to `L_{X_p}(a,b,c) = p^{-1} sum_{x∈F_p} ( x(x+a)(x+b)(x+c) | p ) + O(p^{-1/2})`.
   The **Weil bound** gives `≤ 3 p^{1/2}` for the inner sum unless the quartic is a perfect
   square — which happens exactly at the "ideal" pattern `I(a,b,c)` (one of `a,b,c` is `0`,
   the other two equal). Hence `max |L_{X_p} - I| ≤ 18 p^{-1/2} → 0`, yielding the limit.

## Result (the algorithm and its merit factor)

The rotated Legendre sequence `X_r` has asymptotic merit factor

```
1 / lim_{p→∞} F(X_r) = 1/6 + 8 (r - 1/4)^2     for 0 ≤ r ≤ 1/2,
                     = 1/6 + 8 (r - 3/4)^2     for 1/2 ≤ r ≤ 1,
```

a parabola minimized at the **quarter rotation** `r = 1/4` (and `3/4`), where

```
F(X_{1/4}) → 6.
```

Boundary checks: unrotated `r = 0` gives `1/F = 2/3`, `F → 3/2` (the bare sequence);
Rudin–Shapiro and m-sequences cap at `3` (the multiplicative/quadratic structure is what
reaches `6`). The same parabola holds for Jacobi and modified-Jacobi/twin-prime sequences.

**Beyond 6.** Periodically extending to a total length fraction `T` lowers the spread-out
off-peak energy when `T` is slightly above `1`, while the exact shift `u = n` aligns the
appended `(T - 1)n` terms with their originals and contributes about `(T - 1)^2 n^2`.
The appended Legendre/Jacobi construction and the related `(+,+,-,-)` and `(+,+,-,+)`
product constructions are governed by the same two-parameter limit:

```
1/g(R,T) = 1 - 4T/3 + 4 sum_{m∈N} max(0, 1 - m/T)^2 + sum_{m∈Z} max(0, 1 - |1 + (2R-m)/T|)^2
```

(which collapses to `1/6 + 8(R-1/4)^2` at `T = 1`) has global maximum

```
F_a = 6.342061…,  the largest root of  29 x^3 - 249 x^2 + 417 x - 27,
```

at `T = 1.057827…` (middle root of `4 x^3 - 30 x + 27`) and `R = 3/4 - T/2`, using the
representative `0 ≤ R < 1/2`. In the skew-symmetric Jacobi/product versions the same
optimum is written as `R = 1/4 - T/2`, equivalent by the half-period symmetry of `g`, and
gives the same limit. The additive (Galois/m-sequence) analogue gives
`F_b = 3.342065…`, the largest root of `7 x^3 - 33 x^2 + 33 x - 3`.

## Code

```python
import math
import numpy as np

def is_prime(n):
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    d = 3
    while d * d <= n:
        if n % d == 0:
            return False
        d += 2
    return True

def valid_length(n):
    return n % 4 == 3 and is_prime(n)

def algebraic_sign(i, n):
    j = i % n
    if j == 0:
        return 1
    return 1 if pow(j, (n - 1) // 2, n) == 1 else -1

def build_sequence(n):
    if not valid_length(n):
        raise ValueError("n must be a prime with n % 4 == 3")
    return np.array([algebraic_sign(i, n) for i in range(n)], dtype=np.int64)

def rotate(A, r):
    n = len(A)
    return np.roll(A, -(int(np.floor(r * n)) % n))

def extend_or_truncate(A, t=1.0):
    n = len(A)
    length = int(np.floor(t * n))
    if length <= 0:
        raise ValueError("target length must be positive")
    return A[np.arange(length) % n]

def aperiodic_autocorr_sumsq(A):
    n = len(A)
    return sum(int(np.dot(A[:n-u], A[u:]))**2 for u in range(1, n))

def merit_factor(A):
    n = len(A)
    return n * n / (2.0 * aperiodic_autocorr_sumsq(A))

def periodic_autocorr(A, u):
    return int(np.dot(A, np.roll(A, -u)))

def asymptotic_merit_factor(r, t=1.0):
    R = float(r)
    T = float(t)
    if T <= 0:
        raise ValueError("t must be positive")

    positive_m = sum(
        max(0.0, 1.0 - m / T) ** 2
        for m in range(1, int(math.floor(T)) + 1)
    )
    lo = math.floor(2.0 * R - 2.0 * T) - 2
    hi = math.ceil(2.0 * R + 2.0 * T) + 2
    integer_m = sum(
        max(0.0, 1.0 - abs(1.0 + (2.0 * R - m) / T)) ** 2
        for m in range(lo, hi + 1)
    )
    inverse_g = 1.0 - 4.0 * T / 3.0 + 4.0 * positive_m + integer_m
    return 1.0 / inverse_g

if __name__ == "__main__":
    p = 10007
    X = build_sequence(p)
    assert {periodic_autocorr(X, u) for u in range(1, 40)} == {-1}
    for r in [0.0, 0.25, 0.5]:
        A = extend_or_truncate(rotate(X, r), 1.0)
        print(r, round(merit_factor(A), 3), round(asymptotic_merit_factor(r), 3))
    T = 1.057827
    R = 0.75 - T / 2.0
    print(round(asymptotic_merit_factor(R, T), 6))
```

The code certifies the flat periodic autocorrelation `R(u) = -1`, compares finite rotated
merit factors with the limiting parabola, and evaluates the appended optimum from `g(R,T)`.
