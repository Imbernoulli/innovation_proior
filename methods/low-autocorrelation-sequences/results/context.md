## Research question

Find length-`n` binary sequences `A = (a_0, …, a_{n-1})` with each `a_i ∈ {+1, -1}` whose
aperiodic autocorrelations are collectively as small as possible. The aperiodic
autocorrelation at shift `u` is

```
C_A(u) = sum_{i=0}^{n-1-u} a_i a_{i+u},   u = 0, 1, …, n-1,   C_A(0) = n.
```

The standard scalar measure of collective smallness is the merit factor

```
F(A) = n^2 / ( 2 * sum_{u=1}^{n-1} C_A(u)^2 ),
```

so that a *large* `F` means small off-peak correlation energy. Digital communications
(synchronization, pulse compression, spread-spectrum radar) want such sequences because a
large merit factor corresponds to a near-uniform amplitude spectrum. The central open
problem is asymptotic: determine `lim sup_{n→∞} F_n`, where `F_n = max_A F(A)` over all
`2^n` sequences of length `n`.

## Background

**The L4-norm equivalence.** Let `P_A(z) = sum_i a_i z^i` be the Littlewood polynomial
with `±1` coefficients. On the unit circle,

```
|| P_A ||_4^4 = ∫_0^1 |P_A(e^{2πiθ})|^4 dθ = n^2 + 2 sum_{u=1}^{n-1} C_A(u)^2 = n^2 (1 + 1/F(A)).
```

Minimizing off-peak correlation energy is therefore exactly minimizing the `L4` norm of a
`±1` polynomial — a question Littlewood posed in the 1960s. Since the mean of the squared
modulus over the circle is `n`, `1/F` measures how far `|P_A|^2` deviates from its mean:
flat spectrum ⇔ large merit factor.

**What a random sequence gives.** Over all `2^n` sequences, the mean value of `1/F` is
`(n-1)/n → 1`. So a typical sequence has `F ≈ 1`; anything with `F` bounded above `1`
asymptotically is already "structured."

**Periodic vs. aperiodic correlation.** Alongside the aperiodic `C_A(u)` sits the periodic
autocorrelation

```
R_A(u) = sum_{i=0}^{n-1} a_i a_{(i+u) mod n},   with   R_A(u) = C_A(u) + C_A(n-u),  0 < u < n.
```

The periodic version is algebraically tame: it is a group-ring / character-theoretic
object on the cyclic group `Z_n`, and a sequence with *constant* periodic autocorrelation
at all nonzero shifts is precisely a cyclic difference set. A `(v, k, λ)` cyclic
difference set is a `k`-subset `D ⊆ Z_v` in which every nonzero element of `Z_v` occurs
exactly `λ` times as a difference; setting `a_i = -1` iff `i ∈ D` gives constant
`R_A(u) = v - 4(k - λ)`. The quadratic residues modulo a prime `p ≡ 3 (mod 4)` form such a
set (the Paley / quadratic-residue difference set) with parameters
`(p, (p-1)/2, (p-3)/4)`, so `k - λ = (p+1)/4` and `R_A(u) = p - 4(k - λ) = -1` for all
`u ≠ 0`. The Legendre-sign convention reverses the nonzero signs but has the same constant
periodic autocorrelation when `p ≡ 3 (mod 4)`. There is no comparable group underlying the
*aperiodic* sums, because each `C_A(u)` is a *windowed* (interval-truncated) correlation.

**The character-sum / Gauss-sum toolbox.** For a prime `p`, the Legendre symbol
`(j | p)` is the quadratic multiplicative character, with Euler's criterion
`(j | p) ≡ j^{(p-1)/2} (mod p)` interpreted as `+1`, `-1`, or `0`.
The basic facts available: multiplicativity `(i | p)(j | p) = (ij | p)`; for `k ≠ 0`, the
quadratic Gauss sum `sum_{j} (j | p) ζ^{jk}` has modulus exactly `p^{1/2}` (with an
explicit phase `i^{(p-1)^2/4}`); and the Weil bound, which controls character sums of
polynomials —
`| sum_{x ∈ F_p} ( f(x) | p ) | ≤ (deg f - 1) p^{1/2}` whenever `f` is not a perfect square
in `F_p[x]`. These are the only handles that turn a sum of `±1`'s over an interval into
something estimable.

**Skew symmetry (a structural sieve).** A sequence of odd length `n = 2m+1` is
skew-symmetric if `a_{m+i} = (-1)^i a_{m-i}`. Such sequences have `C_A(u) = 0` for *all
odd* `u`, halving the number of nonzero correlations and roughly doubling the searchable
length for a fixed budget. It is a heuristic (and numerically well-supported) belief that
restricting to skew-symmetric sequences does not change the best asymptotic merit factor.

**The diagnostic numerical picture.** Exhaustive computation fixes `F_n` only for small
`n` (records around `n ≤ 60`); for larger `n` one has lower bounds from stochastic search.
The two largest known finite values, `F_13 ≈ 14.1` and `F_11 ≈ 12.1`, come from Barker
sequences (which exist only for `n ≤ 13`). Beyond that, no `F_n ≥ 10` is known. Decades
of simulated annealing / evolutionary search on this "low-autocorrelation binary sequence"
problem (studied in statistical physics as a spin-glass ground-state problem) find good
finite sequences but leave the asymptotic question open, as the landscape has an enormous
number of local optima.

## Baselines

**Barker sequences.** The ideal: `|C(u)| ≤ 1` for all `0 < u < n`. They give `F = n` (huge)
but provably exist only for `n ∈ {2,3,4,5,7,11,13}`; it is conjectured none exist beyond
`13`.

**Rudin–Shapiro (Golay–Shapiro) sequences.** Defined by the recursion on appended pairs
`X^{(m)} = X^{(m-1)}; Y^{(m-1)}`, `Y^{(m)} = X^{(m-1)}; -Y^{(m-1)}`, with `X^{(0)} = Y^{(0)} = [1]`.
Their aperiodic autocorrelations satisfy a clean recurrence, from which one computes
exactly `F = 3 / (1 - (-1/2)^m) → 3`. This is the earliest explicit infinite family with a
known nonzero asymptotic merit factor — it tops out at `3`, and generalizations of the
recursion (Høholdt–Jensen–Justesen 1985; Borwein–Mossinghoff 2000) provably never beat `3`.

**Maximal-length shift-register (m-) sequences.** Length `n = 2^m - 1`,
`x_i = (-1)^{Tr(β α^i)}` over `F_{2^m}`; equivalently a Singer difference set. The mean of
`1/F` over the `n` cyclic shifts is `(n-1)(n+4)/(3n^2) → 1/3`, suggesting some rotation
might beat `3` — but Jensen–Høholdt (1989) proved every rotation has asymptotic `F = 3`.

**Stochastic / evolutionary search.** Single- and double-flip local search, simulated
annealing, evolutionary algorithms, often restricted to skew-symmetric sequences to extend
reach. These find good *finite* sequences but the asymptotic question remains open.

## Evaluation settings

- **Length regimes.** Small `n` where `F_n` is known exactly (`n ≤ 60`); large `n` (hundreds
  to millions) where only constructed families can be evaluated.
- **Sequence ensembles to test.** Difference-set sequences (quadratic-residue / Paley,
  twin-prime, Singer) and products of such (Jacobi).
- **Metric.** The merit factor `F(A)` from `C_A(u)` above; secondarily the periodic
  autocorrelation `R_A(u)` (to certify difference-set structure) and `max_u |C_A(u)|`.
- **Yardstick families.** Rudin–Shapiro (`F → 3`) and m-sequences (`F → 3`) are the
  pre-existing asymptotic benchmarks; the random baseline `F ≈ 1`.
- **Protocol.** For a candidate family, compute `F` at a ladder of increasing lengths and
  read off the limit; for any family carrying a free parameter, sweep it and locate the
  optimum.

## Code framework

A minimal harness for candidate construction and measurement:

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
    """TODO: identify the arithmetic restrictions on admissible periods."""
    pass

def algebraic_sign(i, n):
    """TODO: choose the +-1 algebraic rule and zero-index convention."""
    pass

def build_sequence(n):
    """Build one period once valid_length() and algebraic_sign() are filled."""
    if not valid_length(n):
        raise ValueError("invalid candidate length")
    return np.array([algebraic_sign(i, n) for i in range(n)], dtype=np.int64)

def transform(A, *params):
    """TODO: derive a candidate transform of one period A (if any is needed)."""
    return A

def aperiodic_autocorr_sumsq(A):
    n = len(A)
    return sum(int(np.dot(A[:n-u], A[u:]))**2 for u in range(1, n))

def merit_factor(A):
    n = len(A)
    return n * n / (2.0 * aperiodic_autocorr_sumsq(A))

def periodic_autocorr(A, u):
    """Diagnostic: certifies difference-set structure when constant over u != 0."""
    return int(np.dot(A, np.roll(A, -u)))

def asymptotic_merit_factor(*params):
    """TODO: closed-form limiting F for the candidate family, if one can be derived."""
    pass
```
