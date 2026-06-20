# Context: escaping the symmetric basin in the n = 29 maximal-determinant problem

## Research question

Among all `29 × 29` matrices with every entry `±1`, which one has the largest possible
absolute determinant? The single thing being designed is a **constructor**: a program that
emits one concrete `29 × 29` sign matrix `H`, scored by `|det(H)|` alone. The constructor's
output is a fixed integer matrix, its determinant is an exact integer, and that integer is the
whole result.

The order `29` is chosen deliberately. When `n ≡ 0 (mod 4)` a *Hadamard matrix* may exist — a
`±1` matrix with orthogonal rows hitting the ceiling `det = n^{n/2}` — and the answer is
essentially known. But `29 ≡ 1 (mod 4)`: orthogonal rows are impossible (an odd-order `±1` matrix
cannot have `HHᵀ = nI`, since off-diagonal inner products of `±1` rows of odd length are odd,
never zero), so the Hadamard bound is unreachable and the true maximum is open. `n = 29` sits in
the regime where the record is the product of decades of number theory and computer search, not a
closed form.

## How the score is defined

Every `29 × 29` integer `±1` determinant is a multiple of `2^{28}`, and for the best constructions
an additional `7^{12}` factors out. The record at this order (Bruce Solomon 2002, in Will Orrick's
maximal-determinant database) is `|det| = 2^28 · 7^12 · 320`. Results are quoted by their
**multiplier** `m := |det(H)| / (2^28 · 7^12)`, and the score normalizes by a fixed denominator:

```
score(H) = |det(H)| / (2^28 · 7^12 · 342).
```

The `342` sits above the record multiplier `320` and below the Barba ceiling
(`49·√57 ≈ 369.94`), so a score of `1.0` has never been reached. Yardsticks:

| Reference point | multiplier `m` | score |
|---|---|---|
| Barba bound (provable ceiling, `n ≡ 1 mod 4`) | 369.94 | 1.0816 |
| **Classical record** (Orrick/Solomon 2002, conjectured optimal) | **320** | **0.9357** |
| Best reported LLM-evolution result (ThetaEvolve) | ~197 | ~0.576 |
| Symmetric-design baseline (Jacobsthal `Q + I`) | 49 | 0.1433 |

## The starting point and its wall

The principled place to start is the symmetric *Jacobsthal* design. For the prime `q = 29`, the
quadratic-residue matrix `Q_{ij} = χ(i − j)` (with `χ` the Legendre symbol) is symmetric and
satisfies `QQᵀ = qI − J`; filling its zero diagonal with `+1` gives a legal `±1` matrix
`R = Q + I`. This realizes the "almost orthogonal" Gram structure parity permits and yields
multiplier exactly `49` (score `0.1433`) — but it is rigid. Direct measurement shows that **not
one of the `841` single-entry sign flips raises `|det|`**: `Q + I` is a *strict local maximum*
under the basic flip move. Any greedy hill-climb from this seed terminates immediately at `49`.
The symmetry that makes the design elegant has parked the search at the bottom of a basin that
looks like a peak to a greedy eye, while the record at `320` lives elsewhere on the landscape.
This is the wall the present method must break: to get anywhere, the search must be willing to
accept moves that make the determinant temporarily *worse*, in order to cross the ridge out of the
symmetric basin.

## Prior art

- **Hadamard's bound (1893).** `|det| ≤ n^{n/2}`, attained only at orthogonal rows; unreachable
  and loose for `n = 29`.
- **Barba (1933) / Ehlich / Wojtas residue-class bounds.** For `n ≡ 1 (mod 4)`,
  `|det| ≤ √(2n−1)·(n−1)^{(n−1)/2} = 2^28 · 7^12 · 369.94…` at `n = 29`; met with equality only by
  a Gram structure that does not exist here.
- **Paley / Jacobsthal construction (1933).** The symmetric quadratic-residue design above; rigid,
  parameter-free, multiplier `49`.
- **Simulated annealing (Kirkpatrick, Gelatt & Vecchi 1983).** The general tool for escaping
  local optima of a discrete objective: propose a local move, accept improving moves always and
  worsening moves with a temperature-dependent Metropolis probability, and cool the temperature
  over time so the search wanders early and settles late. The natural engine for leaving a strict
  local maximum.
- **Computer search for maximal determinants (Orrick, Solomon, Brent, et al., 2000s).** The
  records for non-Hadamard orders come from large-scale search over Gram matrices and `±1`
  configurations seeded by number-theoretic constructions; the `n = 29` record `m = 320` is
  conjectured optimal but unproven.

## The fixed substrate

The harness is a deterministic evaluator. It calls the constructor once, checks the output is a
`29 × 29` array with every entry in `{−1, +1}`, computes the determinant as an **exact integer**
via fraction-free Bareiss elimination, and returns `|det|`, the multiplier `m`, and the score.
Floating-point linear algebra (`numpy.linalg.slogdet`) is available to the constructor as a fast
internal guide, but the *reported* determinant is always the exact integer. The evaluator,
normalization constants, and order `n = 29` are frozen.

## The editable interface

Exactly one function is editable: `construct()`, returning the `29 × 29` `±1` matrix. The
exact-determinant helper and scorer are fixed and shown for reference.

```python
import numpy as np

N = 29
BASE = (2**28) * (7**12)
NORM = BASE * 342

def bareiss_det(M):
    """Exact integer determinant via fraction-free Gaussian elimination."""
    A = [[int(x) for x in row] for row in M]
    n = len(A); sign = 1; prev = 1
    for k in range(n - 1):
        if A[k][k] == 0:
            sw = next((i for i in range(k + 1, n) if A[i][k] != 0), None)
            if sw is None:
                return 0
            A[k], A[sw] = A[sw], A[k]; sign = -sign
        for i in range(k + 1, n):
            for j in range(k + 1, n):
                A[i][j] = (A[i][j] * A[k][k] - A[i][k] * A[k][j]) // prev
        prev = A[k][k]
    return sign * A[n - 1][n - 1]

def score(M):
    d = abs(bareiss_det(M))
    return d, d / BASE, d / NORM     # |det|, multiplier m, normalized score

# ---- EDITABLE: the constructor. ----
def construct():
    ...
```

Every valid output must satisfy shape `(29, 29)` and entries exactly `−1` or `+1`. A randomized
constructor is allowed; the harness reports the determinant of the *returned* matrix, and a
stated seed fixes reproducibility.

## Evaluation settings

A single deterministic instance: `n = 29`, scored by `score(H)`. The reference multipliers —
record `320`, Barba ceiling `369.94`, baseline `49` — are the fixed yardsticks. There is no
partial credit beyond the determinant itself and no way to game the metric except by producing a
sign matrix with a larger determinant.
