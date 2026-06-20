# Context: the maximal-determinant ±1 matrix at n = 29

## Research question

Among all `29 × 29` matrices with every entry `±1`, which one has the largest possible
absolute determinant? The single thing being designed is a **constructor**: a program that
emits one concrete `29 × 29` sign matrix `H`, scored by `|det(H)|` alone. Nothing about the
harness is learned or stochastic at evaluation time — the constructor's output is a fixed
integer matrix, its determinant is an exact integer, and that integer is the whole result.

The order `29` is chosen deliberately, and it is what makes the problem hard. When `n ≡ 0
(mod 4)` a *Hadamard matrix* may exist — a `±1` matrix with mutually orthogonal rows, hitting
the absolute ceiling `det = n^{n/2}` — and for those orders the answer is essentially known.
But `29 ≡ 1 (mod 4)`: orthogonal rows are impossible (an odd-order `±1` matrix cannot have
`HHᵀ = nI`, since off-diagonal inner products of `±1` rows of odd length are odd, never zero),
so the Hadamard bound is unreachable and the true maximum is a genuinely open question. `n = 29`
sits in the regime where the record is the product of decades of number theory and computer
search, not a closed form — which is exactly why it is used as a discovery target.

## How the score is defined

The determinant of a `29 × 29` integer `±1` matrix is always divisible by a large fixed power:
every such determinant is a multiple of `2^{28}`, and for the best constructions an additional
`7^{12}` factors out. The known record at this order, found by Bruce Solomon in 2002 and
tabulated in Will Orrick's maximal-determinant database, is

```
|det| = 2^28 · 7^12 · 320.
```

so it is the convention this task adopts to quote a result by its **multiplier**
`m := |det(H)| / (2^28 · 7^12)` and to normalize the score by a fixed denominator:

```
score(H) = |det(H)| / (2^28 · 7^12 · 342).
```

The denominator's `342` is not itself an achieved value: it sits *above* the record multiplier
`320` and *below* the Barba upper bound for this order (`49·√57 ≈ 369.94`, the provable ceiling
for `n ≡ 1 (mod 4)`). So a score of `1.0` is a target that has never been reached and may be
unreachable; the record construction scores `320/342 ≈ 0.9357`. The headline yardsticks:

| Reference point | multiplier `m` | score |
|---|---|---|
| Barba bound (provable ceiling, `n ≡ 1 mod 4`) | 369.94 | 1.0816 |
| **Classical record** (Orrick/Solomon 2002, conjectured optimal) | **320** | **0.9357** |
| Best reported LLM-evolution result (ThetaEvolve) | ~197 | ~0.576 |
| Symmetric-design baseline (this method) | 49 | 0.1433 |

This method designs the *starting* rung: a principled, parameter-free, number-theoretic sign
matrix that is guaranteed legal and gives a concrete baseline determinant. It is not expected to
approach the record — its purpose is to establish the floor from which any searched construction
must climb.

## Prior art

- **Hadamard's bound (1893).** For any real matrix with entries in `[−1,1]`, `|det| ≤ n^{n/2}`,
  attained iff the rows are orthogonal. *Gap here:* attainable only for `n = 1, 2` and `n ≡ 0
  (mod 4)`; for `n = 29` it is loose and unreachable, giving only an upper bound `29^{14.5}` far
  above what any `±1` matrix of this order can reach.
- **Barba (1933) / Ehlich / Wojtas residue-class bounds.** Sharper ceilings keyed to `n mod 4`.
  For `n ≡ 1 (mod 4)`, Barba's bound is `|det| ≤ √(2n−1)·(n−1)^{(n−1)/2}`, which for `n = 29`
  equals `√57 · 28^{14} = 2^28 · 7^12 · (49√57) = 2^28 · 7^12 · 369.94…`. *Gap:* the bound is met
  with equality only when a `±1` matrix `R` exists with `RRᵀ = (n−1)I + J` (a specific Gram
  structure), which for `n = 29` it does not — the record `320` falls short of `369.94`.
- **Paley / Jacobsthal construction (1933).** For a prime `q ≡ 1 (mod 4)`, the quadratic-residue
  (Legendre-symbol) matrix `Q`, `Q_{ij} = χ(i−j)`, is symmetric and satisfies `QQᵀ = qI − J`.
  Adding the identity gives a `±1` matrix `R = Q + I` with a clean, computable determinant. This
  is the object the present method builds. *Gap:* it is a fixed symmetric design; its determinant
  is far below the record, because the structure that makes it elegant pins it to one local
  configuration.
- **Conference matrices.** A matrix `C` with zero diagonal and `±1` off-diagonal satisfying
  `CCᵀ = (n−1)I` is a *conference matrix*; symmetric conference matrices of order `n+1` exist
  exactly when `n+1 ≡ 2 (mod 4)` and the number theory cooperates. For `n = 29`, `n + 1 = 30 ≡ 2
  (mod 4)` and `29` is a prime `≡ 1 (mod 4)`, so Paley's construction supplies one for free — the
  almost-orthogonal structure this method needs is realizable.

## The fixed substrate

The harness is a thin, deterministic evaluator. It calls the constructor once, receives a
`29 × 29` array, checks that every entry is in `{−1, +1}` and the shape is exact, computes the
determinant as an **exact integer** (a fraction-free Bareiss elimination, so there is no
floating-point rounding in the reported value), and returns `|det|`, the multiplier `m`, and the
normalized `score`. The evaluator, the normalization constants, and the order `n = 29` are frozen.

## The editable interface

Exactly one function is editable: `construct()`, returning the `29 × 29` `±1` matrix. The
exact-determinant helper and the scorer are fixed and shown for reference.

```python
import numpy as np

N = 29
BASE = (2**28) * (7**12)          # structural factor of the n=29 record
NORM = BASE * 342                 # score == 1.0 target (above the record, below Barba)

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

Every valid output must satisfy: shape `(29, 29)`; every entry exactly `−1` or `+1`. There are no
other constraints.

## Evaluation settings

A single deterministic instance: `n = 29`, scored by `score(H)` above. The three reference
multipliers — record `320`, Barba ceiling `369.94`, baseline `49` — are the fixed yardsticks. The
score is the geometric heart of the task: there is no partial credit beyond the determinant
itself, and no way to game the metric except by producing a sign matrix with a larger determinant.
