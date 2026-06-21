# Context: the n = 29 maximal-determinant record

## Research question

Among all `29 × 29` matrices with every entry `±1`, which one has the largest possible absolute
determinant? The thing being designed is a **constructor**: a program that emits one concrete
`29 × 29` sign matrix `H`, scored by `|det(H)|` alone — an exact integer, the whole result.

The order `29` is chosen deliberately. When `n ≡ 0 (mod 4)` a *Hadamard matrix* may exist (a `±1`
matrix with orthogonal rows, hitting `det = n^{n/2}`) and the answer is essentially known. But
`29 ≡ 1 (mod 4)`: orthogonal rows are impossible (off-diagonal inner products of `±1` rows of odd
length are odd, never zero), so the Hadamard bound is unreachable and the true maximum is open.
`n = 29` is the regime where the record is the product of decades of number theory and computer
search, not a closed form.

## How the score is defined

Every `29 × 29` integer `±1` determinant is a multiple of `2^{28}`, and for the best constructions
an additional `7^{12}` factors out. The record (Bruce Solomon 2002, Will Orrick's maximal-
determinant database) is `|det| = 2^28 · 7^12 · 320`. Results are quoted by their **multiplier**
`m := |det(H)| / (2^28 · 7^12)`, and the score normalizes by a fixed denominator:

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
| Rank-one flip annealing (this scaffold's prior rung) | 184.60 | 0.5398 |
| Symmetric-design baseline (Jacobsthal `Q + I`) | 49 | 0.1433 |

## Searching the space of ±1 matrices

The earlier rungs of this ladder search the space of `±1` matrices directly: simulated annealing on
single-entry sign flips, made cheap by a rank-one (matrix-determinant-lemma / Sherman–Morrison)
update so it can run millions of flips from several structured seeds. From the best structured seeds
this search settles near multiplier `184.6`, in the band of the best reported program-evolution
results. `|det|` is a function of `841` coupled signs, and each single-entry flip changes one sign at
a time.

## Gram-matrix formulation

The maximal-determinant problem at `n ≡ 1 (mod 4)` has a standard reformulation through the *Gram
matrix* `G = R Rᵀ` of a candidate `±1` matrix `R`. `G` is a symmetric integer design with `29` on the
diagonal and off-diagonal inner products restricted to the residues the Barba analysis permits, and
`det(R)² = det(G)`. The dedicated maximal-determinant searches work in this space: they enumerate
admissible Gram matrices `G` and decompose an optimal `G` into a `±1` factor `R`.

## Prior art

- **Hadamard (1893) / Barba (1933) / Ehlich / Wojtas bounds.** Upper bounds keyed to `n mod 4`; for
  `n ≡ 1 (mod 4)` the Barba ceiling is `2^28 · 7^12 · 369.94…`, unmet at `n = 29`.
- **Paley / Jacobsthal construction (1933).** The symmetric quadratic-residue design `Q + I`
  (multiplier `49`); the rigid parameter-free baseline.
- **Gram-matrix maximal-determinant search (Orrick, Solomon, Brent; 2000s).** Records for
  non-Hadamard orders come from searching admissible Gram matrices and decomposing them into `±1`
  factors. At `n = 29`, Bruce Solomon found a Gram matrix `G` on 6 July 2002, with
  `det(G) = (2^28 · 7^12 · 320)²`; Will Orrick tabulated it, and R. P. Brent's order-29 page
  published the compressed `G` and, by randomised decomposition, `4918` Hadamard-equivalence classes
  of explicit `±1` solutions `R` (`s29allsofar.txt`). The record `m = 320` is *conjectured* optimal,
  not proven.

## The fixed substrate

The harness is a deterministic evaluator: it calls the constructor once, checks the output is a
`29 × 29` array with entries in `{−1, +1}`, computes the determinant as an **exact integer** via
fraction-free Bareiss elimination, and returns `|det|`, the multiplier `m`, and the score.
Floating-point linear algebra is available as a fast internal guide, but the *reported* determinant
is always the exact integer. The evaluator, normalization constants, and order `n = 29` are frozen.

## The editable interface

Exactly one function is editable: `construct()`, returning the `29 × 29` `±1` matrix. The
exact-determinant helper and scorer are fixed.

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

Every valid output must satisfy shape `(29, 29)` and entries exactly `−1` or `+1`.

## Evaluation settings

A single deterministic instance: `n = 29`, scored by `score(H)`. The reference multipliers —
record `320`, Barba ceiling `369.94`, machine frontier `~197`, rank-one annealing `184.60`, baseline
`49` — are the fixed yardsticks. No partial credit beyond the determinant itself.
</content>
</invoke>
