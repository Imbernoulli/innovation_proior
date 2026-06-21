# Context: reaching the machine-search frontier in the n = 29 maximal-determinant problem

## Research question

Among all `29 × 29` matrices with every entry `±1`, which one has the largest possible
absolute determinant? The single thing being designed is a **constructor**: a program that
emits one concrete `29 × 29` sign matrix `H`, scored by `|det(H)|` alone. The constructor's
output is a fixed integer matrix, its determinant is an exact integer, and that integer is the
whole result.

The order `29` is chosen deliberately. When `n ≡ 0 (mod 4)` a *Hadamard matrix* may exist — a
`±1` matrix with orthogonal rows hitting the ceiling `det = n^{n/2}` — and the answer is
essentially known. But `29 ≡ 1 (mod 4)`: orthogonal rows are impossible (off-diagonal inner
products of `±1` rows of odd length are odd, never zero), so the Hadamard bound is unreachable and
the true maximum is open. `n = 29` sits in the regime where the record is the product of decades of
number theory and computer search, not a closed form.

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
| Flip-annealing (single-entry SA, full `slogdet` scoring) | 149.87 | 0.4382 |
| Symmetric-design baseline (Jacobsthal `Q + I`) | 49 | 0.1433 |

## Prior art

- **Hadamard (1893) / Barba (1933) / Ehlich / Wojtas bounds.** Upper bounds on `|det|` keyed to
  `n mod 4`; for `n ≡ 1 (mod 4)` the Barba ceiling is `2^28 · 7^12 · 369.94…`, unreachable here.
- **Paley / Jacobsthal construction (1933).** The symmetric quadratic-residue design `Q + I`
  (multiplier `49`); rigid and parameter-free. The prime `29` admits a whole family of equally-
  valid relabelings: reindexing rows and columns by `i ↦ k·i (mod 29)` for any unit `k` gives a
  different, equally-structured starting matrix with the same baseline determinant — free, distinct
  structured restart points.
- **Simulated annealing (Kirkpatrick, Gelatt & Vecchi 1983).** Accept worsening moves with a
  cooling Metropolis probability to escape local optima; the engine that broke the symmetric basin.
  Each candidate flip is scored by a full `slogdet` factorization, `O(n³)` per candidate.
- **Matrix determinant lemma & Sherman–Morrison.** A single-entry sign flip at `(i,j)` is a
  *rank-one* perturbation `M ← M + Δ e_i e_jᵀ` with `Δ = −2 M_{ij}`. The matrix determinant lemma
  gives the new determinant in closed form: `det(M + Δ e_i e_jᵀ) = det(M)·(1 + Δ·(M⁻¹)_{ji})`.
  Sherman–Morrison describes how to update a carried matrix inverse after a rank-one perturbation
  in `O(n²)`.
- **Computer / program search for maximal determinants (Orrick, Solomon, Brent; ShinkaEvolve /
  ThetaEvolve).** Records for non-Hadamard orders come from large-scale search; the `n = 29` record
  `m = 320` is conjectured optimal but unproven, and no program-evolution system has matched it —
  the strongest reported machine result is `~0.576` (multiplier `~197`).

## The fixed substrate

The harness is a deterministic evaluator. It calls the constructor once, checks the output is a
`29 × 29` array with every entry in `{−1, +1}`, computes the determinant as an **exact integer**
via fraction-free Bareiss elimination, and returns `|det|`, the multiplier `m`, and the score.
Floating-point linear algebra is available to the constructor as a fast internal guide, but the
*reported* determinant is always the exact integer. The evaluator, normalization constants, and
order `n = 29` are frozen.

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
constructor is allowed; the harness reports the determinant of the *returned* matrix, with stated
seeds for reproducibility.

## Evaluation settings

A single deterministic instance: `n = 29`, scored by `score(H)`. The reference multipliers —
record `320`, Barba ceiling `369.94`, machine frontier `~197`, flip-annealing `149.87`, baseline
`49` — are the fixed yardsticks. There is no partial credit beyond the determinant itself.
