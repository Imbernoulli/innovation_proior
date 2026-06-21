## Research question

Among all `29 × 29` matrices with every entry `±1`, which one has the largest possible
absolute determinant? The single thing being designed is a **constructor**: a program that
emits one concrete `29 × 29` sign matrix `H`, scored by `|det(H)|` alone. The determinant is an
exact integer, and that integer is the whole result.

The order `29` is hard because `29 ≡ 1 (mod 4)`. A Hadamard matrix — a `±1` matrix with
mutually orthogonal rows and determinant `n^{n/2}` — can exist only for `n = 1, 2` and
`n ≡ 0 (mod 4)`. For odd order, orthogonal rows are impossible: the off-diagonal inner product
of two `±1` rows of odd length is odd, never zero. So the Hadamard bound is unreachable and the
true maximum remains open.

## Prior art / Background / Baselines

- **Hadamard's bound.** For any real matrix with entries in `[−1,1]`, `|det| ≤ n^{n/2}`,
  attained exactly when rows are mutually orthogonal. *Gap:* attainable only for `n = 1, 2` and
  `n ≡ 0 (mod 4)`; for `n = 29` it is unreachable, so it gives only a loose ceiling far above
  any achievable `±1` determinant.
- **Barba / Ehlich / Wojtas residue-class bounds.** Sharper upper bounds keyed to `n mod 4`. For
  `n ≡ 1 (mod 4)`, Barba's bound is `|det| ≤ √(2n−1)·(n−1)^{(n−1)/2}`, which at `n = 29` equals
  `2^28 · 7^12 · 369.94…`. *Gap:* equality would require a `±1` matrix `R` with
  `RRᵀ = (n−1)I + J`, a Gram structure that does not exist at `n = 29`; the record falls short,
  so the bound is a ceiling, not a construction.
- **Paley / Jacobsthal quadratic-residue construction.** For a prime `q ≡ 1 (mod 4)`, the
  Legendre-symbol matrix `Q_{ij} = χ(i−j)` is symmetric with `QQᵀ = qI − J`, and `R = Q + I` is a
  computable `±1` matrix. *Gap:* it is a fixed symmetric design; at `q = 29` its determinant
  (`m = 49`) is far below the record.
- **Computer search for maximal determinants.** The best known values for non-Hadamard orders
  come from large-scale search over Gram matrices and `±1` configurations. *Gap:* the record at
  `n = 29` (`m = 320`) is the output of dedicated search infrastructure and is only conjectured
  optimal; reproducing it from scratch in a single constructor is not expected.

## Fixed substrate / Code framework

The harness is a thin, deterministic evaluator. It calls `construct()` once, receives a
`29 × 29` array, checks that every entry is in `{−1, +1}` and the shape is exact, and computes
the determinant as an **exact integer** via fraction-free Bareiss elimination. The reported
determinant is never a floating-point approximation. Floating-point linear algebra
(`numpy.linalg.slogdet`) is available to the constructor as a fast internal guide, but the
returned score is always exact. The evaluator, the normalization constants, and `n = 29` are
frozen.

The determinant of any `29 × 29` `±1` matrix is always divisible by `2^28`, and the best known
constructions carry an additional `7^12` factor. The record is therefore quoted by its
**multiplier** `m := |det(H)| / (2^28 · 7^12)`, with normalized score

```
score(H) = |det(H)| / (2^28 · 7^12 · 342).
```

The normalization `342` sits above the current record multiplier `320` and below the Barba
ceiling `49·√57 ≈ 369.94` for `n ≡ 1 (mod 4)`. A score of `1.0` has never been reached and may
be unreachable; the record scores `320/342 ≈ 0.9357`; the default scaffold scores
`49/342 ≈ 0.1433`.

| Reference point | multiplier `m` | score |
|---|---|---|
| Barba bound (provable ceiling, `n ≡ 1 mod 4`) | 369.94 | 1.0816 |
| Current record (Orrick / Solomon) | 320 | 0.9357 |
| Default scaffold (symmetric design) | 49 | 0.1433 |

## Editable interface

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

# ---- EDITABLE: the constructor. Default fill = symmetric Jacobsthal design. ----
def construct():
    q = 29
    qr = set((i * i) % q for i in range(1, q))
    chi = lambda a: 0 if a % q == 0 else (1 if a % q in qr else -1)
    Q = np.array([[chi(i - j) for j in range(q)] for i in range(q)], dtype=int)
    return Q + np.eye(q, dtype=int)   # R = Q + I, a ±1 matrix
```

Every valid output must satisfy: shape `(29, 29)`; every entry exactly `−1` or `+1`. There are
no other constraints.

## Evaluation settings

A single deterministic instance: `n = 29`, scored by `score(H)` above. If the constructor is
randomized internally, the run is fixed to a stated seed so the determinant is reproducible. The
reference multipliers — record `320`, Barba ceiling `369.94`, scaffold baseline `49` — are the
fixed yardsticks. The score is the only metric: there is no partial credit beyond the
determinant, no held-out set, and no way to improve it except by producing a sign matrix with a
larger absolute determinant.
