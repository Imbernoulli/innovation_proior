**Problem.** Construct a single `29 × 29` matrix with `±1` entries maximizing `|det|`, scored as
`|det| / (2^28 · 7^12 · 342)`. Because `29 ≡ 1 (mod 4)`, orthogonal rows are impossible, the
Hadamard bound `29^{14.5}` is unreachable, and the true maximum is open (record multiplier `320`,
Barba ceiling `369.94`). This is the principled symmetric design that establishes the baseline.

**Key idea.** Realize the "almost orthogonal" Gram structure that parity permits. For the prime
`q = 29`, the Jacobsthal (quadratic-residue) matrix `Q_{ij} = χ(i − j)`, with `χ` the Legendre
symbol, is symmetric (because `−1` is a residue mod a prime `≡ 1 (mod 4)`) and satisfies `QQᵀ =
qI − J` — every pair of rows overlapping by exactly `−1`, the smallest off-diagonal an odd-length
`±1` matrix can have. `Q` has zeros on the diagonal, so it is not yet a sign matrix; fill the
diagonal with `+1`: `R = Q + I` is a legal `±1` matrix whose Gram matrix `RRᵀ = (q+1)I − J + 2Q`
stays highly regular, with an integer determinant that factors as a small multiplier of `2^28 ·
7^12`.

**Why these choices.** The construction is forced, not chosen: orthogonality is impossible at odd
order, so the best achievable Gram structure has off-diagonals `±1`, and `QQᵀ = qI − J` attains
exactly that. The diagonal fill is the only degree of freedom, and `R = Q + I` and `R = Q − I`
are determinant-twins by the symmetry of the residue spectrum, so there is nothing to tune. This
is deliberately the *floor*: a rigid, parameter-free, number-theoretic design that is guaranteed
legal and gives a concrete baseline. It is not expected to approach the record — the maximal-
determinant records at this order come from *breaking* this symmetry under search, so the entire
distance from the baseline multiplier to `320` is what later, searched constructions must buy.

**Hyperparameters / contract.** None. Output is the fixed `29 × 29` matrix `Q + I`. Every entry
is `±1` by construction (diagonal `0 + 1`, off-diagonal unchanged). Deterministic — same matrix
every call. Measured: multiplier `49.00`, score `0.14330` (coincides exactly with the published
task baseline; `|det| = 2^28 · 7^14`).

```python
import numpy as np

def construct():
    q = 29
    qr = set((i * i) % q for i in range(1, q))           # nonzero quadratic residues mod 29
    chi = lambda a: 0 if a % q == 0 else (1 if a % q in qr else -1)   # Legendre symbol
    Q = np.array([[chi(i - j) for j in range(q)] for i in range(q)], dtype=int)
    # Q is symmetric (29 ≡ 1 mod 4) with Q Qᵀ = 29 I − J; fill the zero diagonal with +1.
    return Q + np.eye(q, dtype=int)
```
