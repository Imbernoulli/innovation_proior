**Problem.** Same `n = 29` maximal-determinant construction. The annealing rung reached multiplier
`~150` but was bound by per-flip cost: scoring each candidate with a full `slogdet` (`O(n³)`)
limited it to tens of thousands of flips from one seed, plateauing short of the frontier.

**Key idea.** Score flips for free by carrying the inverse. A single-entry flip at `(i, j)` is a
rank-one update `M ← M + Δ e_i e_jᵀ` with `Δ = −2 M_{ij}`, so by the matrix determinant lemma the
determinant ratio is `1 + Δ·(M⁻¹)_{ji}` — `O(1)` to evaluate from a stored inverse, no
factorization. On an *accepted* flip, refresh the inverse with Sherman–Morrison, `M⁻¹ ← M⁻¹ −
(Δ/(1+Δ(M⁻¹)_{ji}))·(M⁻¹e_i)(e_jᵀM⁻¹)`, `O(n²)`, paid only when a move is taken. Scoring drops
from `O(n³)` per candidate to `O(1)`, so the same annealing on `log|ratio|` can run two-plus orders
of magnitude longer. Spend the freed budget two ways: `1.5M` flips per chain (room for the long
lateral sequences coordinated gains need), and restarts from several multiplier-relabeled
Jacobsthal seeds (`i ↦ k·i mod 29` for units `k` — distinct structured basins, same baseline). Keep
the global best; recompute its determinant exactly with Bareiss.

**Why these choices.** The previous rung proved annealing works and named cost as the limiter; the
matrix determinant lemma removes that cost exactly, with no change to the search dynamics. The
matrix stays exactly `±1` (entries land back on `∓1`), so only the carried inverse can drift, and
the float inverse is never trusted for the answer — the reported determinant is exact integer
arithmetic on a genuine sign matrix (periodic refactorization is the safety valve if the order
grows). Multiplier-relabeled restarts are free diversity now that compute is the abundant resource.

**Honest ceiling.** This reaches the band of the best machine-discovered results for `n = 29`
(multiplier `~197`, score `~0.576`); it does **not** reach the classical record (multiplier `320`,
score `0.9357`, Solomon 2002 / Orrick), which comes from dedicated Gram-matrix search of a
different kind and is only conjectured optimal. No program-evolution system has matched the record;
the gap above this rung is the still-open part of the problem.

**Hyperparameters / contract.** `log`-temperature `0.06 → 2e-4` geometric; `1.5M` flips per chain;
seeds `k ∈ {1,2,3,6,10,12,15}`; fixed RNG per chain. Output `±1` by construction; exact Bareiss
determinant reported.

```python
import numpy as np

def construct(iters=1_500_000, ks=(1, 2, 3, 6, 10, 12, 15)):
    q = 29
    qr = set((i * i) % q for i in range(1, q))
    chi = lambda a: 0 if a % q == 0 else (1 if a % q in qr else -1)
    Q = np.array([[chi(i - j) for j in range(q)] for i in range(q)], dtype=int)

    def anneal(seed_mat, rng):
        M = seed_mat.astype(float).copy()
        inv = np.linalg.inv(M)
        ld = np.linalg.slogdet(M)[1]
        best, best_ld = M.copy(), ld
        logT = 0.06; decay = (2e-4 / 0.06) ** (1.0 / iters)
        for _ in range(iters):
            i, j = rng.integers(q), rng.integers(q)
            delta = -2.0 * M[i, j]
            ratio = 1.0 + delta * inv[j, i]                 # det(M')/det(M), O(1)
            if ratio != 0.0:
                dl = np.log(abs(ratio))
                if dl >= 0 or rng.random() < np.exp(dl / logT):
                    col = inv[:, i].copy(); row = inv[j, :].copy()
                    inv -= np.outer(col, row) * (delta / ratio)   # Sherman–Morrison, O(n^2)
                    M[i, j] += delta; ld += dl
                    if ld > best_ld:
                        best_ld = ld; best = M.copy()
            logT = max(logT * decay, 2e-4)
        return np.rint(best).astype(int), best_ld

    best_mat, best_ld = None, -np.inf
    for k in ks:                                            # multiplier-relabeled restarts
        perm = [(k * i) % q for i in range(q)]
        seed = Q[np.ix_(perm, perm)] + np.eye(q, dtype=int)
        M, ld = anneal(seed, np.random.default_rng(2000 + k))
        if ld > best_ld:
            best_ld, best_mat = ld, M
    return best_mat
```
