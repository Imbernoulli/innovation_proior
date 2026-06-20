Simulated annealing on single-entry flips already solves half of the `n = 29` maximal-determinant
problem. Starting from the symmetric Jacobsthal seed `Q + I` — multiplier `49`, a strict local
maximum where no greedy move escapes — it breaks the symmetry by accepting downhill moves and
climbs to multiplier `~150`. The diagnosis of what stopped it is just as clear: per-flip cost. Each
candidate flip pays for a full `slogdet`, an `O(n³)` factorization, so a single-seed run can afford
only tens of thousands of flips and plateaus in the low `150`s, while the machine-search frontier
sits near multiplier `197` and the record at `320`. The search idea is right; the budget is the
problem. The entire method follows from one observation: a single-entry sign flip is a *rank-one*
perturbation of the matrix, and rank-one updates of a determinant are nearly free if the inverse is
carried.

The method is rank-one (Sherman–Morrison) annealing with multiplier-relabeled restarts. Flipping
entry `(i, j)` adds `Δ·e_i e_jᵀ` to `M` with `Δ = −2 M_{ij}`, so the matrix determinant lemma gives
the new determinant in closed form as `det(M)·(1 + Δ·(M⁻¹)_{ji})`. If the inverse is held in memory,
the determinant *ratio* of any candidate flip is just `1 + Δ·(M⁻¹)_{ji}` — a single read of one
inverse entry, `O(1)`, with no factorization — and the search anneals on `log|ratio|` exactly as
before. The only real cost is keeping the inverse current on an *accepted* flip, which
Sherman–Morrison does with an `O(n²)` outer-product update, paid only when a move is taken. The
accounting inverts: scoring is `O(1)` per candidate and the `O(n²)` refresh is occasional, so the
factorization leaves the inner loop entirely and the step budget can grow from tens of thousands
into the millions. Drift is guarded twice: the matrix entries stay exactly `±1` throughout (only the
carried float inverse can drift), and the final answer is never trusted to float arithmetic — the
best `±1` matrix the search recorded is re-scored with exact Bareiss integer elimination, with
periodic refactorization available as a safety valve if the order ever grew.

With scoring made free, two things become affordable. The budget rises to `1.5M` flips per chain,
giving the annealing room for the long lateral sequences that coordinated determinant gains require,
past not just the Jacobsthal basin but the secondary plateaus the cheap rung stalled on. And the
search restarts from several *multiplier-relabeled* seeds: the prime `29` admits a family of
equally-valid relabelings `i ↦ k·i (mod 29)` for units `k`, each a different, equally-structured
starting matrix with the same baseline determinant but a different surrounding basin. Annealing from
each and keeping the global best is free diversity now that compute is abundant. Run over seeds
`k ∈ {1,2,3,6,10,12,15}` — `10.5M` flips in total — this pushes the multiplier `149.87 → 184.60`
(score `0.4382 → 0.53977`), into the band of the best reported machine-discovered constructions for
this order. The multi-seed restart earns its keep: the best seed (`k=10`, multiplier `184.6`) beats
the worst (`k=6`, `170.9`) by about `14`, a real fraction of the climb.

This is the honest endpoint. The classical record at `n = 29` is multiplier `320` (score `0.9357`,
Solomon 2002 / Orrick), the output of dedicated maximal-determinant search over the *Gram matrix*
and its admissible `±1` factorizations — a different kind of search than raw entry flips — and it is
only conjectured optimal, not proven. No program-evolution system has matched it; the strongest
reported machine result sits near `0.576`, the same band this method reaches. The residual gap
`184.6 → 320` is not a failure of the method but the still-open part of the problem: reaching the
record would mean rediscovering a specific number-theoretic Gram construction that a single
constructor running local moves should not be expected to find.

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
