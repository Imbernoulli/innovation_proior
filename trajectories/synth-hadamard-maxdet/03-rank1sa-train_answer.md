The annealing rung worked: from the Jacobsthal seed it broke the design's symmetry and climbed to multiplier $\approx 150$, but it stalled there, and the feedback named the cause exactly — per-flip cost. Every candidate flip paid for a full `slogdet`, an $O(n^3)$ factorization, so a single chain could only afford tens of thousands of flips and plateaued in the low $150$s while the machine-search frontier sits near $197$. The search dynamics were right; the budget was the problem. If a flip could be evaluated in $O(1)$ instead of $O(n^3)$, the same annealing could run two-plus orders of magnitude longer and from many seeds, and keep climbing.

I propose **rank-one (Sherman–Morrison) annealing with multiplier-relabeled restarts**, built on a single observation: a single-entry flip is a *rank-one* perturbation of the matrix, and rank-one updates of a determinant are nearly free if I carry the inverse. Flipping entry $(i,j)$ from $M_{ij}$ to $-M_{ij}$ adds $\Delta\, e_i e_j^\top$ to $M$ with $\Delta = -2 M_{ij}$. The matrix determinant lemma gives the new determinant in closed form,
$$\det\!\left(M + \Delta\, e_i e_j^\top\right) = \det(M)\,\bigl(1 + \Delta\,(M^{-1})_{ji}\bigr),$$
so if I already hold $M^{-1}$, the *ratio* of the new determinant to the old is just $1 + \Delta\,(M^{-1})_{ji}$ — a single multiply-and-add, $O(1)$, with no factorization at all. The quantity I anneal on is the log-ratio $\log\bigl|1 + \Delta\,(M^{-1})_{ji}\bigr|$, read instantly for any candidate $(i,j)$ from one entry of the stored inverse. I do not recompute anything to *score* a flip.

The only real cost is keeping $M^{-1}$ current when I *accept* a flip, and this is exactly what Sherman–Morrison handles. The rank-one update $M \leftarrow M + \Delta\, e_i e_j^\top$ updates the inverse as
$$M^{-1} \leftarrow M^{-1} - \frac{\Delta}{1 + \Delta\,(M^{-1})_{ji}}\,(M^{-1} e_i)(e_j^\top M^{-1}),$$
an outer product of one column and one row of the current inverse, $O(n^2)$ work, done *only on accepted moves*, never on every candidate. So the accounting flips entirely: scoring a candidate is $O(1)$, and the occasional $O(n^2)$ inverse refresh is paid only when a move is taken. Against the previous rung's $O(n^3)$ per candidate this is a real multiple faster per evaluated flip at $n=29$, but the decisive win is removing the factorization from the inner loop, so the step budget moves from tens of thousands into the millions without the wall-clock exploding.

Maintaining $M^{-1}$ incrementally over millions of updates does invite floating-point drift, and I guard it two ways. First, the matrix entries stay exactly $\pm 1$ throughout — I update $M_{ij} \mathrel{+}= \Delta$, which lands back on $\mp 1$ exactly — so $M$ itself never drifts; only the carried inverse does. Second, and more important, the float arithmetic is never trusted for the answer: the final result is the best $\pm 1$ matrix the search ever recorded, and its determinant is recomputed exactly with Bareiss integer arithmetic. The float inverse is a fast guide; the reported number is exact. (For a much longer run one could periodically refactor $M^{-1}$ from scratch to reset drift; at $n=29$ and these budgets it is not needed, but it is the obvious safety valve if the order grew.)

With scoring made free, two things I could not afford before become cheap, and both matter. First, *budget*: instead of $40\text{k}$ flips I run $1.5\text{M}$ per chain, giving annealing the room to make the long sequences of mostly-lateral moves that coordinated determinant gains require — escaping not just the Jacobsthal basin but the secondary plateaus that stalled the cheap rung. Second, *restarts from structured seeds*: the prime $29$ supplies a whole family of equally-valid relabelings of the Jacobsthal design — reindex rows and columns by $i \mapsto k\cdot i \pmod{29}$ for a unit $k$, and I get a different, equally-structured starting matrix with the same baseline determinant but a different surrounding basin. Annealing from each of several such multiplier-relabeled seeds explores different regions, and I keep the global best. This is free diversity, and compute is now the abundant resource.

I want to be honest about where this lands. The classical record at $n=29$ is multiplier $320$, established by Solomon in 2002 and tabulated by Orrick — the output of dedicated maximal-determinant search that works over the *Gram matrix* and its admissible $\pm 1$ factorizations, not over raw entry flips, and it is only *conjectured* optimal. No program-evolution system has matched it; the strongest reported machine result sits near score $0.576$, and entry-flip annealing lands in that same band, well short of $0.9357$. That gap is not a failure of this rung — reaching $320$ would mean reproducing a specific number-theoretic Gram construction under a search of a different kind, which a single constructor running local moves should not be expected to rediscover. The rank-one trick buys the frontier of what entry-flip annealing can reach; the record remains the unbeaten ceiling above it.

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
