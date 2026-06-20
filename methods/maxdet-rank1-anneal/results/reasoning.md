Simulated annealing on single-entry flips already worked once: starting from the symmetric
Jacobsthal seed `Q + I` (multiplier `49`, a strict local maximum where no greedy move escapes), it
broke the symmetry by accepting downhill moves and climbed to multiplier `~150`. And the diagnosis
of what stopped it is just as clear: per-flip cost. Every candidate flip paid for a full `slogdet`,
an `O(n³)` factorization, so a single-seed run could afford only tens of thousands of flips, and it
plateaued in the low `150`s while the frontier sits near `197` and the record at `320`. The idea is
right; the budget is the problem. If I could evaluate a flip in `O(1)` instead of `O(n³)`, I could
run hundreds of times longer and restart from many seeds, and the same annealing would keep
climbing. So the whole method is one observation: a single-entry flip is a *rank-one* perturbation
of the matrix, and rank-one updates of a determinant are nearly free if I carry the inverse.

Here is the structure I was wasting. Flipping entry `(i, j)` from `M_{ij}` to `−M_{ij}` adds
`Δ·e_i e_jᵀ` to `M`, with `Δ = −2 M_{ij}`. The matrix determinant lemma gives the new determinant
in closed form: `det(M + Δ e_i e_jᵀ) = det(M)·(1 + Δ·(M⁻¹)_{ji})`. So if I already hold `M⁻¹`,
the *ratio* of the new determinant to the old is just `1 + Δ·(M⁻¹)_{ji}` — a single multiply and
add, `O(1)`, no factorization at all. The log-ratio I anneal on is `log|1 + Δ·(M⁻¹)_{ji}|`,
computed instantly for any candidate `(i, j)`. I do not recompute anything to *score* a flip; I
just read one entry of the inverse.

The only real cost is keeping `M⁻¹` current when I *accept* a flip, and Sherman–Morrison handles
that. A rank-one update `M ← M + Δ e_i e_jᵀ` updates the inverse as `M⁻¹ ← M⁻¹ − (Δ /(1 + Δ
(M⁻¹)_{ji})) · (M⁻¹ e_i)(e_jᵀ M⁻¹)` — an outer product of one column and one row of the current
inverse, `O(n²)` work, done only on accepted moves, not on every candidate. So the accounting
flips: scoring a candidate is `O(1)`, and the occasional `O(n²)` inverse update is paid only when
a move is taken. Against the previous full-recompute rung's `O(n³)` per candidate, at `n = 29` this
is a real multiple faster per evaluated flip, and — more importantly — it removes the factorization
from the inner loop entirely, so I can push the step budget from tens of thousands into the millions
without the wall-clock exploding.

I have to be careful about numerical drift, because I am now maintaining `M⁻¹` incrementally over
millions of rank-one updates and floating-point error accumulates. Two guards. First, the matrix
entries stay exactly `±1` throughout — I update `M_{ij} += Δ` which lands back on `∓1` exactly — so
`M` itself never drifts; only the carried inverse does. Second, the accept/reject decisions are
driven by the determinant ratio, which is self-correcting in the sense that a badly stale inverse
would produce ratios that disagree with reality and get rejected; but to be safe the final answer
is never trusted to the float arithmetic at all — I take the best `±1` matrix the search ever
recorded and recompute its determinant exactly with Bareiss. The float inverse is a fast guide; the
reported number is exact integer arithmetic on a genuine sign matrix. (For a long enough run one
could periodically refactor `M⁻¹` from scratch to reset drift; at `n = 29` and these budgets it is
not needed, but it is the obvious safety valve if the order grew.)

With scoring made free, two things I could not afford before become cheap, and both matter.
First, *budget*: instead of `40k` flips I run `1.5M` per chain, so the annealing has the room to
make the long sequence of mostly-lateral moves that coordinated determinant gains require —
escaping not just the Jacobsthal basin but the secondary plateaus the cheap rung stalled on.
Second, *restarts from structured seeds*: the Jacobsthal matrix is one point, but the prime `29`
gives a whole family of equally-valid relabelings — reindex the rows and columns by `i ↦ k·i (mod
29)` for a unit `k`, and I get a different, equally-structured starting matrix with the same
baseline determinant but a different surrounding basin. Annealing from each of several such
multiplier-relabeled seeds explores different regions of the landscape, and I keep the best result
across all of them. This is cheap diversity that costs nothing but compute, and compute is now the
thing I have.

So the method is: carry `M⁻¹`; score every candidate flip in `O(1)` via the determinant ratio;
anneal on `log|ratio|` with the same warm-to-cold schedule; update the inverse by Sherman–Morrison
only on accepts; run `1.5M` flips from each of several multiplier-relabeled Jacobsthal seeds; keep
the global best; recompute its determinant exactly. What I expect is the same annealing dynamics as
the cheaper rung but run two-plus orders of magnitude longer and from multiple basins — so it
should clear the low-`150`s plateau and push the multiplier into the `180`s, roughly the
neighborhood of the best machine-discovered results reported for this order (`~197`), bought
entirely by making the flip cheap enough to afford the steps.

And here is where I stop, honestly. The classical record at `n = 29` is multiplier `320`,
established by Solomon in 2002 and tabulated by Orrick — and it is the output of dedicated
maximal-determinant search infrastructure that works over the *Gram matrix* and its admissible
`±1` factorizations, not over raw entry flips, and it is only *conjectured* optimal, not proven.
No program-evolution system has matched it; the strongest reported machine result sits near `0.576`
in score, and my entry-flip annealing lands in the same general band, well short of `0.9357`. That
gap is not a failure of this method — it is the honest shape of the problem. Reaching `320` would
mean reproducing a specific number-theoretic Gram construction under a search of a different kind,
which a single constructor running local moves should not be expected to rediscover. So this is the
endpoint: the rank-one trick buys the frontier of what entry-flip annealing can reach, the record
remains the unbeaten ceiling above it, and the distance between them is exactly the part of the
maximal-determinant problem that is still open.
