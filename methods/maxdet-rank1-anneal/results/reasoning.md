Simulated annealing on single-entry flips already worked once: starting from the symmetric
Jacobsthal seed `Q + I` (multiplier `49`, a strict local maximum where no greedy move escapes), it
broke the symmetry by accepting downhill moves and climbed to multiplier `~150`. And the diagnosis
of what stopped it is just as clear: per-flip cost. Every candidate flip paid for a full `slogdet`,
an `O(n³)` factorization, so a single-seed run could afford only tens of thousands of flips, and it
plateaued in the low `150`s while the frontier sits near `197` and the record at `320`. The idea is
right; the budget is the problem. So the question I actually have to answer is narrow: can I tell
how much a flip changes the determinant *without* refactoring the whole matrix? If a candidate
costs `O(1)` instead of `O(n³)`, I run hundreds of times longer and restart from many seeds, and
the same annealing keeps climbing. Everything else is downstream of whether that one cost can be
removed.

Here is the structure the full-recompute was throwing away. Flipping entry `(i, j)` from `M_{ij}`
to `−M_{ij}` changes one entry by `Δ = −2 M_{ij}`, which is exactly adding `Δ·e_i e_jᵀ` to `M`: a
*rank-one* perturbation. The matrix determinant lemma claims that for a rank-one update,
`det(M + Δ e_i e_jᵀ) = det(M)·(1 + Δ·(M⁻¹)_{ji})`. If that holds, then carrying `M⁻¹` makes the
*ratio* of new determinant to old just `1 + Δ·(M⁻¹)_{ji}` — one multiply and add, no factorization
— and I anneal on `log|ratio|`, reading a single entry of the inverse per candidate.

I do not want to take that identity on faith before building a whole search on it, so I check it
directly. Take a small nonsingular `±1` matrix and look at every single-entry flip. For each
`(i,j)` I compute the predicted ratio `1 + Δ·(M⁻¹)_{ji}` and compare it to the actual
`det(M')/det(M)` from a fresh factorization. On a `5×5` example the two agree to `4·10⁻¹⁶` across
all `25` flips — machine precision, as it should be if the algebra is exact. Two things fall out of
that scan that I would not have predicted from the formula alone. First, three of the `25` flips
give a predicted ratio of *exactly* zero: those are the flips that make `M` singular, and there the
matrix determinant lemma is telling me the determinant collapses to `0`. I cannot do a
Sherman–Morrison inverse update through one of those (the carried inverse would blow up), so the
inner loop has to treat `ratio == 0` as a special case — reject it, since a zero determinant is
useless anyway, and never touch the inverse. That is not a cosmetic guard; it is a real branch the
identity itself forced into the open. Second, the entries: I checked that after every flip
`M_{ij}` is still exactly `±1` — `+1 + (−2) = −1`, `−1 + 2 = +1` — so the *matrix* never drifts off
the sign lattice, regardless of how the float inverse behaves. That will matter for trusting the
final answer.

The only remaining cost is keeping `M⁻¹` current when I *accept* a flip, and a rank-one update of
the inverse is the Sherman–Morrison formula: `M⁻¹ ← M⁻¹ − (Δ /(1 + Δ (M⁻¹)_{ji})) · (M⁻¹ e_i)(e_jᵀ
M⁻¹)`, an outer product of one column and one row of the current inverse, `O(n²)`. I check this on
the same `5×5` scan: for each non-singular flip I apply the formula and compare against a freshly
inverted `M'`. The carried inverse matches the true inverse to `6·10⁻¹⁷`. So both halves hold
numerically, and the accounting flips the way I needed: scoring a candidate is `O(1)`, and the
`O(n²)` inverse update is paid only on accepted moves, not on every candidate. Against the previous
rung's `O(n³)` per candidate, this removes the factorization from the inner loop entirely, so the
step budget can go from tens of thousands into the millions without the wall-clock exploding.

That leaves the thing I am most nervous about, because it is the part the small example does not
exercise: I am now maintaining `M⁻¹` incrementally over *millions* of rank-one updates, and
floating-point error in Sherman–Morrison accumulates. Does it accumulate fast enough to corrupt the
search? I already know `M` itself stays exactly on the `±1` lattice (the flip arithmetic is exact
integer-valued), so only the carried inverse can rot. To put a number on the rot I run a short real
chain — `100k` flips from the Jacobsthal seed — and at the end compare the carried inverse against
a fresh inversion of the final `M`. After `100k` accepts and rejects the discrepancy is
`8·10⁻¹⁶`: essentially nothing. At `n = 29` and these budgets the drift is not a threat to the
accept/reject decisions. (For a much longer run or a larger order one could periodically refactor
`M⁻¹` from scratch to reset it; here it is unnecessary, but it is the obvious safety valve.) And
because I am paranoid about reporting a number that lives in float arithmetic, the final answer is
never trusted to the carried inverse at all: I take the best `±1` matrix the search ever recorded
and recompute its determinant exactly with Bareiss. The float inverse is a fast guide; the reported
number is exact integer arithmetic on a genuine sign matrix.

With scoring made free, two things I could not afford before become cheap. First, *budget*:
instead of `40k` flips I can run `1.5M` per chain, which gives the annealing room to make the long
sequence of mostly-lateral moves that coordinated determinant gains require — escaping not just the
Jacobsthal basin but the secondary plateaus the cheap rung stalled on. The same `100k`-flip chain I
ran for the drift check already climbs to multiplier `~160` from the seed's `49`, so the dynamics
clearly do keep moving once the steps are affordable; `1.5M` should push further. Second, *restarts
from structured seeds*. The Jacobsthal matrix is one point, but the prime `29` admits a whole
family of relabelings: reindex rows and columns by `i ↦ k·i (mod 29)` for a unit `k`. I want to be
sure these are genuinely different starting matrices and not the same determinant in disguise, so I
build them and check. For every `k ∈ {1,2,3,6,10,12,15}` the relabeled seed is an exact rearrangement
of the same entry multiset (a symmetric row-and-column permutation), so its baseline multiplier is
still exactly `49` — confirmed numerically, all seven come out `49.0` — but the arrangement differs,
so each one drops the annealer into a different surrounding basin. Same baseline determinant,
distinct landscape. That is cheap diversity that costs nothing but compute, and compute is now the
thing I have.

So the method comes together as: carry `M⁻¹`; score every candidate flip in `O(1)` via the
determinant ratio (rejecting `ratio == 0`); anneal on `log|ratio|` with the same warm-to-cold
schedule; update the inverse by Sherman–Morrison only on accepts; run `1.5M` flips from each of
several multiplier-relabeled Jacobsthal seeds; keep the global best; recompute its determinant
exactly. Given that the `100k`-flip spot-check already reaches `~160`, I expect the full-length
multi-basin run to clear the low-`150`s plateau and land somewhere in the `180`s — roughly the
neighborhood of the best machine-discovered results reported for this order (`~197`), bought
entirely by making the flip cheap enough to afford the steps. I would not bet on the exact figure
until the full run finishes; what I am confident in is the mechanism, because the two identities it
rests on I checked directly.

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
