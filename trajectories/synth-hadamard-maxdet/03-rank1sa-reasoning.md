The annealing worked — it broke the Jacobsthal symmetry and climbed from multiplier `49` to about
`150` — and the feedback named exactly what stopped it: per-flip cost. Every candidate flip paid
for a full `slogdet`, an `O(n³)` factorization, so a single-seed run could afford only tens of
thousands of flips, and it plateaued in the low `150`s while the frontier sits near `197` and the
record at `320`. The idea is right; the budget is the problem. And I can size the problem precisely
by comparing the two rungs' numbers: the baseline `49 → 149.87` is a determinant ratio of about
`3.06×`, bought by `40k` flips; the remaining climb to the `197` band is another `~1.3×` in
determinant, which — if the returns per flip keep diminishing the way local search usually does —
plausibly needs not `1.3×` more flips but one to two *orders of magnitude* more. If I could evaluate
a flip in `O(1)` instead of `O(n³)`, I could run hundreds of times longer and restart from many
seeds, and the same annealing would keep climbing. So the whole rung is one observation: a
single-entry flip is a *rank-one* perturbation of the matrix, and rank-one updates of a determinant
are nearly free if I carry the inverse.

Here is the structure the previous rung was throwing away. Flipping entry `(i, j)` from `M_{ij}` to
`−M_{ij}` adds `Δ · e_i e_jᵀ` to `M`, with `Δ = −2 M_{ij}` (if the entry is `+1` it drops by `2`,
if `−1` it rises by `2`). This is a rank-one update, and the matrix determinant lemma gives the new
determinant in closed form: `det(M + Δ e_i e_jᵀ) = det(M) · (1 + Δ · e_jᵀ M⁻¹ e_i) = det(M) · (1 +
Δ · (M⁻¹)_{ji})`. So if I already hold `M⁻¹`, the *ratio* of the new determinant to the old is just
`1 + Δ · (M⁻¹)_{ji}` — a single multiply and a single add, reading one entry of the stored inverse,
`O(1)`, no factorization at all. The log-ratio I anneal on is `log|1 + Δ · (M⁻¹)_{ji}|`, computed
instantly for any candidate `(i, j)`. I do not recompute anything to *score* a flip; I just index
into the inverse. That is the entire difference from the previous rung, and it is a difference in
asymptotic class, not a constant factor: scoring drops from `O(n³)` to `O(1)`.

I want to be sure the lemma is right and not merely plausible, and there is a clean check that ties
it to something I already trust — the linearity of the determinant in a single entry. Expanding
`det(M)` along row `i` gives `det(M) = Σ_j M_{ij} C_{ij}`, where `C_{ij}` is the signed cofactor,
which does not depend on `M_{ij}` itself; so the determinant is *linear* in the entry `M_{ij}` with
slope `C_{ij} = ∂ det / ∂ M_{ij}`. Flipping the entry changes it by `Δ = −2 M_{ij}`, so the exact
change in determinant is `Δ · C_{ij}`, i.e. `det(M') = det(M) + Δ C_{ij}`. Now use the standard
identity `(M⁻¹)_{ji} = C_{ij} / det(M)` (the inverse is the transposed cofactor matrix over the
determinant). Substituting, `det(M') = det(M) + Δ · det(M) · (M⁻¹)_{ji} = det(M)·(1 + Δ (M⁻¹)_{ji})`
— exactly the matrix determinant lemma. So the `O(1)` ratio is not a new object to distrust; it is
the cofactor expansion I already believe, read through the stored inverse. As a degenerate sanity
point, if a candidate flip lands on a singular matrix the ratio is exactly `0` (`1 + Δ(M⁻¹)_{ji} =
0` precisely when `Δ C_{ij} = −det(M)`), the log is `−∞`, and the Metropolis rule rejects it — the
same clean handling of singular candidates the previous rung got from `slogdet` returning `−∞`.

The only real cost is keeping `M⁻¹` current when I *accept* a flip, and Sherman–Morrison handles
that exactly. For a rank-one update `M ← M + Δ e_i e_jᵀ`, the inverse updates as `M⁻¹ ← M⁻¹ − Δ ·
(M⁻¹ e_i)(e_jᵀ M⁻¹) / (1 + Δ (M⁻¹)_{ji})` — the denominator is the very ratio I already computed to
score the move, and the numerator is the outer product of column `i` of `M⁻¹` with row `j` of
`M⁻¹`, an `O(n²)` operation. Crucially this is paid *only on accepted moves*, not on every
candidate. So the accounting inverts relative to the previous rung: scoring a candidate is `O(1)`,
and the occasional `O(n²)` inverse refresh is spent only when a move is actually taken. Put the
numbers at `n = 29`: the old rung paid roughly `(2/3)·29³ ≈ 1.6×10^4` float operations to score
*each* candidate; this rung pays about `2` operations to score a candidate and, on an accept,
`29² = 841` to refresh the inverse. Even if every proposal were accepted, that is `~843` versus
`~1.6×10^4` per step, and since the accepted fraction is well below one once the schedule cools, the
effective per-candidate cost falls by a large multiple — but the headline is not the constant
factor, it is that the factorization is gone from the inner loop entirely, so I can push the step
budget from `40k` into the millions without the wall-clock exploding. Concretely I plan `1.5M`
flips per chain, roughly `37×` the previous budget, and expect the per-flip throughput to rise on
top of that.

I have to be careful about numerical drift, because I am now maintaining `M⁻¹` incrementally over
more than a million rank-one updates per chain and floating-point error accumulates in the carried
inverse. Two guards, and a fallback. First, the matrix entries stay exactly `±1` throughout — I
update `M_{ij} += Δ` and it lands back on `∓1` exactly, an integer, so `M` itself never drifts; only
the *carried* inverse does. Second, the accept/reject decisions are driven by the determinant ratio,
which is somewhat self-correcting: a badly stale inverse would produce ratios that disagree with the
true determinant and, being wrong, would tend to propose moves that a correct evaluator rejects — so
gross drift shows up as degraded search rather than silent corruption of the answer. Third and
decisive: the final answer is never trusted to float arithmetic at all. I take the best `±1` matrix
the search ever recorded and recompute its determinant exactly with the harness's Bareiss integer
elimination. The float inverse is a fast *guide*; the reported number is exact integer arithmetic on
a genuine sign matrix. For a much longer run one could periodically refactor `M⁻¹` from scratch to
reset accumulated error — an `O(n³)` reset amortized over many `O(1)` steps — but I can bound the
drift over one chain and see that it stays negligible. Each Sherman–Morrison update injects a
relative error of order `ε·κ` with `ε = 2.2×10^{−16}` and `κ` the condition number (`~10^{3}` for
these near-singular sign matrices). Over `K = 1.5×10^{6}` updates the accumulation is at worst
linear, `K·ε·κ ≈ 1.5×10^{6}·2.2×10^{−16}·10^{3} ≈ 3×10^{−7}`, and if the per-step errors behave
more like a random walk it is `√K·ε·κ ≈ 1224·2.2×10^{−13} ≈ 3×10^{−10}`. Either figure is orders of
magnitude below the `~0.02` log-scale at which accept/reject decisions turn, so the carried inverse
stays faithful enough to guide the search across an entire `1.5M`-flip chain without a mid-run
refactor. The refactor is a safety valve I note rather than need — it would only start to matter at
much larger `n` or much longer chains, where `K·ε·κ` climbs toward the decision scale.

With scoring made free, two things I could not afford before become cheap, and both matter. First,
*budget*: instead of `40k` flips I run `1.5M` per chain, so the annealing has the room to make the
long sequence of mostly-lateral moves that coordinated determinant gains require — the previous rung
argued the escape from a symmetric extremum is a *sequence* of individually-neutral-or-worse moves,
and a longer sequence can reach configurations a short one cannot, escaping not just the Jacobsthal
basin but the secondary plateaus the cheap rung stalled on. Second, *restarts from structured
seeds*. The Jacobsthal matrix is one point, but the prime `29` hands me a whole family of
equally-valid relabelings for free. Reindex rows and columns by `i ↦ k·i (mod 29)` for a unit `k`;
since `29` is prime, every `k ∈ {1, …, 28}` is a unit and the map is a bijection. Under it the
Jacobsthal entries transform as `Q_{ki, kj} = χ(k(i−j)) = χ(k)·χ(i−j)`, so the relabeled seed is
either a genuine permutation of `Q` (when `k` is a quadratic residue, `χ(k) = +1`) or of `−Q` (when
`k` is a non-residue, `χ(k) = −1`) — in both cases a legal `±1` design with the same baseline
determinant `m = 49`, but sitting in a *different* basin of the single-flip landscape because the
neighbor structure around it is relabeled. Annealing from each of several such seeds explores
different regions and I keep the global best. This is diversity that costs nothing but compute, and
compute is now the thing I have.

There is a real choice here between spending the freed budget as one very long chain or as several
restarts, and it is worth settling by argument rather than reflex. The previous rung's plateau was a
basin phenomenon: within one connected basin, once the temperature cools the improving moves get
rarer and the marginal return per flip falls off. Pouring all `10.5M` flips into a *single* chain
would buy an ever-diminishing return inside whichever basin that one seed leads to — the tail of a
saturating curve. Splitting the same budget across seven independent chains from seven distinct
basins instead samples the *spread* of basin qualities, and since the best-of-seven is what I report,
I gain from the variance across seeds, not just the depth within one. The counter-risk is that seven
`1.5M` chains each individually under-explore relative to one `10.5M` chain — but `1.5M` is already
`37×` the budget that reached the `150`s, well past the point where a single chain saturates its
basin, so each restart is long enough to bottom out its own basin and the marginal flip is better
spent opening a new basin than deepening an exhausted one. That is the case for multi-start over one
long chain, and it is only affordable because the `O(1)` score made `10.5M` total flips reachable at
all. I keep it deliberately simple — independent restarts, best-of — rather than reaching for
parallel tempering or a population method, which would buy cross-basin mixing at the price of two or
three more schedule knobs I cannot yet justify from data.

The cooling schedule itself I reuse unchanged, and there is a reason it transfers rather than needing
a retune. The `O(1)` trick changes only *how* I score a flip, not *what* the flip does to the
determinant: the distribution of `Δlog|det|` over candidate flips is a property of the objective and
the configuration, identical to the previous rung's, so the temperature bracket `0.06 → 2×10^{−4}`
that matched that `Δlog` scale still matches it. What changes is only the number of steps the glide
is spread over. Cooling geometrically from `0.06` to `2×10^{−4}` over `N = 1.5×10^{6}` steps gives a
per-step decay of `(2×10^{−4}/0.06)^{1/N} ≈ 0.9999962`, which halves the temperature about every
`ln(0.5)/ln(0.9999962) ≈ 1.8×10^{5}` steps — so still roughly `8` halvings across the chain, the
same warm-to-cold *shape* as the `40k` run, just stretched so each temperature level gets far more
proposals. That is exactly what I want: the same annealing profile that worked, given far more time
at every temperature to find the long lateral sequences the escape needs.

Which `k`'s? I want a spread, not near-duplicates. Note that `−1` is a residue mod `29`, so
`χ(−k) = χ(k)`, and the relabeling by `29 − k = −k` is the coordinate-reversed mirror of the
relabeling by `k` — so `k` and `29 − k` give mirror-related basins and I should take one
representative from each pair rather than both. I pick `k ∈ {1, 2, 3, 6, 10, 12, 15}`: seven small,
well-separated units, of which `1` and `6` are quadratic residues (`χ(k) = +1`, permutations of
`Q`) and `2, 3, 10, 12, 15` are non-residues (`χ(k) = −1`, permutations of `−Q`), so the set mixes
both flavors of seed. Seven chains of `1.5M` flips is `10.5M` flips in total — three orders of
magnitude past the previous rung's `40k` — reachable only because each flip is now `O(1)` to score.

One temptation I will resist: now that any rank-one change is `O(1)` to score, I could enlarge the
move set — a whole-row rewrite `M ← M + e_i vᵀ` is also rank-one and equally cheap to evaluate by
the same lemma. But a whole-row *negation* only multiplies the determinant by `±1` (useless, as the
previous rung established), and a general row rewrite is a genuinely different, larger move whose
effect I have not characterized. If I change both the budget *and* the move set at once, I cannot
attribute the resulting climb to either, and this rung's entire claim is that the previous plateau
was a *budget* wall — so I hold the single-entry move fixed and vary only the budget and the seeds.
That keeps the comparison to the previous rung clean; a richer move set is a separate lever for a
later rung if this one plateaus.

So the rung is concrete: carry `M⁻¹`; score every candidate flip in `O(1)` via the determinant
ratio `1 + Δ (M⁻¹)_{ji}`; anneal on `log|ratio|` with the same warm-to-cold schedule that worked
before (`log`-temperature `0.06` decaying geometrically to `2×10^{−4}`); refresh the inverse by
Sherman–Morrison, `O(n²)`, only on accepts; run `1.5M` flips from each of the seven
multiplier-relabeled Jacobsthal seeds with a fixed RNG per chain; keep the global-best `±1` matrix;
recompute its determinant exactly with Bareiss. What I expect is the same annealing dynamics as the
previous rung but run two-plus orders of magnitude longer and from multiple basins — so it should
clear the low-`150`s plateau and push the multiplier into the `180`s, roughly the neighborhood of
the best machine-discovered results reported for this order (`~197`), bought entirely by making the
flip cheap enough to afford the steps. If the multi-seed spread matters, I also expect the best seed
to noticeably beat the worst, which would confirm that the free structured diversity is buying a
real slice of the climb rather than seven chains all converging to the same value.

Let me make the expectation falsifiable in the task's own units before I run it. On the frozen
`m/342` score, a multiplier in the `180`s reads as roughly `0.53`–`0.54`, so if the diagnosis is
right I should see the single `n29` metric move from the previous rung's `0.438` to around `0.54`, a
`+0.10` jump — smaller than the baseline-to-annealing jump, exactly as diminishing returns predict,
but real. Two secondary signatures would corroborate the mechanism rather than a lucky seed. First,
throughput: with scoring at `O(1)` instead of a full factorization, the chain should evaluate on the
order of hundreds of thousands of flips per second rather than the previous rung's tens of thousands,
so `10.5M` total flips finish in tens of seconds of wall-clock — if instead it crawls, my cost model
is wrong. Second, the multi-start spread: I expect the best of the seven relabeled seeds to beat the
worst by a clear margin in multiplier, which is the fingerprint that the free structured diversity is
buying a genuine slice of the climb; if all seven converged to the same value, the restart machinery
would be dead weight and one long chain would have been the better spend. Those are the three things
I will read off the feedback — the score jump, the throughput, and the seed spread — and each is a
number the mechanism commits me to.

And here is where I stop, honestly, because this rung buys the frontier of one kind of search and
not the record. The classical record at `n = 29` is multiplier `320`, attributed to Orrick and
Solomon, and — as the initial framing already flags — the best known values at non-Hadamard orders
come from large-scale search over *Gram matrices* and their admissible `±1` factorizations, not over
raw entry flips. That is a fundamentally different search space: it optimizes the symmetric integer
design `G = R Rᵀ` directly, in the small rigid space of admissible Gram matrices, and only decomposes
the winning `G` back into a `±1` matrix afterward. Single-entry flip annealing, however cheaply I
can afford it, is a walk on the fixed landscape of `±1` matrices where neighbors differ in one sign,
and the record is not a short coordinated walk from any structured seed I can start from. So I do not
expect this rung to reach `320`; no program-evolution system has, and the strongest reported machine
result sits near score `0.576`, exactly the band my `180`s multiplier would land in. That gap is not
a failure of the rank-one trick — the trick does precisely its job, converting the previous rung's
cost wall into two-plus orders of magnitude more steps and a real push up the curve. It is the honest
shape of the problem: local entry-flip annealing tops out near the machine frontier, the Gram-space
record stands above it, and the distance between them is the part of the `n = 29` maximal-determinant
problem that a local-move constructor is not built to close.
