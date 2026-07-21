The annealing broke the Jacobsthal symmetry and climbed from `49` to about `150`, and the feedback
named exactly what stopped it: per-flip cost. Every candidate paid for a full `slogdet`, an `O(n³)`
factorization, so a single-seed run afforded only tens of thousands of flips and plateaued in the low
`150`s while the frontier sits near `197`. The idea is right; the budget is the problem. And I can size
it: the `49 → 150` climb was a determinant ratio of about `3×` bought by `40k` flips; the remaining
climb toward `197` is another `~1.3×`, which — if returns per flip keep diminishing the way local
search does — plausibly needs one to two *orders of magnitude* more flips. If I could score a flip in
`O(1)` instead of `O(n³)`, I could run hundreds of times longer and restart from many seeds. And a
single-entry flip is a *rank-one* perturbation, whose determinant update is nearly free if I carry the
inverse.

Flipping entry `(i, j)` from `M_{ij}` to `−M_{ij}` adds `Δ·e_i e_jᵀ` to `M`, with `Δ = −2 M_{ij}`. The
matrix determinant lemma gives `det(M + Δ e_i e_jᵀ) = det(M)·(1 + Δ·(M⁻¹)_{ji})`, so holding `M⁻¹`,
the *ratio* of new determinant to old is `1 + Δ·(M⁻¹)_{ji}` — one multiply, one add, one stored entry,
`O(1)`, no factorization. The log-ratio I anneal on is `log|1 + Δ·(M⁻¹)_{ji}|`. This is not a
constant-factor win: scoring drops from `O(n³)` to `O(1)`. If a candidate lands on a singular matrix
the ratio is exactly `0`, the log `−∞`, and Metropolis rejects it — the same clean singular-handling
`slogdet` gave.

The only real cost is keeping `M⁻¹` current on accepted flips, and Sherman–Morrison does it exactly:
`M⁻¹ ← M⁻¹ − Δ·(M⁻¹e_i)(e_jᵀM⁻¹)/(1 + Δ(M⁻¹)_{ji})`, an outer product, `O(n²)`, paid *only on
accepts*. So the accounting inverts — `O(1)` to score, occasional `O(n²)` to refresh — and the
factorization is gone from the inner loop. That is what lets me push the budget from `40k` into the
millions: I plan `1.5M` flips per chain.

Carrying `M⁻¹` incrementally over more than a million updates does accumulate floating-point error, so
three guards. The matrix entries stay exactly `±1` (`M_{ij} += Δ` lands back on `∓1`), so `M` never
drifts — only the carried inverse does. Stale-inverse errors are somewhat self-correcting: they produce
ratios that disagree with the true determinant and so propose moves a correct evaluator rejects,
showing up as degraded search, not silent corruption. And decisively, the final answer is never trusted
to float — I take the best `±1` matrix recorded and recompute its determinant exactly with Bareiss.
Bounding the drift, each update injects relative error `~ε·κ ≈ 2.2×10^{−16}·10³`, so over `1.5×10^6`
updates the accumulation is at worst `~3×10^{−7}`, orders below the `~0.02` log scale where decisions
turn; a periodic refactor from scratch is a safety valve I note rather than need.

With scoring free, two things become cheap. First, *budget*: `1.5M` flips per chain gives room for the
long lateral sequences coordinated gains require. Second, *restarts from structured seeds*. The prime
`29` hands me a family of relabelings for free: reindex by `i ↦ k·i (mod 29)` for a unit `k`, and since
`29` is prime every `k` is a bijection. The Jacobsthal entries transform as `Q_{ki,kj} = χ(k)·χ(i−j)`,
so the relabeled seed is a permutation of `Q` (when `χ(k) = +1`) or of `−Q` (when `χ(k) = −1`) — either
way a legal `±1` design with the same baseline `m = 49` but a *different* basin of the flip landscape.

Spending the freed budget as several restarts rather than one very long chain is a real choice. The
previous plateau was a basin phenomenon — within one connected basin the marginal return per flip
saturates once cool — so pouring `10.5M` flips into a single chain buys only the diminishing tail of
one seed's basin. Splitting across seven chains from seven distinct basins samples the *spread* of
basin qualities, and since I report the best-of-seven I gain from variance across seeds, not just depth
within one. Each `1.5M` chain is already `37×` the budget that reached the `150`s, well past
single-chain saturation, so the marginal flip is better spent opening a new basin. I keep it simple —
independent restarts, best-of — rather than parallel tempering and its extra knobs.

The cooling schedule transfers unchanged: the `O(1)` trick changes only *how* I score a flip, not what
the flip does to the determinant, so the distribution of `Δlog|det|` is identical and the bracket
`0.06 → 2×10^{−4}` still fits, only stretched over `1.5×10^6` steps. For the seeds I want spread, not
near-duplicates: `−1` is a residue mod `29`, so `k` and `29 − k` give mirror basins and I take one of
each pair. I pick `k ∈ {1, 2, 3, 6, 10, 12, 15}` — seven small, well-separated units, `1` and `6`
residues (permutations of `Q`), the rest non-residues (permutations of `−Q`), mixing both flavors.
Seven chains of `1.5M` is `10.5M` flips, three orders past `40k`, reachable only because each flip is
now `O(1)`. I hold the single-entry move fixed — changing the budget *and* the move set at once would
make the climb unattributable, and this rung's claim is that the previous plateau was a *budget* wall.

I expect the same annealing dynamics run two-plus orders of magnitude longer and from multiple basins
to clear the low-`150`s plateau and push well into the band of the best machine-discovered results for
this order (`~197`). Two things would corroborate the mechanism over a lucky seed: throughput rising to
hundreds of thousands of flips per second, and a clear margin between the best and worst of the seven
seeds — if all seven converged to the same value, the restart machinery is dead weight and one long
chain would have been the better spend.

And here is where I stop honestly, because this buys the frontier of one kind of search and not the
record. The classical record at `320` comes from large-scale search over *Gram matrices*, not raw entry
flips: it optimizes the symmetric integer design `G = RRᵀ` directly, in the small rigid space of
admissible Gram matrices, and decomposes the winning `G` into a `±1` matrix only afterward. Single-entry
flip annealing walks the fixed landscape of `±1` matrices, and the record is not a short coordinated
walk from any structured seed. So I don't expect `320`; the trick does its job — converting the cost
wall into two-plus orders of magnitude more steps — and local entry-flip annealing tops out near the
machine frontier, with the Gram-space record standing above it.
