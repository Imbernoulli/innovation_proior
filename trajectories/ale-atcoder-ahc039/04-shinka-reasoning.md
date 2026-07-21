The annealing works, it plateaus, and the rung-3 table locates the plateau precisely. Reversibility
was vindicated where I predicted: seed 2 went from the greedy's `3188` to `3967`, moving from `a =
3618, b = 431` to `a = 4073, b = 107` — recovering `455` of the sacrificed mackerel while shedding
`324` more sardine to near the floor. But two seeds tell the other half. Seed 1 came in below the
greedy, `2330` against `2362`: warm-started from the box, it carved sardine to `b = 312` but stuck
at `a = 2641`, never extending to grab the outlying mackerel the greedy reached at `2770`, its
perimeter pinned at the `360000` safe cap. And seed 3 held the box entirely (`m = 4`, `b = 500`
untouched): the undirected search never stumbled onto the carve even starting right next to it. Both
are the same disease: proposals are undirected, so the rare flip that would fix a misclassified fish
is one in thousands, drowned in flips that touch already-correct boundary.

This is the wall the benchmark hit. ALE-Agent's SA reached `2880`; then a program-evolution system,
ShinkaEvolve, evolved that same SA to `3140`, with two changes and only two. Those AtCoder numbers
live on a relative scale I cannot compare to my raw `a − b + 1` means, but the *character* of the
jump is what I am after: not a new algorithm, the same SA made to search in the right direction. So
I reproduce those two levers in my grid representation and see whether they move my number on the
seeds rung 3 left stranded.

The first lever is caching the validation: never recompute from the whole structure what you can
maintain incrementally on the patch a move touches. I already cache per-cell counts (O(1) scoring)
and the running edge count (O(1) perimeter); what I have not cached is which cells are currently on
the boundary — and in rung 3 that cost me, since a remove sampled a random region cell and threw it
away if interior. The operator I am about to add needs to sample boundary cells and inspect their
outside neighbors constantly, which by rescanning the region would be `O(region)` per step. So I add
a per-cell boundary-flag cache. Flipping cell `(i, j)` can change a cell's boundary status only if
that cell is `(i, j)` or one of its four neighbors, so refreshing the full `3 × 3` window after each
accepted flip is a correct-by-construction superset, and every candidate can now ask "is this a
boundary cell?" and "what are its outside neighbors?" in O(1).

The second lever redirects the search: the targeted edge move. Identify a misclassified fish and
greedily move the nearest edge to correct its state. In my grid, with probability `P_TARGET` I
sample a boundary cell and look at its outside neighbors: if one is mackerel-rich I propose adding it
(moving the edge outward to capture a misclassified mackerel), and if the boundary cell is itself
sardine-heavy I propose removing it (moving the edge inward to release a misclassified sardine).
Adding or removing one boundary cell shifts the local edge by exactly one cell-width, so a flip
aimed at a misclassified fish *is* the nearest edge stepping over to reclassify it. The proposal
still passes through the same Metropolis acceptance and cooling, so the search can decline a fix that
costs too much elsewhere.

Seed 1 is where this should bite. It was left at `a = 2641`, perimeter pinned at `360000`, `b` still
`312`; undirected SA could not help because the extend-moves were refused by the perimeter gate. The
targeted operator makes the trade affordable as a pair: a targeted remove of a sardine-heavy boundary
cell hands back perimeter while raising `a − b`, and a targeted add of a mackerel-rich outside cell
spends that freed perimeter reaching outward — roughly perimeter-neutral, landing back at the cap
with a sardine cell traded out and a mackerel cell traded in. Neither half is affordable alone
against a saturated boundary; as a directed sequence they net to zero perimeter and a positive score
swing. On seed 3, symmetrically, the directed remove aims straight at the `500`-sardine boundary the
undirected search never carved.

I keep the uniform flip in the mix rather than replacing it, because rung 3's one clear success —
seed 2 — came from a large high-temperature reshape that a pure targeted search would over-commit
away from. So with `P_TARGET = 0.55` I direct a little over half the proposals at real errors and
keep the rest for undirected exploration; `0.55` is judgment, not derived. Everything else carries
over: the schedule `T0 = 8 → T1 = 0.05`, the simple-point and pinch checks, the exact running
perimeter, the warm start. The mix changes the density of useful proposals, which is the whole
lever — in rung 3 essentially no proposals were error-directed, so near the plateau the chance a
flip landed on a fixable spot was a fraction of a percent; now `55%` sample a boundary cell and
immediately inspect it. The gain per fix is small, a handful of fish, but the number of fixes
attempted per second was the starving quantity.

The operator is safe on already-good nets: on a boundary with no misclassified fish the targeted add
finds its best outside neighbor has `Av − Bv ≤ 0`, a downhill move the cooled acceptance rejects, and
the targeted remove only fires on a sardine-heavy cell a clean boundary lacks. So on a near-optimal
net like seed 2 at `a = 4073, b = 107` it mostly proposes rejected moves and cannot degrade. There
is a sharper consequence: targeted moves are by design usually uphill, and a positive-`Δ` move is
accepted at any temperature — so unlike rung 3's payoff, which leaned on lucky high-temperature
reshapes that differ run to run, the directed gains are improving moves the search takes whenever it
samples the right cell, which predicts the lift should be reproducible across runs. If instead the
gain swings wildly, it is coming from chance reshapes and my mechanism story is wrong.

More elaborate versions I set aside: a global priority queue of the most-misclassified fish is
strictly more directed but needs a heap updated `O(log n)` per flip and, since my finest unit is a
cell, collapses to ranking cells anyway; sampling a boundary cell and its four neighbors is O(1) and
captures the same aim. Periodic whole-boundary recomputation is cheaper per line but staler between
refreshes, and the operator reads the cache far more than the region changes, so a stale set would
mis-aim proposals. Every seed still ends with the same final true-perimeter guard falling back to
the best rectangle, so a directed proposal inherits every validity guarantee rung 3 earned.

Seed by seed I expect: seed 1 to recover its uncaptured mackerel once the targeted remove frees the
length the targeted add needs; seed 3 to carve past the box for the first time; seed 2, already near
its frontier, only a small directed gain; seed 4, already well carved at `b = 68`, to edge up; and
seed 5, so heavily overlapping that `b` is still `1222` and its perimeter near the limit, the one I
am least hopeful about — there may be no profitable carve to target, so it could hold or fall back to
the box. The overall gain should be modest, real, and reproducible — the same character as `2880 →
3140`: not a new algorithm, but the SA made to search in the right direction with a cache that makes
the directed proposals affordable.
