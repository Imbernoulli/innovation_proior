The rectangle taught me where the value is and where it isn't, and the seed-by-seed numbers make the
lesson quantitative rather than impressionistic. Reading down the rung-1 table, the sardine count `b`
is the whole story: it is `510` on seed 1, `1795` on seed 2, `535` on seed 3, `114` on seed 4, and
`1207` on seed 5. That number is exactly the tax the convex box pays for being unable to bend. Every
one of those sardine is a point I could in principle stop catching if the boundary could route around
it, and since the objective is `a − b + 1`, shedding a sardine while holding the mackerel is worth a
full point. So the recoverable value per seed — what I would gain if I could keep the current mackerel
and drop *all* the sardine — is `510, 1795, 535, 114, 1207`, averaging `832` across the five. That is
an enormous headroom sitting on top of the `2704` mean, and it is distributed wildly unevenly: seed 2
alone is carrying `1795` recoverable, seed 5 carries `1207`, while seed 4 is nearly clean at `114`.
The box on seed 2 already catches `4366` of the `5000` mackerel — it is a big, greedy box — but it
drowns in `1795` sardine to do it. That is not a case where I need to find more mackerel; it is a case
where I need to *carve sardine out of a region I have already enclosed*. Seed 4, by contrast, has
almost nothing to recover: its box catches `3201` mackerel against `114` sardine, so any clever
carving there is fighting over a hundred points and risks doing net harm. The table is telling me
plainly that the second rung's job is sardine excision on the overlapping seeds, and that it must be
careful not to break the seeds that are already clean.

There is a second column I should read before I design anything: the perimeters. Rung 1 spent
`203334, 333334, 333334, 180000, 313334` on the five seeds, all comfortably under the `4 × 10^5` cap,
so the *headroom* is `196666, 66666, 66666, 220000, 86666`. This is a warning that lands right on the
seed that matters most. Seed 2 — the `1795`-sardine opportunity — has only about `66666` of perimeter
left to spend on a wigglier boundary, which is the *least* headroom of any seed except its twin seed 3.
So on precisely the instance where I most want to snake the boundary around interior sardine, I have
the least length to do it with. Seeds 1 and 4 have huge headroom (`~200000`), but seed 4 has nothing
to carve and seed 1 has a moderate `510` to chase. The constraint and the opportunity are misaligned,
and any method I build has to be efficient with perimeter exactly where it is scarcest.

The problem hands me a far richer net — any rectilinear simple polygon up to a thousand vertices — and
the rectangle uses four of them and none of the perimeter budget's flexibility. I want a net whose
boundary can follow the *shape* of where the mackerel are, and I have to choose a representation for
"arbitrary rectilinear shape" that I can actually optimize. Let me walk the candidates honestly rather
than grab the first one. I could represent the net as a union of a few axis-aligned rectangles glued
into a single polygon — an L, a plus, a comb — which is more expressive than one box and has few
degrees of freedom. But choosing which rectangles and gluing them into one legal simple polygon is
fiddly, and a union of a handful of rectangles still cannot cut an arbitrary interior notch to release
a sardine pocket buried mid-shoal. I could represent it as a per-column vertical interval — a
histogram or monotone staircase — which sheds sardine cheaply along one axis but is blind to pockets
that need the boundary to reach in from two sides. I could go straight to optimizing polygon vertices
directly, which is the fully general thing, but keeping an arbitrary vertex polygon simple and legal
under edits is a minefield and I have no exact optimizer for it. The representation that subsumes the
L, the comb, and the staircase as special cases, while giving me a clean local edit, is a *grid of
cells*: choose a connected set of cells, and its outer boundary is automatically a rectilinear polygon.
So I commit to that.

Here is the representation concretely. Bucket the sea into a `G × G` grid and give each cell a weight
equal to `(#mackerel − #sardine)` inside it — the same signed stamping that made the box tractable,
now at cell granularity. A net is a subset of cells, `a − b` is the sum of the chosen cells' weights,
and the boundary of the subset — the unit edges where an inside cell meets an outside cell or the grid
border — is a rectilinear closed curve. As long as the subset is connected and has no holes, that curve
is a single simple polygon I can trace and emit. This turns "design an arbitrary rectilinear net" into
"select a good connected, hole-free cell region," a purely combinatorial problem. And crucially, the
value of a region decomposes additively over its cells, so I can evaluate the effect of adding or
removing a single cell locally, without re-summing the whole region — the linearity I noticed on the
box carries straight over.

Before I pick an algorithm I want to know the two *local* quantities a cell-region method lives on:
how a single cell changes the score, and how it changes the perimeter. The score change is trivial —
adding cell `c` changes `a − b` by exactly `W[c]`, one lookup. The perimeter change takes a moment of
counting. A cell has four unit edges. When I add it, each edge that faced an already-inside neighbor
*disappears* from the boundary (it becomes interior), and each edge that faced outside or the grid
border *joins* the boundary. So if the cell has `s` of its four neighbors already inside, adding it
changes the boundary-edge count by `4 − 2s`: `s = 0` gives `+4` (a lone cell, four new edges), `s = 1`
gives `+2` (a cell on a flat edge), `s = 2` gives `0` (a cell tucking into an L-corner — free
perimeter), `s = 3` gives `−2`, and `s = 4` gives `−4` (filling a one-cell dent, which shortens the
boundary). This little formula has a consequence I should flag now: the greedy will find it cheapest,
perimeter-wise, to fill concavities — cells with two or three inside neighbors cost `0` or negative
perimeter — which biases the region toward *convexification*, the opposite of the notch-cutting I
actually want for carving sardine out. The perimeter accounting quietly pulls against the objective,
and that tension is going to matter.

Now the algorithm. The most direct way to build a good region is to grow it: start from the single
densest cell and, at each step, look at the frontier — the outside cells adjacent to the current region
— and add the one that most increases the total weight, provided two things hold. First, the perimeter
must stay legal: I keep a running boundary-edge count, apply the `4 − 2s` delta, and reject any add
whose traced perimeter would exceed the budget (with a safe margin of `396000` under the `4 × 10^5`
cap to absorb grid-line rounding in the traced polygon). Second, I must not punch a hole: if adding a
cell would seal off an empty pocket, the region stops being a simple polygon. To check that, I
flood-fill the outside from the grid border and forbid any addition that leaves an unreachable empty
cell. That hole check is the expensive part, and its cost decides the whole shape of this rung. A
border flood-fill is `O(G^2)` — about `2500` cell-visits at `G = 50` — and I run it on each tentative
best add, of which there are up to the region size, another `~2500`. So a single grow is on the order
of `2500 × 2500 ≈ 6 × 10^6` operations, times three resolutions is `~2 × 10^7`, which a one-shot pass
in Python absorbs in a couple of seconds. But it forecloses something: I cannot afford this validity
check *millions* of times. If I wanted to run `10^7` candidate moves each paying an `O(G^2)` hole test,
that is `2.5 × 10^{10}` operations, hopelessly slow. The global flood-fill is exactly what makes this a
one-shot greedy rather than a search — and it plants the flag for what a search would need instead: a
validity test that is `O(1)`, not `O(G^2)`.

When I first run the pure version I hit a wall I should have seen coming from the `4 − 2s` bias. A
greedy that adds only positive-weight cells stops far too early. The frontier of a good region is
often a ring of slightly sardine-heavy cells, and beyond that collar sit rich mackerel cells the
greedy will never reach because it refuses to step through the negative ring. The region freezes
small, well below even the rectangle. So I let the greedy take the *highest-weight* admissible frontier
cell at each step even when that weight is negative — it can spend a little to bridge a gap — but I
track the best total weight the region has ever had and, at the end, restore that best snapshot rather
than wherever the walk happened to stop. The patience I allow before giving up is `G × G` non-improving
steps, which at `G = 50` is `2500` — effectively "keep trying to bridge until you have considered
adding most of the grid," generous on purpose so that a thick negative collar can still be crossed if
something rich lies past it. The bridging is speculative, but because I only ever emit the best-ever
snapshot, a bridge that never pays off is never charged to the final answer. This is the greedy's one
concession to non-monotonicity, and it is what lets the region reach past a sardine ring into a
mackerel pocket.

I want to watch the snapshot mechanism work on a hand example so I trust it. Picture a short chain of
cells: the seed is worth `+10`, immediately to its right sit two collar cells worth `−3` and `−3`, and
beyond them a rich cell worth `+20`. A pure positive greedy starts at weight `10`, looks at its
frontier, sees only the `−3` collar (and some zeros), finds no positive move, and freezes at `10`. The
bridging greedy instead takes the best available even when negative: it adds the first `−3` (weight
`7`), adds the second `−3` (weight `4`), then reaches the `+20` and jumps to weight `24`. Because `24`
now exceeds the best-ever `10`, the four-cell region becomes the stored snapshot, and I have netted
`+14` over the frozen start by spending `−6` of collar to reach a `+20` pocket. The snapshot logic also
protects me in the mirror case: if the `+20` had instead been a `+2`, the walk would peak at weight `6`
after the bridge — below the seed's `10` — and the restore would hand back the original single cell,
so the speculative bridge costs nothing. That is exactly the behavior I want: bridging is allowed, but
only kept when it pays.

Two invariants make the emitted polygon legal, and it is worth being precise about which one the
greedy gets for free and which one it must actively police, because that distinction is the whole
reason this stays a one-shot greedy. Connectivity is free: I only ever add a cell that is adjacent to
the current region, and I never remove a cell, so the region is a single connected blob at every step
by construction — I never have to test it. Hole-freeness is *not* free, and it is the one thing that
can turn a connected blob into an illegal net. The failure mode is concrete: if the region grows into
a `3 × 3` ring of inside cells with the center left outside, the boundary is no longer one cycle but
two — an outer square and an inner square around the trapped hole — and tracing it yields a
self-referential mess, not a simple polygon. The border flood-fill catches exactly this: after a
tentative add, I flood the *outside* from the grid border, and if any empty cell is unreachable, that
cell is enclosed and the add just punched a hole, so I reject it. This is precisely the hole condition
— an outside cell the exterior cannot reach is by definition sealed in — so the check is not a
heuristic but an exact test, and it is why I can guarantee a single simple polygon before I ever trace
one.

The trace itself is the payoff for keeping the region hole-free. I walk the unit edges where an inside
cell meets outside or the border, orienting each so the region stays on one side, which threads them
into a single directed cycle through the boundary vertices; then I merge collinear runs so a straight
staircase side collapses to one edge and the vertex count stays far below the `1000` cap. A connected,
hole-free region guarantees this walk closes into one loop — if I had allowed a hole, the walk would
capture only one of the two cycles and silently drop the other, which is exactly the corruption the
flood-fill exists to prevent. So the pipeline is airtight by the time it emits: connectivity by
construction, hole-freeness by flood-fill, perimeter by the running edge count, and vertex count by
collinear merging.

This helps, and I can reason about *where* it should help from the rung-1 table. The seeds carrying the
most recoverable sardine — seed 2 at `1795`, seed 5 at `1207` — are where a carved rectilinear boundary
has the most to gain by cutting sardine out of a mackerel shoal. But I have to hold two competing facts
together. On seed 2 the opportunity is largest *and* the perimeter headroom is smallest (`~66666`), so
the notches I need to reach interior sardine may cost more length than I have; a compact region of area
`A` already spends roughly `4√A` of perimeter just on its outline, and every notch adds more, so on
seed 2 the carve is likely perimeter-limited and will shed only part of the `1795`. And there is the
irreversibility I cannot design away: this is a single forward pass with no undo. A cell that looked
good when the region was small can become a liability once the shape settles, and the greedy cannot
remove it; it can saturate the perimeter on a ragged boundary and then have no budget left to fix a
mistake. So I expect real gains on the overlapping seeds but fragile ones — the greedy will capture
*some* of the recoverable sardine, not all of it, and it may capture it inefficiently.

Two cheap defenses make the rung robust rather than brilliant, and both are motivated directly by the
table. Because the perimeter-versus-carving tradeoff plays out differently at different cell sizes — a
finer grid cuts finer notches but its tighter cells mean the unit-edge budget (`396000 / cw`, so about
`119` edges at `G = 30`, `158` at `G = 40`, `198` at `G = 50`) buys fewer, smaller steps, while a
coarser grid covers area cheaply but quantizes the boundary bluntly — there is no single resolution
that is right for every layout. So I run the greedy at `G ∈ {30, 40, 50}` and keep whichever region
scores best by the internal weight; trying three is nearly free next to the flood-fills I am already
paying. And I include the rung-1 rectangle itself as one more candidate, computed the same way by a
prefix-sum sweep, so that whenever the grown region fails to beat the box — which is exactly what I
expect on the clean seeds 3 and 4, where there is almost no sardine to excise and a ragged region can
only lose — I fall back to the box. This guarantees the rung *dominates* rung 1: it is the rectangle
plus the option of a carved region, and it picks the better of the two by their internal estimates,
with the exact evaluator having the final word so the discretization never inflates the reported score.

I should check that ranking candidates by internal weight is actually safe, because I am now comparing
a carved region against a box using a number that is not the graded number. For a grid-aligned region
the internal weight is the exact sum of `(#mackerel − #sardine)` over the chosen cells, and since the
traced polygon boundary *is* the shared cell boundary, that sum equals the exact count of fish inside
the polygon — with the same lone caveat as the box, a fish sitting precisely on a boundary line, of
which I expect a fraction of one. So the internal estimate tracks the exact evaluator to within a fish
or so for both a region and a box, which means comparing their internal weights to decide which to emit
is faithful: I will not throw away a genuinely better region because its estimate was miscalibrated
against the box's. The one place this could bite is a near-tie, where a point or two of estimation
noise flips the choice — and that is exactly the seed-5-style situation where a carve and the box are
within a hair of each other. There, whichever I emit, the exact evaluator's verdict is what I report,
so a wrong pick costs at most that hair, not a real regression.

If the mechanism reading is right, the objective table should move in a specific, falsifiable pattern.
Seeds 3 and 4, clean and low-sardine, should fall back to the box and essentially tie rung 1 — a
non-regression by construction. Seeds 1 and 2, where sardine sits interleaved and there is genuine
carving to do, should improve, with seed 2 improving the most in absolute terms because its `1795`
recoverable dwarfs the others, *if* the greedy can find and afford the carve within its scarce
perimeter. Seed 5 is the one I am least sure of: it carries the second-largest recoverable (`1207`),
but its shoals are heavily overlapping and its box already spends a fair amount of perimeter, so a
one-shot greedy may simply fail to find a carve that beats the box by the internal estimate and fall
back — in which case seed 5 ties rung 1 rather than improving. So my honest prediction is: clear wins
on seeds 1 and 2, holds on 3 and 4, and seed 5 a coin-flip between a modest win and a fallback tie,
with the mean landing a couple hundred points above `2704`. The objective table will tell me which
way seed 5 broke and whether seed 2's carve was as perimeter-limited as I fear.

I can make the seed-2 prediction sharper, and in doing so name the specific cost the irreversibility
will exact. The rung-1 box on seed 2 held `4366` mackerel and `1795` sardine. To shed that sardine,
the grown region has to route its boundary between mackerel and sardine that are spatially interleaved
— and because the net is one connected blob with no remove move, the mackerel that sit *on the wrong
side* of a sardine pocket cannot be kept once I carve the pocket out; they get orphaned outside the
blob and the greedy has no way to reach back and re-include them. So I do not expect the carve to hold
all `4366` mackerel while dropping the sardine. I expect `a` to fall meaningfully below `4366` at the
same time `b` falls well below `1795`, with the *net* `a − b` rising — a real gain bought by
sacrificing mackerel it cannot recover. If instead seed 2's `a` stays near `4366` while `b`
collapses, then the interleaving is looser than I think and the greedy is doing better than its
irreversibility should allow; either way the `a` and `b` columns, not just the objective, will tell me
whether the mechanism I am positing is the one actually operating. That gap — mackerel sacrificed on a
carve that cannot be undone — is the precise quantity a reversible search would aim to recover, and it
is worth predicting now so I can measure it against what comes next.

One small consistency choice: I compute the rectangle-fallback candidate on the `50 × 50` grid rather
than one of the coarser ones, so the fallback box is at least as sharp as the finest region I try and
the rung never loses to rung 1 merely because the fallback was quantized more bluntly than the carve
it is competing against. And the whole procedure is deterministic given the instance — same grid, same
greedy, same box — so this rung, like the last, is a reproducible baseline I can trust as a measuring
stick rather than a number that wanders between runs.

So this rung is the right second step and no more. It replaces the four-cornered box with a region
whose boundary can bend around interior sardine, it respects perimeter and hole-freeness by
construction, and it strictly improves on the rectangle by keeping the rectangle as a fallback. But
its engine is a one-shot greedy with no reversibility, forced into that shape by an `O(G^2)` hole check
too expensive to run millions of times, and running on a coarse static grid. That is exactly its
ceiling, and I can state each limit as a concrete failure: it cannot trade a bad cell now for a better
configuration later because it has no remove move; it wastes perimeter on ragged boundaries and then
cannot repair them; and it is at the mercy of a single grid resolution per candidate. The way past all
three is the same move — stop treating region construction as a forward-only greedy and start treating
it as *search*, where cells can be both added and removed, downhill steps are accepted to escape the
greedy's traps, and the price of admission is exactly the thing the flood-fill denied me: a validity
check cheap enough to afford millions of times. That is what the next rung has to build.
