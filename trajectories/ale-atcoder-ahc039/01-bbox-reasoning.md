I need to put down one legal net and catch more mackerel than sardine, and before I reach for
anything clever I want the simplest object that is guaranteed valid and gives me a number to beat.
Everything downstream — every wigglier boundary, every search — will be judged against whatever
this first thing scores, so I want the floor to be both trivially legal and genuinely good of its
kind, not a strawman I can beat by accident. The constraints are forgiving about one shape in
particular: a single axis-aligned rectangle is always a legal net. It has four vertices, which sits
inside `4 ≤ m ≤ 1000`; its four edges are axis-parallel by construction; it is trivially simple, a
convex closed curve that cannot cross itself; and its perimeter is easy to keep under `4 × 10^5`.
So a rectangle is the floor I can stand on without any risk of producing an invalid output, and the
only real question is *which* rectangle.

Before I even choose one, I want to know how hard the perimeter constraint actually presses on a
rectangle, because that will tell me whether it is a live part of the optimization or a formality.
A box has width `w = x2 − x1` and height `h = y2 − y1`, both living inside the sea `[0, 10^5]`, so
each of `w` and `h` is at most `10^5`. The perimeter is `2(w + h) ≤ 2(10^5 + 10^5) = 4 × 10^5`,
with equality only when `w = h = 10^5` — the full square. Every *proper* sub-box of the sea is
therefore strictly under the budget, and even the full square lands exactly on it rather than over.
The upshot is blunt: for a single axis-aligned rectangle bounded by the grid, the perimeter ceiling
never binds. No box I could draw inside the sea is illegal by length. I will still fold a perimeter
guard into the search loop, but I should be honest with myself that on this rung it is a near-vacuous
check that fires only at the degenerate full-square corner I would never want anyway. The reason I
keep it is that it costs one comparison, and the moment the net stops being a single convex box and
is allowed to bend into a long ragged boundary, that same length budget stops being a formality and
becomes *the* binding constraint. Writing the guard now, when it does nothing, is me pre-paying for
the regime where it does everything.

The objective inside a rectangle is clean, and it is worth writing out exactly because the same
accounting will carry through every later net. If I fix the four sides at `x ∈ [x1, x2]`,
`y ∈ [y1, y2]`, then `a − b` is just the number of mackerel in that box minus the number of sardine
in it. That is a sum of per-point contributions where each mackerel counts `+1` and each sardine
counts `−1`, so if I imagine a weight `+1` stamped at every mackerel and `−1` at every sardine, the
value of a rectangle is simply the total weight it covers. Maximizing `a − b` over all axis-aligned
boxes is therefore the classic *maximum-weight axis-aligned rectangle* problem on a point set with
signed weights. I cannot literally try every real-valued box, but I do not need to, and I want to be
sure of that before I build anything on it. Take any optimal box and look at its right edge sitting
at some `x`-coordinate that is not on top of a data point. Slide that edge left until it just reaches
the next mackerel or sardine `x`-coordinate. Nothing in the strip I swept over was a data point, so
the covered weight did not change; the box is still optimal, and now its right edge is snapped to a
coordinate. The same argument works for all four edges independently, so there is always an optimum
whose every side lies on a line through the data. The search over boxes is therefore finite — bounded
by pairs of data coordinates per axis — and prefix sums over a grid of such lines make evaluating any
box a constant-time operation. This is also the first appearance of a property I will lean on hard
later: `a − b` is a *linear functional* over the covered points, so it decomposes additively over any
partition of the plane into cells. That linearity is the entire reason prefix sums work here, and it
is the same reason a later net that changes by one cell at a time will be able to update its score in
O(1) rather than recounting — the accounting I set up now is reusable.

It is worth being explicit about why I optimize `a − b` and not `a` alone, because the difference is
exactly the tradeoff the whole problem is about, and the signed weight encodes it for free. Suppose I
have a candidate box and I consider pushing its right edge outward by one strip. That strip contains
some mackerel `Δa` and some sardine `Δb`; the box's `a` rises by `Δa` no matter what, so an objective
of "maximize `a`" would extend greedily and end up as the bounding box of all mackerel, dragging in
every sardine among them. But the signed value changes by `Δa − Δb`, so the sweep only keeps the
extension when the strip is mackerel-favorable — a strip with `10` mackerel and `12` sardine moves
`a − b` by `−2` and is rejected even though it would add ten mackerel. The `+1/−1` stamping means the
optimizer automatically stops each edge exactly where the marginal fish flip from net-helpful to
net-harmful, which is the correct local condition for a maximum-weight box. This is also why the
problem is genuinely hard for shapes richer than a box: for a rectangle the optimal stopping point of
each edge is decoupled and the whole thing is a clean polynomial sweep, whereas for an arbitrary
rectilinear polygon the "where does the boundary want to be" question couples across the entire
boundary at once and admits no such closed-form sweep — which is the deeper reason I solve the box
exactly here and leave the polygon to search later.

Let me make sure I am reaching for the right object and not talking myself into the first idea. There
are cheaper things I could do. I could take the bounding box of *all* the mackerel — but that box
also swallows every sardine interleaved among them, and on any overlapping layout its `b` would be
enormous, so it is only good when the species are already cleanly separated, which is exactly the
easy case I do not need help with. I could pick a box by a density threshold — grow a box around the
single densest mackerel cell until the marginal density drops — but "until the density drops" is an
arbitrary stopping rule with no claim to optimality, and it can stop short of a rich pocket sitting
one thin sardine-collar away. Or I could skip straight to a many-vertex rectilinear polygon that
bends around the sardine. That is where the real value is, but it is the wrong *first* move: I have
no baseline to tell me whether a clever polygon is actually earning its complexity, I have real risk
of emitting an invalid net while I debug the geometry, and there is no clean exact optimizer for the
polygon family the way there is for the rectangle family. The max-weight box is the one member of
the design space I can solve *exactly* (up to grid resolution) and *safely*, and that is precisely
what a floor should be. So I commit to it and refuse the temptation to be clever one rung early.

It also helps to know what the *trivial* floors below me are, so I can judge whether the box is
actually doing work. Two nets require no thought at all. The empty-ish minimal net catches nothing,
so `a = b = 0` and the objective is `max(0, 0 − 0 + 1) = 1`. The opposite extreme — the full square,
which is legal because its perimeter is exactly `4 × 10^5` — catches every fish, all `5000` mackerel
and all `5000` sardine, so `a − b = 0` and again the objective is `max(0, 0 + 1) = 1`. Both degenerate
extremes score a flat `1`; the value lives entirely in the *asymmetry* a well-placed boundary can
create between the two species. A max-weight box that scores, say, in the thousands is therefore not
squeaking past a hard baseline — it is clearing the trivial floor by three orders of magnitude, which
tells me the signed-weight framing is capturing real structure and not just luck. Any later net that
fails to beat this box is worse than "catch the densest pocket," and any net that fails to beat `1`
is worse than doing nothing, which is a useful pair of guardrails to carry forward.

Here is the move concretely. I lay down a grid of candidate vertical and horizontal cut lines — a
coarse uniform set across `[0, 10^5]` is enough to find a strong box, and refining it costs only
more grid lines. I bucket every fish into the grid cell it falls in, accumulating per-cell
`(#mackerel − #sardine)`. Then I build a 2D prefix-sum table over the cells so that the signed
weight of *any* rectangle aligned to the grid lines is a constant-time difference of four table
entries: `sum(i1..i2, j1..j2) = PS[i2][j2] − PS[i1][j2] − PS[i2][j1] + PS[i1][j1]`. Now I can sweep
over all `(x1 < x2, y1 < y2)` choices of grid lines, read off each box's `a − b` in O(1), and keep
the best while skipping any box whose perimeter would exceed the budget.

The bucketing step deserves a moment because it is where an off-by-one would quietly corrupt the
whole search. Each fish is placed by binary-searching its coordinate into the cut lines, then clamped
into `[0, G−2]` so a fish exactly at `x = 10^5` (the far edge) lands in the last real cell rather than
falling off the end of the table. This clamp is why the buckets remain a clean partition — every fish
lands in exactly one cell, no fish is dropped, and the per-cell `(#mackerel − #sardine)` sums to the
global `a − b` of the whole sea. The costs here are all small and one-time: bucketing the `2N = 10^4`
fish is `O(N)`, building the prefix table over the `G × G ≈ 3600` cells is `O(G^2)`, and only the box
sweep is superlinear. So the entire pipeline is dominated by the `~3.3 × 10^6` box evaluations, and
everything upstream of the sweep is free by comparison — which is another reason not to over-optimize
the sweep with Kadane when the naive version already fits the budget.

I want to hand-check the prefix-sum machinery on a tiny case, both to confirm the four-corner formula
and — more usefully — to *see* the rectangle's ceiling in miniature, because the toy makes concrete
what the coarse real instance only hints at. Take a `3 × 3` grid of cell weights: a ring of eight
cells each worth `+5` surrounding a single centre cell worth `−50` (a fat mackerel doughnut with a
sardine pit in the middle). The whole grid sums to `8 × 5 − 50 = −10`. Building the prefix table row
by row, `PS[1][3] = 15` (the top row of three `+5` cells), `PS[2][2] = −35`, `PS[3][3] = −10`, and
the four-corner formula on the top row gives `PS[1][3] − PS[0][3] − PS[1][0] + PS[0][0] = 15`,
matching the direct sum — so the machinery is right. Now the interesting part: what is the best box?
A box `[r1, r2] × [c1, c2]` contains the centre cell iff it spans both the middle row and the middle
column, and any box that contains the centre eats the `−50`. To dodge it, the box must live entirely
in one strip — a full row or a full column of the ring — and the best such strip is worth `3 × 5 =
15`. So the maximum-weight rectangle here scores `15`. But the doughnut itself — the eight-cell ring,
a shape a rectilinear polygon could trace by looping around the pit — is worth `40`. The rectangle
captures `15` where the true structure holds `40`; roughly two-thirds of the available value is
unreachable, not because I searched badly but because a convex single body cannot enclose the ring
without also enclosing the hole. This is the whole story of the rung in one hand-computed example.

To be sure the `15` is genuinely the best box and not just the first good one I found, I map the rest
of the box family in the toy. Any box that contains the centre pays the `−50`: the whole grid is
`−10`, the middle column is `5 − 50 + 5 = −40`, and a `2 × 2` corner block that reaches the centre is
`5 + 5 + 5 − 50 = −35` — all far worse than a clean strip. Any box that avoids the centre is confined
to a single row or a single column, whose best is the full strip of three at `15`. So `15` is the true
maximum over the entire `3 × 3` box family, confirmed by exhaustion, and the doughnut's `40` is
genuinely out of reach for *any* rectangle, not just a greedy one. The `40 : 15` ratio is an artifact
of the extreme `−50` I chose, but the direction generalizes: the more a sardine pocket sits *interior*
to a mackerel shoal, the larger the fraction of value a rectangle must forfeit, and the real instances
are built to place exactly such interior pockets.

There is a resolution knob I should check at its limits before fixing it. As `K → ∞` the cut lines
become dense, the snapped box converges to the true real-valued maximum-weight box, and the only loss
is compute. As `K` shrinks the box gets blocky — its edges fall on a coarse lattice, so it cannot hug
a shoal whose natural boundary runs between grid lines, and it may be forced either to include a strip
of sardine or to exclude a strip of mackerel it would rather split. Neither limit touches the *shape*
ceiling the doughnut exposed; a finer grid finds a better rectangle, never a non-rectangle. So `K`
only trades compute for how tightly the single box hugs the data, and I set it where the sweep is
still cheap: `K = 60`.

That worked example also sharpens *how* I should search. The naive sweep is four nested loops over
`(i1, i2, j1, j2)`, which is `O(G^4)` for a `G × G` grid — I could instead collapse a dimension with
Kadane's maximum-subarray trick: fix the top and bottom rows, sum the enclosed columns into a 1D
array, and run a 1D max-subarray in `O(G)`, for `O(G^3)` overall. For `K = 60` cut lines per axis
that is the difference between roughly `60^3 ≈ 2.2 × 10^5` and the naive `≈ 3.3 × 10^6` (about `61`
lines give `61 choose 2 ≈ 1830` intervals per axis, so `1830^2 ≈ 3.3 × 10^6` boxes), a ~15× constant
factor. Tempting — but I talk myself out of it. Kadane's collapse hides the box's *width* inside the
subarray, so folding a per-box perimeter test into it is awkward, and at `K = 60` the naive sweep is
only a few million O(1) reads, which finishes in about a second in Python; the simplicity and the
clean per-candidate perimeter gate are worth more than a constant factor I do not need. The
resolution itself is a budget choice: cost grows like `K^4`, so `K = 60` (~3.3M boxes) is comfortable
while `K = 200` would be `≈ 4 × 10^8` boxes and far too slow for a Python sweep. I fix `K = 60` as
fine enough to place a strong box and cheap enough to sweep exhaustively.

There is a subtlety in *how* I score, and it is worth getting right because it will haunt every later
rung. The grid I use for the prefix-sum search is a discretization: a box snapped to coarse grid
lines is only an approximation to the true best box, and the number I optimize internally is a grid
estimate. But the *evaluator* is exact — it does a real integer point-in-polygon test with
on-boundary-counts-as-inside — so the internal number and the graded number can differ. I want to
know by how much. For a box whose sides are exactly grid lines, the cell buckets already count fish
exactly, so the only possible mismatch is a fish sitting precisely on a chosen edge, where my
half-open cell convention and the evaluator's boundary-is-inside rule can disagree by one. The grid
lines sit at the `61` integers `⌊i · 10^5 / 60⌋`; a random integer fish coordinate lands on any given
line with probability `~61 / 10^5`, so across `10^4` fish only a handful touch *any* line, and only
those on the two chosen edges of the winning box matter — an expected fraction of a single fish. So
the estimate tracks the exact count to within essentially zero, the gap is harmless for a rectangle,
and to be safe I will *report only the evaluator's exact score*, never the internal estimate. The
discretization guides the choice; it never inflates the result.

What do I expect this to buy, and where will it fall short? A single rectangle will do surprisingly
well when the mackerel cluster densest in one compact region with the sardine mostly elsewhere — then
a box around that region scores high, catching most of one species and little of the other. It will
do poorly in exactly the situation the problem is built to create: mackerel and sardine drawn from
*overlapping* shoals, so that any rectangle large enough to capture a mackerel pocket also swallows
the sardine threaded through it, and any rectangle small enough to dodge the sardine also clips off
mackerel at the edges. A rectangle is convex and single-bodied; it cannot reach around a sardine
cluster, it cannot bridge two separated mackerel pockets without paying for the sardine-rich gap
between them, and it has no way to cut a notch. The doughnut toy already showed the shape of the loss:
wherever a sardine pocket sits *inside* a mackerel shoal, the box must either eat the pocket or
abandon the shoal around it. The whole point of the problem — that the net is an arbitrary rectilinear
polygon — is precisely the freedom a rectangle throws away.

If that mechanism reading is right, it makes a prediction I can check against the one metric I have,
the mean objective. On the seeds where the two species are spatially separated I expect the box to
capture nearly all of one shoal at a low sardine cost, so the objective there should be high and
close to whatever a smarter net could do. On the overlapping seeds I expect `b`, the sardine caught,
to be conspicuously large relative to the separated seeds — the box bleeding sardine it cannot dodge
— and the objective there to lag well below the separated seeds. I do not yet know the absolute
numbers, but the *spread* across seeds should track how interleaved each layout is, and the mean
should sit somewhere in the low-to-mid range that any single convex net is capped at. The objective
table will tell me whether the separation story holds; if instead the box scores uniformly across
seeds, my picture of overlapping-versus-separated shoals is wrong and I would need to rethink where
the value hides.

A few implementation choices follow from wanting this floor to be *trustworthy* rather than merely
good. The sweep is fully deterministic — no randomness anywhere — so the same instance always yields
the same box, which matters because a baseline that jitters run to run is useless for measuring what
later rungs add. I fold the perimeter check *inside* the box loop, testing `2(w + h)` before I read
the box's weight, rather than sweeping first and filtering afterward; even though I established the
check is near-vacuous for a grid-bounded box, keeping it in the loop means the code is already shaped
for the regime where perimeter *does* bind, and there is no separately-maintained list of "surviving"
boxes to drift out of sync with the scoring. And the output is always exactly `m = 4` vertices, the
four corners in order, which is the minimum legal vertex count — so I am spending none of the `1000`-
vertex allowance, leaving the entire budget of boundary complexity untouched for a net that can
actually use it. That last observation is really the diagnosis of this rung stated as a resource
count: I am handed up to a thousand vertices and a long perimeter, and I am using four vertices and a
convex body. The gap between what the constraints permit and what a rectangle exploits is the whole
opportunity the next rung inherits.

So I am proposing the bounding box as the *right* first rung and nothing more: it is the simplest
guaranteed-legal net, it is found exactly (up to grid resolution) by a fast prefix-sum sweep whose
perimeter guard is free even though it barely bites, and it gives a concrete, honest baseline
objective that every later, more flexible net must beat. The limitation is baked into its shape, not
its tuning: with only four corners and one convex body, it cannot separate interleaved species, and
no amount of finer grid or longer sweep changes that. The moment I want to exclude a sardine pocket
sitting inside a mackerel shoal — to trace the doughnut instead of settling for the strip — I need a
boundary that can bend, a rectilinear region with many vertices, and that is what the next rung has
to build, by giving up the single box for a *grid of cells* I can selectively include.
