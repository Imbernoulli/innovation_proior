The greedy region taught me two things at once: that a rectilinear net built from grid cells is the
right object, and that growing it by a one-shot forward pass is the wrong engine. The greedy cannot
take back a cell it regrets, it spends its perimeter budget on whatever ragged boundary it stumbles
into, and it lives or dies by a single grid resolution. Every one of those failures is a failure of
*irreversibility*. So I want to keep the representation — a connected, hole-free set of grid cells
whose outer boundary traces a simple rectilinear polygon — but drive it with something that can both
add *and* remove cells and is willing to step downhill to climb out of the basins the greedy fell
into. That points at local search with Metropolis acceptance: propose a flip, take improvements,
take small worsenings with a temperature-dependent probability, cool over the run. The question is
whether I can make each step cheap enough to run millions of them, and legal enough that what I emit
is always a valid net.

The move set is the natural one for a cell region. At each step I either *add* an outside cell
adjacent to the region or *remove* a boundary cell of the region — a single-cell flip of the
boundary, which adds or shaves one staircase step. The first thing I want to know is what a flip
costs to score. If I precompute, once, the mackerel count `a[c]` and sardine count `b[c]` in every
cell, then adding cell `c` changes `a − b` by exactly `a[c] − b[c]`, and removing it by the
negative. Whether the running total I carry actually stays equal to a full recount is the kind of
claim I have burned myself on before, so let me trace it on a tiny three-cell example. Say
`a = (3, 0, 1)`, `b = (1, 4, 1)`, so the per-cell `a − b` is `(2, −4, 0)`. Start empty with running
score `0`; add cell 0 (`+2 → 2`), add cell 1 (`−4 → −2`), add cell 2 (`+0 → −2`), then remove cell 1
(`+4 → 2`). Recomputing `a − b` over the in-set directly after each step gives `2, −2, −2, 2`. They
agree at every step. So the running update is exact, and the change in `a − b` for any candidate flip
is one array lookup and one addition. That is the whole game for speed: I never recount fish during
the search, I only read the cell's stored counts. The catch is that this assumes each flip leaves the
running boundary-edge and topology bookkeeping consistent too, which I have to check separately.

Three things have to stay legal under every flip. Perimeter: I keep a running count of boundary
unit-edges; adding or removing a cell changes it by a small local amount computed from the cell's
four neighbours, so I can reject any flip that would push the traced perimeter over `4 × 10^5`
without ever retracing the polygon. Connectivity and hole-freeness are the subtle ones: a flip must
not disconnect the region, and must not open or close a hole, because then the boundary stops being a
single simple polygon. Recomputing global connectivity after every flip would destroy the speed, so I
reach for the classical digital-topology test — a cell flip is "simple" (preserves the region's
topology) iff the foreground transitions exactly once as you walk the eight-cell ring around it.
That is a constant-time check on a 3×3 window. Before I trust it I want to walk a few concrete
windows and count those transitions by hand. Writing the ring clockwise from the top as
`(N, NE, E, SE, S, SW, W, NW)` and counting `0→1` transitions: a cell with one in-region neighbour,
say only `W` in — ring `(0,0,0,0,0,0,1,0)` — gives transition count 1, accepted, a clean coastline
extension. A cell whose only in-region neighbours are two *opposite* sides, `N` and `S` — ring
`(1,0,0,0,1,0,0,0)` — gives count 2, rejected; good, because adding it would bridge two separate
arms. Two diagonal in-neighbours, `NE` and `SW` — ring `(0,1,0,0,0,1,0,0)` — also count 2, rejected.
So far the test does exactly what I want.

Then I tried a case that surprised me. A one-cell concave notch being filled — center with in-region
neighbours on `N`, `S`, and `W`, ring `(1,0,0,0,1,0,1,0)` — gives transition count 3, so the
simple-point test *rejects* it. But is filling a notch actually illegal? I built the before/after
regions explicitly: a small block with a single-cell dent open from the top, traced its outer
boundary, and counted cycles — one cycle before the fill, one cycle after. Topologically the fill is
perfectly fine; the boundary stays a single simple polygon. So the simple-point test is
*conservative*: it forbids some legal moves, not just illegal ones. That is a real limitation, not a
bug — the test guarantees that every accepted flip is safe, but it does not let through every safe
flip, and a notch the search would like to deepen one cell at a time may be blocked because the
intermediate has three orthogonal neighbours. I decide to accept that. Correctness of the emitted net
matters more than reaching every reachable shape, and the search has many other routes to the same
region (approach the notch from the side, or via removals). I note it as the price of an O(1) check
and move on.

There is one more trap, and here the crossing-number test is not just conservative but actually
*insufficient*. Two cells touching only at a corner make the boundary a figure-eight, not a simple
polygon. I worried whether the transition count already catches every such pinch, so I enumerated all
256 eight-neighbourhoods of a just-added cell and looked for ones where the transition count is 1 (so
the test accepts) yet some 2×2 block through the center is a checkerboard (a diagonal pinch). There
are exactly four such configurations. The simplest: the center is in-region and its only in-region
neighbour is `NW` — ring `(0,0,0,0,0,0,0,1)`, transition count 1, accepted by the simple-point test.
But the center and its `NW` diagonal are both in while `N` and `W` are both out, so the two cells
meet only at a corner. I traced that two-cell region's boundary directly: two cycles, a figure-eight.
So the simple-point test alone would emit an illegal net here. That forces a second guard: forbid any
flip that completes a diagonal-only 2×2 block. Concretely, for each of the four 2×2 blocks with the
toggled cell as a corner, reject if the cell and its diagonal are one class and the two orthogonals
are the other. With both checks — transition count exactly 1 *and* no diagonal pinch — every accepted
flip keeps the traced boundary a single simple rectilinear cycle, which is what the evaluator demands.

Now the search itself. I anneal: propose a random legal flip; if it improves `a − b`, take it; if it
worsens `a − b` by `Δ`, take it anyway with probability `exp(Δ/T)` for a temperature `T` cooled
geometrically over the run. Early, when `T` is high, the region wanders — accepting cells that
locally cost catch — so it can reshape itself out of the basins the greedy got stuck in; late, when
`T` is low, only improving flips survive and the region settles. I keep the best region ever seen
and emit that at the end. The bet is that reversibility plus downhill tolerance finds a configuration
the forward greedy never could.

But where do I *start*? My first instinct is to start from a single seed cell and let SA grow the
whole region, the way the greedy did. When I think it through, that fights the binding constraint.
The hard limit here is perimeter, and a single big rectangle is the most perimeter-efficient shape
there is — it encloses the most area per unit of boundary. If I grow from a dot, SA spends almost its
entire budget of moves just inflating a blob up to a reasonable size, paying boundary cost the whole
way, and barely gets to the interesting part. So a warm start: compute the best
perimeter-constrained rectangle exactly with a 2D prefix-sum sweep, initialize the region to *that*,
and let SA spend its budget carving notches and reaching for outlying mackerel.

That warm start raises a question I should answer numerically, because it controls the grid
resolution too. At `G = 50` the cell side is `100000 / 50 = 2000`, so a unit boundary edge is length
`2000` and the budget `400000` buys at most `200` unit-edges. A `k × k` cell square has `4k`
boundary edges, so the largest square that fits has `4k · 2000 ≤ 400000`, i.e. `k = 50` — the whole
grid. In fact the same holds at `G = 25, 100, 200`: a full-field square always lands at perimeter
exactly `4 × 100000 = 400000`, dead on the budget, at every resolution. So the constraint is not
"can a big box fit" — a box covering the entire field always fits exactly — it is that such a box
also scoops up every sardine. The whole value of SA is in *irregular* boundary, and irregular
boundary is precisely what eats perimeter. Each one-cell staircase notch replaces one flat unit-edge
with three (in, across, out), a net `+2` unit-edges, costing `2 × 2000 = 4000` of length. A full-grid
square has zero slack, so it cannot afford a single notch. To carve notches the region must first
*shrink*: a square of side `49` cells uses perimeter `392000`, leaving slack `8000` — exactly two
notches; side `45` uses `360000`, leaving `40000` — ten notches; side `40` leaves room for twenty.
That tells me two things. First, the warm-start box should not be the maximal field-filling square
but a high-value sub-rectangle that leaves perimeter headroom, which the prefix-sum sweep finds
because it maximizes enclosed weight, not area. Second, the resolution choice is a real tension:
finer cells make each notch sharper (a notch is one cell deep) but each notch still costs two edges,
and finer cells mean a fixed `200`-edge equivalent wraps proportionally less coordinate length per
edge only if the region is ragged — for smooth boundary all resolutions tie. When I sweep
resolutions the middle grid `G = 50` balances notch sharpness against the bookkeeping cost of a finer
array, so I fix it for this rung.

What I expect: the warm-started, reversible SA should clearly beat the one-shot greedy on the
overlapping seeds, because it can do what the greedy cannot — start from a strong box and then
*remove* sardine-heavy cells along the boundary and *add* mackerel cells just outside it, trading the
net's shape against the catch in both directions. On the easy seeds where the box is already
near-optimal, SA should hold the box — it has nothing better to find, and the best-ever region is
kept — so the rung never regresses below the warm start. I should also be honest about the ceiling I
am about to hit, and the perimeter arithmetic above pins it down. The single-cell flip is a *local*
move, but the useful trades are *global*: every notch I want to cut costs two edges, and once the
boundary is near its length limit those two edges have to be paid for by shaving length somewhere
else. A blind random flip almost never proposes that coordinated shrink-here-cut-there pair — most
random flips touch boundary that is already correct, and the few that would fix a genuinely
misclassified fish are drowned out and individually blocked by the budget. The search will plateau
not because the idea is wrong but because the proposals are undirected and the perimeter coupling is
global. That undirectedness, and the cost of revalidating each candidate, are exactly the two levers
a *next* step would want to pull.
