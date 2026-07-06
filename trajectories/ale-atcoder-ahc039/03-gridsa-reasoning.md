The greedy region taught me two things at once, and the second-rung table lets me put numbers on
both. First, a rectilinear net built from grid cells is the right object: on the overlapping seeds it
beat the rectangle decisively — seed 2 went from `2572` to `3188`, seed 1 from `2134` to `2362` — by
carving sardine out of a mackerel shoal that a box could only swallow. Second, growing that region by
a one-shot forward pass is the wrong engine, and the clearest evidence is *how* seed 2 won. The
rung-1 box on seed 2 held `4366` mackerel and `1795` sardine; the grown region came in at `3618`
mackerel and `431` sardine. So the greedy shed `1364` sardine but paid for it by giving up `748`
mackerel — it could not hold the whole catch while cutting the sardine out, because it is a single
connected blob with no remove move, and the mackerel stranded on the far side of a carved pocket got
orphaned outside the region with no way to reach back and re-include them. The net gain was real,
`+616`, but I can see the leak: of the `1795` recoverable sardine, only about a third turned into
score, and `748` mackerel were sacrificed to get even that. That sacrificed catch is the price of
irreversibility, and it is measured, not hypothetical.

The rest of the table sharpens the diagnosis. Seeds 3 and 4 fell back to the box (`m = 4`, identical
to rung 1) because they are already clean — seed 4 at `b = 114` has almost nothing to carve — and the
greedy correctly declined to make them worse. Seed 5 actually dipped `8` points below rung 1: it
carries `1207` recoverable sardine, the second most, yet the one-shot greedy failed to find a carve
that beat the box by its internal estimate and fell back, essentially tying. And the two seeds that
*did* carve, seeds 1 and 2, both saturated their perimeter — `393334` and `393332`, right against the
safe budget — with ragged boundaries of `50` and `36` vertices. So the greedy, exactly as I feared,
spends its length budget on whatever boundary it stumbles into and then has nothing left to repair it.
Every one of these failures — the sacrificed mackerel on seed 2, the missed carve on seed 5, the
saturated ragged boundaries on seeds 1 and 2 — is a failure of *irreversibility*: the greedy cannot
take back a cell it regrets, cannot re-grab a mackerel it orphaned, cannot shave a wasteful notch to
free length for a better one. So I keep the representation — a connected, hole-free set of grid cells
whose outer boundary traces a simple rectilinear polygon — and I replace the greedy with local search
that can both add *and* remove cells, and that is willing to step downhill to escape the traps the
greedy fell into. This is exactly the method the benchmark's strong baseline uses: simulated annealing
on the net, moving the boundary in and out with cheap incremental scoring. ALE-Agent took this to
fifth place at performance `2880`, so it is the right rung to aim for.

The move set is the natural one for a cell region, and it directly answers the greedy's deficiency. At
each step I either *add* an outside cell adjacent to the region or *remove* a boundary cell of the
region — a single-cell flip that adds or shaves one staircase step. The remove move is the one the
greedy never had: it is what lets the search release a sardine-heavy boundary cell, and the add move
is what lets it re-capture a mackerel cell the carving orphaned. Each flip changes the catch by exactly
that cell's `(#mackerel − #sardine)`, so if I precompute, once, the mackerel count `Av[c]` and sardine
count `Bv[c]` in every cell, the change in `a − b` for any candidate flip is an `O(1)` lookup. That is
the whole game for speed: I never recount fish during the search, I only read the cell's stored counts
and add or subtract them. It is the same linearity I have leaned on since the box, now funding an
incremental score instead of a prefix sum.

I should ask whether single-cell flips are the right *granularity* of move, or whether I am leaving
speed on the table by moving one cell at a time. The tempting alternative is a block move — add or
remove a whole row of frontier cells, or a small rectangle of them, in one step — which would reshape
the region far faster and could vault across a plateau a single-cell walk crawls along. But a block
move breaks the two properties the whole method is built on. Its score change is no longer one lookup
but a sum over the block, `O(block size)`; worse, its validity is no longer a single `3 × 3` test,
because a block's entire new boundary has to be checked for pinches and disconnection, and its
perimeter delta is a whole-perimeter recomputation rather than a `4 − 2s` local update. I would be
trading the `O(1)` flip that funds tens of millions of proposals for an `O(block)` flip that funds far
fewer, and the annealing convergence depends on sheer proposal volume. A second alternative, multiple
restarts from different seed cells, would help escape a bad basin — but warm-starting from the exact
best box already lands the search in a strong basin, so restarts are spending budget to solve a problem
I have largely designed away. And optimizing continuous polygon vertices directly is the fully general
move, but keeping an arbitrary vertex polygon simple under edits is expensive and it abandons the
`O(1)` grid accounting outright. So I keep the single-cell flip: it is the one move for which both score
and validity are `O(1)`, and that is what buys the flip volume the anneal needs.

But `O(1)` scoring is only half of what I need, and the greedy showed me why. The thing that forced the
greedy to be one-shot was its `O(G^2)` hole check: a border flood-fill I could afford once per grown
region but never millions of times. A search that proposes tens of millions of flips cannot pay
`2500` operations per validity test — that would be `~10^{11}` operations and the search would crawl. So
the whole viability of turning this into a search rests on replacing the global flood-fill with a
*local* validity test. Three things must stay legal under every flip. Perimeter: I keep a running count
of boundary unit-edges and apply the same `4 − 2s` delta I derived for the greedy, rejecting any flip
that would push the traced perimeter over budget, `O(1)`. Connectivity and hole-freeness together: a
flip must not disconnect the region and must not open or close a hole, or the boundary stops being a
single simple cycle. The classical result I need here is the digital-topology *simple-point* test — a
foreground cell can be flipped without changing the topology iff, walking the eight-cell ring around
it, the foreground transitions from outside to inside exactly once. That crossing-number check is
`O(1)` on a `3 × 3` window, and it replaces the flood-fill entirely.

I want to hand-check the simple-point test on the cases that matter, because if it is wrong the search
emits illegal nets. Consider *adding* a cell and counting `0→1` transitions around its eight-ring.
An isolated add, all eight neighbors outside: the ring is all zeros, zero transitions — rejected,
which is right, since a lone cell would be a second disconnected component. An add that bridges two
separate arms of the region, say an inside neighbor to the left and another to the right with top and
bottom outside: the ring reads inside, out, inside, out as I walk it, giving two `0→1` transitions —
rejected, which is right, because joining two arms either merges components or seals a hole between
them. An ordinary add on a flat boundary, one inside neighbor and the rest outside: exactly one
transition — accepted, which is right, that is the staircase step I want. The test does what I need on
all three. There is one trap it alone misses: two cells touching only at a corner make the boundary a
figure-eight, not a simple polygon — concretely, cells at `(0,0)` and `(1,1)` inside with `(0,1)` and
`(1,0)` outside, whose boundary pinches at the shared corner vertex. The evaluator demands a *simple*
polygon, so I add a `2 × 2` diagonal-pinch guard that forbids any flip creating such a checkerboard.
With the simple-point test and the pinch guard together, every accepted flip keeps the region a single
simple rectilinear cycle, and both checks are `O(1)` — which is precisely the property the flood-fill
denied the greedy and the reason a search is now affordable at all.

Now the search itself. I anneal: propose a random legal flip; if it improves `a − b`, take it; if it
worsens `a − b` by `Δ`, take it anyway with probability `exp(Δ/T)` for a temperature `T` I cool
geometrically over the run. I want the temperatures calibrated to the actual scale of `Δ`, not guessed.
With `10^4` fish over a `50 × 50 = 2500`-cell grid, a cell holds about `4` fish on average, so a typical
boundary flip moves `a − b` by a single-digit amount, larger in dense cells. At the start temperature
`T0 = 8`, a `Δ = −4` downhill flip is accepted with probability `exp(−4/8) = exp(−0.5) ≈ 0.61`, and a
steeper `Δ = −10` with `exp(−1.25) ≈ 0.29` — so early on the region wanders freely, accepting cells
that locally cost catch, which is what lets it reshape out of the basins the greedy got stuck in. At
the end temperature `T1 = 0.05`, that same `Δ = −4` flip has probability `exp(−80) ≈ 0` — essentially
only improving flips survive and the region settles. The geometric schedule spans exactly the range
from "accept a typical downhill step" to "pure hill-climb," matched to the per-cell delta scale. I keep
the best region ever seen and emit that at the end; the whole bet is that reversibility plus downhill
tolerance finds a configuration the forward greedy never could.

The flip budget is what makes or breaks that bet, and it forces a language decision. The previous two
rungs were Python, fine for a one-shot sweep or a single grow. But annealing needs the region to be
proposed-at millions of times: each flip does an edge-delta over four neighbors, a simple-point test
over eight plus four pinch checks, an `exp` and an RNG draw — a few dozen operations. In C++ that is
tens of nanoseconds per flip, so a `~5`-second budget buys on the order of `10^7`–`10^8` flips; in
Python the same work would be a hundred times slower and deliver perhaps `10^5`–`10^6` flips, far too
few for the temperature to cool through a meaningful anneal. The flip volume *is* the method, so I move
to C++ with a fast xorshift RNG. This is not gold-plating; it is the difference between annealing and
merely jostling.

But where do I *start*? My first instinct is to start from a single seed cell and let the search grow
the whole region, the way the greedy did. When I think it through against the perimeter constraint, that
is wasteful and, worse, it fights the binding constraint. The hard limit here is perimeter, and a
single rectangle is the most perimeter-efficient shape there is — it encloses the most area per unit of
boundary, since a compact region of area `A` spends only about `4√A` on its outline regardless of cell
size. If I make the search grow from a dot, it spends almost its entire flip budget just inflating a
blob up to a reasonable size — to enclose the roughly `1000` cells a big mackerel shoal occupies at
`G = 50` it needs about a thousand accepted add-moves, each waiting for a random proposal to land on the
frontier — and it barely reaches the interesting part, carving notches. The right move is a warm start.
I compute the best perimeter-constrained rectangle exactly with a prefix-sum sweep — the rung-1 idea —
initialize the region to *that*, and let the search spend its whole budget on the thing it is uniquely
good at: cutting staircase notches to release sardine pockets and extending the boundary to grab
outlying mackerel. On seed 2 this matters concretely: warm-starting at the box means the search begins
already holding the `4366` mackerel the box caught, so its job is to shed sardine by removing boundary
cells *while re-adding* any mackerel it orphans — precisely the two-directional trade the greedy could
not make. Starting in a strong basin instead of the empty grid is the difference between refining a
good net and reconstructing a mediocre one.

Let me trace the two-directional trade on a small ledger to see that it recovers exactly what the
greedy lost. Suppose along the boundary the search sits with a sardine-heavy cell inside the region
worth `−6` (say two mackerel, eight sardine) and, just outside, a mackerel cell worth `+7` that the
box's straight edge cut off. The greedy, growing once, either included the `−6` cell early and cannot
remove it, or never reached the `+7` cell because a negative collar blocked it — either way it is
stuck. The search does both as a pair: remove the `−6` cell, `Δ = −(−6) = +6`, and add the `+7` cell,
`Δ = +7`, for a combined `+13` swing in `a − b`. Neither move alone is remarkable; it is the
*sequence* — release a sardine cell here, capture a mackerel cell there — that the one-shot greedy
structurally cannot perform, and it is precisely the `748`-mackerel-and-`1364`-sardine tangle seed 2
left on the table. At high temperature the search will even accept a worse intermediate state to get
from one to the other, which is the whole point of tolerating downhill steps. Multiply this little
`+13` trade across a boundary that has dozens of such misclassified cells and the seed-2 recovery I am
predicting becomes concrete arithmetic rather than hope.

There is an asymmetry between the two directions that I should keep in mind because it biases the
search. Adding a frontier cell with one or two inside neighbors lengthens or holds the boundary
(`4 − 2s` is `+2` or `0`), so adds are gated by the perimeter budget and get refused near the limit.
Removing a boundary cell almost always *shortens* the boundary, so removes are essentially never
blocked by perimeter. This means that when the region is pressed against its length limit — as seeds 1
and 2 already are — the search can freely shed but struggles to extend, so it drifts toward shedding
sardine rather than capturing outlying mackerel. That is fine for seed 2, whose problem is excess
sardine, but it is exactly wrong for seed 1, whose remaining gain lives in mackerel just *outside* a
perimeter-saturated boundary. The asymmetry is another reason I expect seed 1 to lag: the moves that
would help it are the ones the perimeter gate most often refuses.

I also have to fix a grid resolution, and I let the second rung's lesson settle it. There the
perimeter-versus-carving tradeoff pulled in both directions: finer cells carve finer notches but their
tighter unit-edge budget buys fewer steps, coarser cells cover area cheaply but quantize the boundary
bluntly. Because the search reuses one grid for all its millions of flips, I cannot try three
resolutions per instance the way the one-shot greedy did; I need one. I fix `G = 50` — cell side `2000`,
which divides `10^5` exactly so the grid lines are clean — as the middle ground that is fine enough to
carve useful notches and coarse enough that the perimeter budget still wraps a large region. In unit
edges the budget is `400000 / 2000 = 200`, and I hold a safe margin of `20` cell-sides under the cap so
the traced polygon, which can run slightly longer than the running edge count after border effects and
collinear merging, never busts the true `4 × 10^5`. A final true-perimeter check falls back to the best
rectangle if a traced polygon is ever degenerate, so the rung never emits an illegal net and never
regresses below rung 1.

It is worth being precise about how exact the running perimeter count actually is, because the choice
of `G = 50` makes it cleaner than it first looks. Since `50` divides `100000` exactly, every cell side
is exactly `2000` with no rounding, so every boundary unit-edge has length exactly `2000` and the true
traced perimeter is `bedges × 2000` — an identity, not an approximation. Collinear merging fuses
adjacent unit-edges into longer sides but does not change total length, so the running `bedges` I
maintain incrementally through the `4 − 2s` deltas gives me the exact perimeter for free, and the
perimeter gate is therefore exact rather than heuristic. The `20`-cell-side safe margin is then not
covering a length approximation at all; it is pure defense against a few edges of bookkeeping slack at
the grid border and in the trace, so that even a worst-case off-by-a-handful in the edge count cannot
push the emitted polygon over the true `4 × 10^5`. Knowing the accounting is exact matters because I am
going to trust it millions of times without ever retracing the polygon; if it drifted, the search would
slowly wander into illegal territory and only discover it at the final check.

Before I run it I want to lay down falsifiable predictions from the rung-2 table, because the value of
warm-starting from the box is testable seed by seed. The search initializes at the box and keeps the
best-ever region, so on every seed the emitted net's internal `a − b` is at least the box's — which
means seeds 3 and 4, where the greedy fell back to the box, should not merely tie the box but can be
*carved slightly past* it, since the search starts from the box and only ever stores improvements; a
small drop in seed 4's `b = 124` or seed 3's `b = 500` that the greedy's from-scratch rebuild never
found should now be reachable. Seed 2 should show the biggest gain: the search can recover much of the
`748` mackerel the greedy sacrificed while holding the sardine low, pushing `a` back up toward `4366`
with `b` staying small, so its objective should climb well above `3188`. Seed 5, heavily overlapping,
should improve modestly or hold — there is recoverable sardine but the shoals are tangled. And seed 1 is
the one I am genuinely worried about, for a specific reason: the greedy already found a strong carve
there, a `50`-vertex region at the saturated perimeter `393334`. My search does *not* warm-start from
that carved region — it starts from the *box*, `a = 2643`, `b = 510` — and it has to rediscover the
carve from scratch by undirected random flips, under a slightly tighter safe budget. There is a real
chance it lands short of what the greedy already achieved on seed 1, so seed 1 could come in at or
below rung 2 even as the mean rises. If that happens it is not a defect of the method but a symptom of
the next ceiling. The mean should clear `2872`, led by seed 2.

How I *sample* the proposed flip is deliberately simple, and its simplicity is going to be part of the
ceiling. I keep a list of the region's cells; to propose an add I pick a random region cell and a
random one of its four neighbors, and to propose a remove I pick a random region cell directly. That
is cheap and needs no extra structure, but it is blunt in a way I can already predict. A remove is only
legal on a *boundary* cell, yet I sample uniformly over all region cells, so once the region is large
and mostly interior, the great majority of remove proposals land on interior cells and are thrown away
after the boundary test — wasted draws. And an add proposed from a random region cell toward a random
neighbor is just as likely to point at an already-inside neighbor, or at a stretch of boundary that is
already correct, as at the one spot where a misclassified fish sits. So even before the perimeter gate
bites, a large fraction of my millions of proposals are spent looking at boundary that has nothing
wrong with it. This is tolerable because each wasted proposal is `O(1)` and I have tens of millions of
them, but it is pure dilution: the search is undirected, and the fraction of proposals aimed at an
actual error shrinks as the region grows. That dilution, compounded with the perimeter gate refusing
the extend-moves seed 1 needs, is what I expect to see as a plateau.

I should be honest about that ceiling now, because I can see it coming from the same table that
motivates the rung. The single-cell flip is a *local* move, and the perimeter budget is *global* and
binding — seeds 1 and 2 already sit against it. Once the boundary is near its length limit, every
useful notch I want to cut has to be paid for by shaving length somewhere else, and a blind random flip
almost never proposes that coordinated trade: it lands on boundary that is already correct far more
often than on the rare spot where releasing one sardine cell and extending toward one mackerel cell
would both help. The search will plateau not because reversibility is the wrong idea — it plainly fixes
the greedy's sacrificed-mackerel leak — but because the proposals are undirected, and because each
candidate still costs a validity revalidation even when it is pointless. That undirectedness, and the
cost of revalidating each candidate, are exactly the two levers the benchmark's next step pulls to go
from fifth place toward second, and that is the endpoint rung. For now, the job is to show that a
warm-started, reversible, `O(1)`-scored, `O(1)`-validated anneal beats the one-shot greedy — and the
seed-2 recovery is where I expect to see it most clearly.
