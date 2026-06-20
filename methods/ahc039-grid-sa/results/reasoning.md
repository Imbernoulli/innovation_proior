The greedy region taught me two things at once: that a rectilinear net built from grid cells is the
right object, and that growing it by a one-shot forward pass is the wrong engine. The greedy cannot
take back a cell it regrets, it spends its perimeter budget on whatever ragged boundary it stumbles
into, and it lives or dies by a single grid resolution. Every one of those failures is a failure of
*irreversibility*. So I keep the representation — a connected, hole-free set of grid cells whose
outer boundary traces a simple rectilinear polygon — and I replace the greedy with local search
that can both add *and* remove cells, and that is willing to step downhill to escape the traps the
greedy fell into. This is exactly the method the benchmark's strong baseline uses: simulated
annealing on the net, moving the boundary in and out, with cheap incremental scoring. ALE-Agent
took this to fifth place at performance `2880`, so it is the right rung to aim for.

The move set is the natural one for a cell region. At each step I either *add* an outside cell
adjacent to the region or *remove* a boundary cell of the region — a single-cell flip of the
boundary, which adds or shaves one staircase step. Each flip changes the catch by exactly that
cell's `(#mackerel − #sardine)`, so if I precompute, once, the mackerel count and sardine count in
every cell, then the change in `a − b` for any candidate flip is an O(1) lookup. That is the whole
game for speed: I never recount fish during the search, I only read the cell's stored counts and
add or subtract them. This is the incremental scoring the benchmark relies on, done here against a
grid index instead of a kd-tree, but with the same payoff — millions of candidate moves per second.

Three things have to stay legal under every flip, and each needs a cheap incremental check.
Perimeter: I keep a running count of boundary unit-edges; adding or removing a cell changes it by a
small local amount I can compute from the cell's four neighbors, so I reject any flip that would
push the traced perimeter over `4 × 10^5` without ever retracing the polygon. Connectivity and
hole-freeness: a flip must not disconnect the region or open or close a hole, because then the
boundary stops being a single simple polygon. Recomputing global connectivity after every flip
would destroy the speed, so I use the classical digital-topology test — a cell flip preserves the
region's topology iff the foreground transitions exactly once as you walk the eight-cell ring
around it (the "simple point" condition). That is a constant-time check on a 3×3 window. There is
one more trap the simple-point test alone misses: two cells touching only at a corner make the
boundary a figure-eight, not a simple polygon, so I also forbid any flip that would create a
diagonal pinch in a 2×2 block. With both checks the traced boundary is always a single simple
rectilinear cycle, which is what the evaluator demands.

Now the search itself. I anneal: propose a random legal flip; if it improves `a − b`, take it; if
it worsens `a − b` by `Δ`, take it anyway with probability `exp(Δ/T)` for a temperature `T` I cool
geometrically over the run. Early, when `T` is high, the region wanders — accepting cells that
locally cost catch — so it can reshape itself out of the basins the greedy got stuck in; late, when
`T` is low, only improving flips survive and the region settles. I keep the best region ever seen
and emit that at the end. The whole bet is that reversibility plus downhill tolerance finds a
configuration the forward greedy never could.

But where do I *start*? My first instinct is to start from a single seed cell and let SA grow the
whole region, the way the greedy did. When I think it through, that is wasteful and, worse, it
fights the binding constraint. The hard limit here is perimeter, and a single big rectangle is the
most perimeter-efficient shape there is — it encloses the most area per unit of boundary. If I make
SA grow from a dot, it spends almost its entire budget of moves just inflating a blob up to a
reasonable size, paying boundary cost the whole way, and barely gets to the interesting part:
carving notches. The right move is a warm start. I compute the best perimeter-constrained rectangle
exactly with a prefix-sum sweep — the rung-1 idea — initialize the region to *that*, and let SA
spend its entire budget on the thing it is uniquely good at: cutting staircase notches to release
sardine pockets and extending the boundary to grab outlying mackerel. Starting in a strong basin
instead of the empty grid is the difference between SA refining a good net and SA reconstructing a
mediocre one.

I also have to choose a grid resolution, and the perimeter ceiling makes this a real tension.
Finer cells let the boundary hug the fish more closely, but each cell is smaller, so a fixed
perimeter budget — a fixed number of unit-edges times the cell side — covers far less area, and the
region cannot grow large enough to hold the mackerel. Coarser cells cover more area per edge but
quantize the boundary so the notches are blunt. When I sweep resolutions I find a middle grid that
balances the two: fine enough to carve useful notches, coarse enough that the perimeter budget
still wraps a large region. I fix that resolution for the rung.

What I expect: the warm-started, reversible SA should clearly beat the one-shot greedy, because on
the overlapping seeds it can do what the greedy cannot — start from the best box and then *remove*
the sardine-heavy cells along the boundary and *add* the mackerel cells just outside it, trading the
net's shape against the catch in both directions. On the easy seeds where the box is already
near-optimal, SA should hold the box (it has nothing better to find), so the rung never regresses.
And I should be honest about the ceiling I am about to hit. The single-cell flip is a *local* move,
and the perimeter budget is *global* and binding: once the boundary is near its length limit, every
useful notch I want to cut has to be paid for by shaving length somewhere else, and a blind random
flip rarely proposes that coordinated trade. The search will plateau not because the idea is wrong
but because the proposals are undirected — most random flips touch boundary that is already correct,
and the few that would fix a genuinely misclassified fish are drowned out. That undirectedness, and
the cost of revalidating each candidate, are exactly the two levers the benchmark's *next* step
pulls — and that is the endpoint rung.
