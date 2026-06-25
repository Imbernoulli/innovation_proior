The rectangle taught me where the value is and where it isn't. On the seeds where one mackerel
cluster sat apart from the sardine, the box did fine; on the seeds where the species overlap, it
hemorrhaged sardine because a rectangle has no way to bend its boundary around a sardine pocket or
to reach two mackerel pockets without swallowing the gap between them. The problem hands me a far
richer net — any rectilinear simple polygon up to a thousand vertices — and the rectangle uses four
of them. So I want a net whose boundary can *follow the shape* of where the mackerel are, and I
need a way to describe such a boundary that I can actually search over.

The most direct handle I can think of is to stop describing the boundary at all and describe the
*region* instead. Bucket the sea into a `G × G` grid; give each cell a weight equal to
`(#mackerel − #sardine)` inside it. Then a net is just a subset of cells, `a − b` is the sum of the
chosen weights, and the net's outline is whatever unit grid-edges separate an inside cell from an
outside cell or from the border. If that outline is a single closed loop I can trace it and emit
the polygon. So the whole question reduces to: which connected subset of cells has the largest
weight sum, subject to its outline being one simple loop of length `≤ 4 × 10^5`?

Before I trust that reduction I should check that the outline of a cell subset really *is* a single
clean rectilinear polygon, because that is the part I am taking on faith. Take a `2 × 2` block of
cells and walk its boundary the way I intend to in code — for each inside cell, emit the directed
unit-edge on each side that faces outside, oriented so the loop is consistent. The `2 × 2` block
emits 8 directed unit-edges, and chaining them by "head of one = tail of the next" gives
`(2,2) → (3,2) → (4,2) → (4,3) → (4,4) → (3,4) → (2,4) → (2,3) → back to (2,2)`: it closes, and it
visits all 8 edges exactly once. Merging the collinear runs collapses the four straight sides into
their corners and leaves a 4-vertex square, which is exactly the right polygon. So for a filled
convex block the trace behaves; I will have to keep an eye on whether it stays a single loop for
more ragged regions (a thin diagonal pinch could in principle split it), but the construction is
sound enough to build on.

The selection problem itself — largest-weight connected hole-free cell region under a perimeter
cap — is the kind of thing that has no clean exact algorithm, so I will grow a region greedily and
see how far that gets. Start from the single densest cell and, at each step, look at the frontier
(outside cells adjacent to the current region) and add the most valuable admissible one. "Admissible"
has to encode the two constraints. The perimeter one I want to maintain incrementally rather than
re-tracing every step: when I add a cell, it contributes its 4 unit-edges to the boundary but cancels
two boundary edges for every neighbor already inside (the shared edge stops being a boundary). So the
running count should change by `4 − 2·(inside neighbors)`. I do not want to just assert that, so let
me grow a small region — single cell, then an L, then close it to a `2 × 2` — and compare the running
count `4 − 2k` against a brute-force recount of exposed sides at each step. Single cell: 4, recount 4.
Add a cell with one inside neighbor: `4 − 2 = 2`, running total 6, recount 6. Add the third with one
inside neighbor: `+2`, total 8, recount 8. Close the square — the fourth cell touches two inside
neighbors, so `4 − 4 = 0`, total stays 8, recount 8. Every step agrees, so the incremental rule is
exact and I can carry one integer instead of re-tracing.

That integer counts unit grid-edges, and the world perimeter is that count times the cell width
`cw = 10^5 / G`. It is worth sanity-checking the units against the cap. The whole grid's border is
`4G` unit edges, so the full-grid perimeter is `4G · (10^5/G) = 4 × 10^5` — exactly the cap, at every
resolution. That is reassuring: the budget is precisely "you may outline the entire sea once," and any
real region spends less. There is one wrinkle: I emit vertices at rounded coordinates `round(i·cw)`,
and for `G = 30` the grid lines fall at `3333.33…`, so each is off by up to a third of a unit; a polygon
with a few hundred edges could drift a little past a budget I computed from exact `cw`. So I will check
against a safe budget of `396000` rather than `400000`, which leaves `4000` world units of slack —
comfortably more than the few-hundred-edges-times-fraction-of-a-unit worst case. (For `G = 40` and
`G = 50` the grid lines are integers and there is no drift at all.)

The hole constraint is the other half of admissibility. If adding a cell seals off an empty pocket,
the region's outline is no longer one loop — it becomes the outer boundary plus a hole boundary, which
is not a simple polygon. The clean test is: flood-fill the outside starting from the grid border; if any
empty cell is unreachable, it is trapped inside, so I reject the addition. This is `O(G²)` per step,
which I can afford at these grid sizes.

So I code the greedy: seed the densest cell, repeatedly add the highest-weight admissible frontier
cell. And on the overlapping seeds it stalls badly — it grabs the one dense cell and a couple of
neighbors and then quits, scoring below the rectangle. I want to understand the stall concretely
rather than guess, so I build a toy grid: two mackerel pockets of weight `+5`, one at cell `(1,1)` and
one at `(1,5)`, with a thin wall of `−1` cells at `(1,2),(1,3),(1,4)` between them and zeros everywhere
else. A greedy that only ever adds a positive-weight cell seeds at `(1,1)`, finds every frontier cell
is `−1` or `0`, and stops — final region `{(1,1)}`, weight `5`. It can never cross the `−1` collar to
reach the second `+5`, even though the two pockets together are worth `10`. That is the stall, and it is
structural: a good region's frontier is frequently a ring of mildly sardine-heavy cells with richer
mackerel just past it.

The fix that the toy suggests is to let the greedy spend: take the highest-weight frontier cell even
when it is negative, so the region can pay a little to bridge to something better. But then the walk
will wander past its own peak — it keeps eating negative cells once the good ones run out — so the
final state is not what I want to keep. So I track the best total weight the region has ever reached
and, when the walk ends, restore that best snapshot rather than wherever it stopped. Re-running the
toy with this rule: the region now does reach both pockets and the best snapshot it restores has
weight `10`, double the positive-only result. One detail I did not predict and find worth noting: in
the toy it reached the second pocket by routing *around* the `−1` wall through the zero-weight cells
on the border rather than driving straight through the collar — the bridging does not have to pay the
worst price, it pays whatever the cheapest detour costs, and the snapshot restore means even a wasteful
detour is never charged to the answer unless it paid for itself. That is exactly the behavior I wanted,
arrived at honestly.

With bridging plus snapshot restore the grown regions genuinely beat the rectangle on the overlapping
seeds, where the rectilinear boundary carves sardine out of a mackerel shoal. But I want to be honest
about the limits, because they are real and they bound this rung. It is a single forward pass with no
way to undo an early inclusion: a cell that looked good when the region was small can become a liability
once the shape is settled, and the greedy cannot remove it. The perimeter ceiling sharpens this — a
ragged, many-notched boundary spends its edge budget fast (every notch is `+2` edges that buy no
extra cell), so on some layouts the greedy saturates the perimeter on a mediocre region with no budget
left to fix it. And the grid is static and coarse: too coarse and the boundary cannot hug the fish;
too fine and each cell is tiny, so the fixed perimeter budget covers far less area and the region
stays small. There is no single resolution that is right for every instance.

Two cheap defenses make the rung robust rather than brilliant. I run the greedy at a few grid
resolutions — `G ∈ {30, 40, 50}` — and keep whichever region scores best by internal weight; different
layouts prefer different cell sizes, and trying three is nearly free. And I include the rung-1 rectangle
itself as one more candidate, computed by a prefix-sum sweep, and take the best-scoring candidate
overall. I should be careful about what that buys me. Because the rectangle is always in the candidate
pool and I keep the maximum internal estimate, the selected candidate's internal weight is at least the
rectangle's — so by the internal estimate this rung never scores below the box. The frozen evaluator,
not the internal weight, has the final word on the reported number, and the internal weight is only an
estimate of `a − b` (it can disagree with the exact count at the margins), so I will phrase this as "the
rung keeps the rectangle as a floor and adds the option of a carved region," and let the harness confirm
it on the five seeds rather than claiming dominance outright.

So this rung is the right second step and no more. It replaces the four-cornered box with a region
whose boundary can bend, respects perimeter and hole-freeness by construction, and keeps the rectangle
as a fallback. But its engine is a one-shot greedy with no reversibility and a coarse static grid, and
that is exactly its ceiling: it cannot trade a bad cell now for a better configuration later, it wastes
perimeter on ragged boundaries, and it is at the mercy of a single grid resolution. The way past all
three is the same move — stop treating region construction as a forward-only greedy and start treating
it as *search*, where cells can be both added and removed, downhill steps are accepted to escape the
greedy's traps, and the only cost I pay is making each candidate move cheap enough to afford millions
of them. That is the next rung.
