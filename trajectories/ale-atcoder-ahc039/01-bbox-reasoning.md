I need to put down one legal net and catch more mackerel than sardine, and before I reach for
anything clever I want the simplest object that is guaranteed valid and gives me a number to
beat. A single axis-aligned rectangle is always a legal net: four vertices, inside `4 ≤ m ≤
1000`; edges axis-parallel by construction; trivially simple; and its perimeter `2(w + h)` with
`w, h ≤ 10^5` reaches the `4 × 10^5` cap only at the full square, so every proper sub-box is
under budget. On this rung the perimeter ceiling never really binds, but I still fold the guard
into the loop — it costs one comparison, and the moment the boundary is allowed to bend into a
ragged shape that same length budget becomes *the* binding constraint. The only real question is
*which* rectangle.

Fix the sides at `x ∈ [x1, x2]`, `y ∈ [y1, y2]`: then `a − b` is a sum of per-point
contributions, `+1` per mackerel and `−1` per sardine. Stamp those weights and a rectangle's
value is the total weight it covers, so the best box is the maximum-weight axis-aligned rectangle
on a signed point set. I cannot try every real-valued box, but I do not need to: slide any optimal
box's right edge left until it reaches the next data `x`-coordinate — the strip swept over held
no points, so the weight is unchanged — and the same holds for all four edges independently. There
is always an optimum with every side on a data line, so the search is finite and prefix sums make
each box a constant-time read. This additive decomposition of `a − b` over the covered points is
what later lets a net that changes one cell at a time update its score in O(1).

Why `a − b` and not `a` alone: an objective of "maximize `a`" keeps extending every edge outward
and ends as the bounding box of all mackerel, dragging in every interleaved sardine. The signed
sweep instead stops each edge exactly where the marginal fish flip from net-helpful to net-harmful
— ten mackerel against twelve sardine moves `a − b` by `−2` and is rejected. For a rectangle those
four stopping points decouple into a clean polynomial sweep; for an arbitrary rectilinear polygon
the "where does the boundary want to be" question couples across the whole boundary, which is why
I solve the box exactly here and leave the polygon to search later. The bounding-box-of-mackerel
alternative is good only when the species are already cleanly separated — the easy case I do not
need help with — while the max-weight box is the one member of the design space I can solve exactly
up to grid resolution, which is what a floor should be. Both degenerate extremes score a flat `1`
(empty net: `a = b = 0`; full square: `a − b = 0`), so the value lives entirely in the asymmetry a
well-placed boundary creates between the species.

Concretely: lay a coarse uniform grid of cut lines across `[0, 10^5]`, bucket every fish into its
cell accumulating `(#mackerel − #sardine)`, build a 2D prefix-sum table, then sweep all `(x1 < x2,
y1 < y2)` grid-line boxes reading each `a − b` in O(1) and keep the best while skipping any whose
perimeter exceeds budget. Each fish is bucketed by binary-searching its coordinate and clamped into
the last real cell, so a fish exactly at `x = 10^5` lands in-range rather than off the table.

The rectangle's ceiling shows in miniature. Take a `3 × 3` grid: a ring of eight `+5` cells around
a `−50` centre. Any box containing the centre eats the `−50`; to dodge it the box must live in a
single row or column of the ring, whose best is `3 × 5 = 15`. But the ring itself — a shape a
rectilinear polygon could trace by looping around the pit — is worth `40`. A convex single body
cannot enclose the ring without also enclosing the hole. The exact ratio is an artifact of the
`−50` I chose, but the direction generalizes: the more a sardine pocket sits interior to a mackerel
shoal, the larger the fraction of value a rectangle must forfeit.

The resolution is a knob, and cost grows like `K^4`: `K = 60` gives about `3.3 × 10^6` boxes,
comfortable for a Python sweep, while `K = 200` would be `~4 × 10^8`, far too slow. A finer grid
finds a better rectangle, never a non-rectangle, so it never touches the shape ceiling the doughnut
exposed. Kadane's max-subarray trick would collapse a dimension to `O(G^3)`, but it hides the box's
width inside the subarray and makes the per-box perimeter test awkward, and at `K = 60` the naive
sweep already finishes in about a second — I fix `K = 60`.

One subtlety in scoring carries forward: I optimize a grid estimate while the evaluator does an
exact integer point-in-polygon test with boundary-counts-as-inside. For a box whose sides are grid
lines the buckets count fish exactly; the only possible mismatch is a fish sitting precisely on a
chosen edge, where my half-open convention and the evaluator's boundary rule can disagree by one —
a fraction of a fish across the whole instance. I report only the evaluator's exact score, so the
discretization guides the choice and never inflates the result.

A single rectangle does well when the mackerel cluster densest in one compact region with sardine
mostly elsewhere, and poorly in exactly the case the problem is built to create: overlapping shoals,
where any box large enough to capture a mackerel pocket swallows the sardine threaded through it.
So the a-priori prediction is that the sardine count `b` should be conspicuously large on the
overlapping seeds relative to the separated ones, dragging the objective down; if instead the box
scores uniformly across seeds, my overlapping-versus-separated picture is wrong.

The sweep is fully deterministic, so the same instance always yields the same box — a baseline that
jittered run to run would be useless as a measuring stick. And I emit exactly `m = 4` vertices, the
minimum legal count: handed up to a thousand vertices and a long perimeter, this rung uses four and
one convex body. That gap between what the constraints permit and what a rectangle exploits is the
opportunity the next rung inherits.
