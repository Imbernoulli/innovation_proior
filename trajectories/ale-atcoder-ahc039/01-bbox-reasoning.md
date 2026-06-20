I need to put down one legal net and catch more mackerel than sardine, and before I reach for
anything clever I want the simplest object that is guaranteed valid and gives me a number to beat.
The constraints are forgiving about one shape in particular: a single axis-aligned rectangle is
always a legal net. It has four vertices, which sits inside `4 ≤ m ≤ 1000`; its four edges are
axis-parallel by construction; it is trivially simple; and as long as I keep it from spanning the
whole grid its perimeter stays under `4 × 10^5`. So a rectangle is the floor I can stand on without
any risk of producing an invalid output, and the only question is *which* rectangle.

The objective inside a rectangle is clean. If I fix the four sides at `x ∈ [x1, x2]`, `y ∈ [y1,
y2]`, then `a − b` is just the number of mackerel in that box minus the number of sardine in it.
That is a sum of per-point contributions where each mackerel counts `+1` and each sardine counts
`−1`, so if I imagine a weight `+1` stamped at every mackerel and `−1` at every sardine, the value
of a rectangle is the total weight it covers. Maximizing `a − b` over all axis-aligned boxes is
therefore the classic *maximum-weight axis-aligned rectangle* problem on a point set with signed
weights. I cannot literally try every real-valued box, but I do not need to: the optimum's sides
can always be snapped to coordinate lines through the data, so the search is finite, and I can make
it cheap with prefix sums.

Here is the move. I lay down a grid of candidate vertical and horizontal cut lines — a coarse
uniform set across `[0, 10^5]` is enough to find a strong box, and refining it costs only more grid
lines. I bucket every fish into the cell of that grid it falls in, accumulating per-cell
`(#mackerel − #sardine)`. Then I build a 2D prefix-sum table over the cells, so the signed weight
of *any* rectangle aligned to the grid lines is a constant-time difference of four table entries.
Now I can sweep over all `(x1 < x2, y1 < y2)` choices of grid lines, read off each box's `a − b` in
O(1), and keep the best — while skipping any box whose perimeter `2·(width + height)` would exceed
`4 × 10^5`. That last check matters: a box that is too wide or too tall is illegal no matter how
good its catch, so the perimeter ceiling is a real constraint even on a rectangle, and I bake it
into the loop rather than discovering it at scoring time.

There is a subtlety in *how* I score, and it is worth getting right because it will haunt every
later rung. The grid I use for the prefix-sum search is a discretization: a box snapped to coarse
grid lines is only an approximation to the true best box, and a fish near a boundary line could be
counted on the wrong side. But the *evaluator* is exact — it does a real integer point-in-polygon
test with on-boundary-counts-as-inside — so the number I optimize internally (a grid estimate) and
the number I am graded on (the exact count) can differ slightly. For a rectangle this gap is small
and harmless: the grid just picks a good box, and the exact evaluator then reports its true `a − b
+ 1`. I will let the grid resolution be fine enough that the estimate tracks the truth closely, and
I will *report only the evaluator's exact score*, never the internal estimate.

What do I expect this to buy, and where will it fall short? A single rectangle will do surprisingly
well when the mackerel cluster densest in one compact region with sardine mostly elsewhere — then a
box around that region scores high. It will do poorly in exactly the situation the problem is built
to create: mackerel and sardine drawn from *overlapping* shoals, so that any rectangle large enough
to capture a mackerel pocket also swallows the sardine threaded through it, and any rectangle small
enough to dodge the sardine also clips off mackerel at the edges. A rectangle is convex and
single-bodied; it cannot reach around a sardine cluster, it cannot bridge two separated mackerel
pockets without paying for the sardine-rich gap between them, and it has no way to cut a notch. The
whole point of the problem — that the net is an arbitrary rectilinear polygon — is precisely the
freedom a rectangle throws away.

So I am proposing the bounding box as the *right* first rung and nothing more: it is the simplest
guaranteed-legal net, it is found exactly (up to grid resolution) by a fast prefix-sum sweep
respecting the perimeter budget, and it gives a concrete, honest baseline objective that every
later, more flexible net must beat. The limitation is baked into its shape, not its tuning: with
only four corners and one convex body, it cannot separate interleaved species. The moment I want to
exclude a sardine pocket sitting inside a mackerel shoal, I need a boundary that can bend — a
rectilinear region with many vertices — and that is what the next rung has to build, by giving up
the single box for a *grid of cells* I can selectively include.
