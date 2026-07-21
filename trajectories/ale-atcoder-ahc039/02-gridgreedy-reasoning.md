The rectangle taught me where the value is, and the seed-by-seed numbers make the lesson
quantitative. The sardine count `b` is the whole story: `510, 1795, 535, 114, 1207` across the five
seeds. Every one of those sardine is a point I could stop catching if the boundary routed around it,
and since the objective is `a − b + 1`, shedding a sardine while holding the mackerel is worth a
full point. So the recoverable value per seed — keep the current mackerel, drop the sardine — is
exactly those numbers, averaging `832`, an enormous headroom on the `2704` mean and distributed
wildly unevenly: seed 2 carries `1795`, seed 5 carries `1207`, while seed 4 is nearly clean at
`114`. Seed 2's box already catches `4366` of `5000` mackerel but drowns in `1795` sardine — not a
case where I need more mackerel, a case where I need to carve sardine out of a region I have already
enclosed. Seed 4, at `114` sardine, has almost nothing to recover, and clever carving there risks
net harm. This rung's job is sardine excision on the overlapping seeds, done carefully enough not to
break the clean ones.

The perimeters carry a second warning. Rung 1 spent `203334, 333334, 333334, 180000, 313334`,
leaving headroom `196666, 66666, 66666, 220000, 86666`. This lands badly: seed 2, the `1795`-sardine
opportunity, has only about `66666` of perimeter left, the least of any seed — precisely the
instance where I most want to snake the boundary around interior sardine, I have the least length
to do it with. The constraint and the opportunity are misaligned, so whatever I build has to be
efficient with perimeter exactly where it is scarcest.

The problem hands me any rectilinear simple polygon up to a thousand vertices; the rectangle used
four. I want a net whose boundary follows the shape of where the mackerel are. A union of a few
glued rectangles is more expressive but still cannot cut an arbitrary interior notch to release a
buried pocket; a per-column vertical interval sheds sardine cheaply along one axis but is blind to
pockets needing the boundary to reach in from two sides; optimizing polygon vertices directly is
fully general but keeping an arbitrary-vertex polygon simple under edits is a minefield. The
representation that subsumes the L, the comb, and the staircase while giving a clean local edit is a
grid of cells: choose a connected set of cells, and its outer boundary is automatically a
rectilinear polygon.

Bucket the sea into a `G × G` grid, cell weight `(#mackerel − #sardine)`. A net is a subset of
cells, `a − b` is the sum of chosen weights, and the boundary is a rectilinear closed curve; as long
as the subset is connected and hole-free, that curve is a single simple polygon I can trace.
Designing an arbitrary net becomes selecting a good connected, hole-free cell region, and the value
again decomposes additively, so adding or removing one cell is a local update.

Two local quantities drive the method. Adding cell `c` changes the score by exactly `W[c]`, one
lookup. The perimeter change takes counting: a cell has four unit edges, and with `s` of its four
neighbors already inside, the boundary-edge count changes by `4 − 2s` — `+4` for a lone cell, `0`
tucking into an L-corner, `−4` filling a one-cell dent. This has a consequence worth flagging: the
greedy will find it cheapest to fill concavities (`s = 2, 3` cost zero or negative perimeter),
biasing toward convexification — the opposite of the notch-cutting I want for carving sardine. The
perimeter accounting quietly pulls against the objective.

The most direct build is to grow the region: start from the densest cell and at each step add the
frontier cell that most increases weight, under two guards. Perimeter stays legal via the running
`4 − 2s` count against a safe budget of `396000` under the cap. And I must not punch a hole: if an
add seals off an empty pocket the region stops being simple, so I flood-fill the outside from the
grid border and forbid any add that leaves an unreachable empty cell — an exact test, since an
outside cell the exterior cannot reach is by definition enclosed. That flood-fill is `O(G²)`,
affordable once per grown region but not millions of times, which is exactly what makes this a
one-shot greedy rather than a search.

A greedy adding only positive-weight cells stops far too early, because the frontier of a good
region is often a ring of slightly sardine-heavy cells with rich mackerel cells beyond it, and the
greedy refuses to step through the negative collar. So I let it take the highest-weight admissible
frontier cell even when negative — spending a little to bridge a gap — while tracking the best total
weight ever seen and restoring that snapshot at the end. A bridge that never pays off is never
charged to the answer; a bridge that reaches a `+20` pocket past a `−3, −3` collar nets `+14` and
becomes the stored region. The patience before giving up is `G × G` steps, generous so a thick
collar can still be crossed.

Connectivity is free: I only add cells adjacent to the region and never remove, so it stays one
connected blob. Hole-freeness is not free — a `3 × 3` ring with the centre left out has two boundary
cycles, and the border flood-fill catches exactly this. Once hole-free, the trace walks the unit
edges where inside meets outside, threads them into one directed cycle, and merges collinear runs so
the vertex count stays far below `1000`.

Where should this help? The seeds carrying the most recoverable sardine — seed 2 at `1795`, seed 5
at `1207` — are where a carved boundary gains most. But seed 2 is doubly bound: the opportunity is
largest and the perimeter headroom smallest, so the notches may cost more length than I have and the
carve sheds only part of the `1795`. And this is a single forward pass with no undo — a cell that
looked good early can become a liability once the shape settles, and the greedy can neither remove
it nor re-grab a mackerel it orphaned. So I expect real but fragile gains on the overlapping seeds
and near-ties on the clean ones. The sharpest a-priori consequence is on seed 2: to shed sardine the
region must route between interleaved fish, and mackerel stranded on the wrong side of a carved
pocket get orphaned with no reach-back move, so `a` should fall meaningfully below `4366` even as
`b` falls well below `1795`, the net still rising — a real gain bought by sacrificing catch it
cannot recover.

Two cheap defenses, both from the table. Because the perimeter-versus-carving tradeoff plays out
differently at different cell sizes — a finer grid cuts finer notches but its tighter unit-edge
budget buys fewer steps — I run `G ∈ {30, 40, 50}` and keep the best by internal weight; three grows
are nearly free next to the flood-fills. And I include the rung-1 rectangle itself as a candidate,
computed on a `50 × 50` grid by the same prefix-sum sweep, so whenever the grown region fails to
beat the box — which I expect on the clean seeds — I fall back to it. This guarantees the rung
dominates rung 1: the rectangle plus the option of a carved region, picked by internal estimate with
the exact evaluator having the final word. Ranking by internal weight is faithful because a
grid-aligned region's weight is the exact fish count up to the same fraction-of-a-fish boundary
caveat as the box.

So the engine is a one-shot greedy with no reversibility, forced into that shape by an `O(G²)` hole
check too expensive to run millions of times. It cannot trade a bad cell now for a better
configuration later, it wastes perimeter on ragged boundaries it cannot repair, and it is at the
mercy of one grid resolution. The way past all three is the same move: treat region construction as
search, where cells are both added and removed and downhill steps are accepted, and the price of
admission is the thing the flood-fill denied — a validity check cheap enough to afford millions of
times.
