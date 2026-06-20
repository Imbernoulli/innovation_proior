The rectangle taught me where the value is and where it isn't. On the seeds where one mackerel
cluster sat apart from the sardine, the box did fine; on the seeds where the species overlap, it
hemorrhaged sardine because a rectangle has no way to bend its boundary around a sardine pocket or
to reach two mackerel pockets without swallowing the gap between them. The problem hands me a far
richer net — any rectilinear simple polygon up to a thousand vertices — and the rectangle uses four
of them. I want a net whose boundary can follow the *shape* of where the mackerel are, and the
most direct way to get a rectilinear region of arbitrary shape is to build it out of grid cells:
choose a connected set of cells, and its outer boundary is automatically a rectilinear polygon.

So here is the representation I will commit to for this rung. Bucket the sea into a `G × G` grid;
give each cell a weight equal to `(#mackerel − #sardine)` inside it. A net is now just a subset of
cells, and `a − b` is the sum of the weights of the chosen cells. The boundary of that subset — the
unit edges where an inside cell meets an outside cell or the grid border — is a rectilinear closed
curve, and as long as the subset is connected and has no holes, that curve is a single simple
polygon I can trace and emit. This turns "design an arbitrary rectilinear net" into "select a good
connected cell region," which is a combinatorial problem I can attack greedily.

The greedy is the obvious one: start from the single densest cell and grow. At each step look at
the frontier — the outside cells adjacent to my current region — and add the one that most
increases the total weight, provided two things hold. First, the perimeter must stay legal: adding
a cell changes the boundary edge count by a known local amount (it removes the shared edges with
neighbors already inside and adds its exposed edges), and I keep a running boundary-edge count so I
can reject a cell that would push the traced perimeter over `4 × 10^5`. Second, I must not punch a
hole: if adding a cell would enclose an empty pocket, the region stops being a simple polygon, so I
flood-fill the outside from the grid border and forbid any addition that leaves an unreachable
empty cell.

When I first run this I hit a wall that I should have seen coming: a pure "add only positive-weight
cells" greedy stops far too early. The frontier of a good region is often a ring of slightly
sardine-heavy cells, and beyond that ring sit rich mackerel cells the greedy will never reach
because it refuses to step through the negative collar. The region freezes small, well below even
the rectangle. So I let the greedy take the *highest-weight* frontier cell at each step even when
that weight is negative — it can spend a little to bridge a gap — but I track the best total weight
the region has ever had and, at the end, restore that best snapshot rather than wherever the walk
happened to stop. That way the bridging is allowed but never charged to the final answer unless it
paid off.

This helps, and the grown regions now genuinely beat the rectangle on the overlapping seeds, where
the rectilinear boundary carves sardine out of a mackerel shoal. But the greedy is still
unmistakably fragile, and I want to be honest about why. It is a single forward pass with no way to
undo an early inclusion: a cell that looked good when the region was small can become a liability
once the region's shape is settled, and the greedy cannot remove it. The perimeter ceiling makes
this worse — a ragged, many-notched boundary spends its edge budget fast, so on some layouts the
greedy saturates the perimeter on a mediocre region and simply has no budget left to fix it. And
the grid is static and coarse: too coarse and the boundary cannot hug the fish closely; too fine
and each cell is tiny, so the fixed perimeter budget covers far less area and the region stays
small. There is no single resolution that is right for every instance.

Two cheap defenses make the rung robust rather than brilliant. I run the greedy at a few grid
resolutions and keep whichever region scores best by the internal weight — different layouts prefer
different cell sizes, and trying three is nearly free. And I include the rung-1 rectangle itself as
one more candidate, computed the same way by a prefix-sum sweep, so that whenever the grown region
fails to beat the box, I simply fall back to the box. This guarantees the rung *dominates* the
previous one: it is the rectangle plus the option of a carved rectilinear region, and it picks the
better of the two by their internal estimates, with the exact evaluator having the final word.

So this rung is the right second step and no more. It replaces the four-cornered box with a region
whose boundary can bend, it respects perimeter and hole-freeness by construction, and it strictly
improves on the rectangle by keeping the rectangle as a fallback. But its engine is a one-shot
greedy with no reversibility and a coarse static grid, and that is exactly its ceiling: it cannot
trade a bad cell now for a better configuration later, it wastes perimeter on ragged boundaries,
and it is at the mercy of a single grid resolution. The way past all three is the same move — stop
treating region construction as a forward-only greedy and start treating it as *search*, where
cells can be both added and removed, downhill steps are accepted to escape the greedy's traps, and
the only cost I pay is making each candidate move cheap enough to afford millions of them. That is
the next rung.
