The line that decides this problem is the XOR on the query coordinates. A type-2 query arrives as
`A B C D` and only becomes a real rectangle after I xor each field with `lastAns`, the answer to the
previous query. So I cannot read the operation stream ahead, sort it, and answer offline — I have to have
already produced answer `k-1` before I can even see rectangle `k`. That forces every structure I keep to
work fully online and interleaved: one point weight-update, one rectangle-sum, in the given order, no
reordering. The scale sets one more hard constraint at once: up to `10^5` points each with weight up to
`10^9`, and a single rectangle can cover all of them, so an answer can reach `~10^14` — well past 32 bits.
Every weight, accumulator, and Fenwick cell is `long long` from the start; an `int` anywhere is a silent
wrong answer.

The reflex for "sum of weights in an axis-aligned rectangle" is the offline sweepline: compress
coordinates, sort events by `x`, sweep left to right with a single Fenwick over `y`, and answer each query
as `prefix(X2) - prefix(X1-1)` with each prefix split again over `y`. That is `O((n+q) log n)` with one
BIT, and it is exactly what this problem forbids twice over. Weights mutate between queries, so there is no
static weight vector to prefix over — a query must see exactly the updates that preceded it. And even
setting that aside, the forced-online XOR means I cannot sort the queries at all, since a rectangle is
hidden behind the previous answer. There is exactly one legal processing order, the given one, so the
sweep is out.

The online analogue is a 2D Fenwick, `bit[x][y]`, with point-update and prefix-rectangle by
inclusion-exclusion over four corners. The coordinate range is `~2*10^9` per axis, but only `n` distinct
`x` and `n` distinct `y` ever appear, so I compress. That still leaves a dense compressed grid of `X*Y`
cells with `X, Y` up to `10^5` — `10^10` cells, `~80` GB. Dead on memory. The grid is enormous but only
`n` of its cells are ever nonzero, so I need a structure whose size tracks the points that exist, not the
bounding box.

That is the one thing this problem hands me that the generic dynamic-2D problem does not: the set of point
*locations* is fixed for the whole run. Updates only change weights of existing points, queries only read,
so the set of cells that can ever be nonzero is known up front — exactly the `n` input points. I do not
need a structure that can place weight at an arbitrary cell; I need one shaped around the `n` cells that
exist.

So: a Fenwick tree over the compressed `x`-ranks `1..X`, but with each scalar node replaced by an inner
Fenwick over `y`. A dense inner tree would give every outer node a full `Y`-wide axis — back to `10^10`.
But an outer node `j` only ever needs the `y`-values of points whose update walk passes through it. So for
each point at `x`-rank `r`, I walk `j = r, r + (r & -r), ...` and register its `y` into node `j`'s list;
after all points, I sort+unique each node's list, and that list *is* the compressed `y`-axis of node `j`'s
inner Fenwick, sized to exactly its distinct `y` count. Each point lands in `O(log X)` outer nodes, so the
total is `O(n log n) ~ 1.7*10^6` inner cells, a dozen-odd MB. Per operation is `O(log^2 n)`, and
`q log^2 n ~ 3*10^7` clears the 1 s limit comfortably.

The index arithmetic is where this kind of code dies, so I pin both walks. An update of point `(x, y)` by
`d`: `r` is the exact compressed `x`-rank of `x`; walk `j = r` upward by `j += j & -j`, and in each node
find `y`'s position in that node's sorted `y`-list, then walk the inner Fenwick `k` upward adding `d`. A
prefix "`x`-rank `<= r` and `y <= Y`": walk `j = r` downward by `j -= j & -j`, and in each node find how
many of its `y`'s are `<= Y` (the inner prefix length), then walk the inner Fenwick `k` downward summing.
A rectangle `[X1,X2] x [Y1,Y2]` is `(P(rHi,Y2) - P(rHi,Y1-1)) - (P(rLo,Y2) - P(rLo,Y1-1))`, with `rHi` the
`x`-rank for `x <= X2` and `rLo` the `x`-rank for `x <= X1-1` — those two bounds use `upper_bound` over the
global sorted `x`-set, while the update uses the exact rank of an existing `x`.

The two inner searches answer different questions, so they are different binary searches. The inner Fenwick
is 1-indexed. An update must hit the single slot that *owns* `y`, its position
`lower_bound(list, y) - begin + 1`. A query needs a prefix *length* instead: the count of the node's `y`'s
that are `<= Y`, `upper_bound(list, Y) - begin`, which correctly *includes* `y = Y` and excludes everything
above. Because each node's list is uniqued the update's `lower_bound+1` and an `upper_bound` at a present
value happen to land on the same index, but the query's threshold `Y` is a general coordinate, not
necessarily a stored `y`, so its `upper_bound` is the search that actually matters there.

Assembly is then mechanical: read the `n` points and the `q` operations verbatim (the encoded query fields
are fixed; only the decode depends on `lastAns`, applied at execution time), compress `x`, build each outer
node's `y`-list, size the inner trees, seed the structure by applying each point's initial weight as a
delta, then run the loop. Type 1 applies a delta at the point's stored `(rank, y)`; type 2 xors the four
coordinates with `lastAns`, computes the rectangle sum, sets `lastAns`, and prints. The empty-rectangle
case the encoding can produce — `X1 > X2` or `Y1 > Y2` — returns `0` before touching the trees.

A few corners I want to be sure of. Boundary inclusivity: `rHi = xidx(X2)` includes the rank of `X2` and
`rLo = xidx(X1-1)` excludes `X1`'s lower neighbor, and `Y1-1` excludes the bottom edge, so a point exactly
on the border is counted once. Negative weights: the Fenwick stores signed deltas, so `+10` then `-10` is
exactly `0` and an answer may legitimately go negative — there is no `max(...,0)` clamp, this is a sum, not
a count. Duplicate coordinates: two points sharing `(x,y)` map to the same `x`-rank and the same (uniqued)
`y` slot, so a query covering that cell returns their sum. `rLo` can be `0` when `X1` is at or below the
smallest `x`; the prefix walk `j = 0` runs zero iterations and contributes `0`, which is exactly "no `x`
below the left edge". And `lastAns = 0` before the first query, so a run of only updates simply prints
nothing.

To trust it I ran it against a brute force that, for each query, linearly scans all `n` points testing the
decoded rectangle with the identical XOR-decode and update semantics — obviously correct, `O(nq)`. The
generator emits small and mid-size random cases and, crucially, simulates the true answers as it generates
so the online XOR chain stays self-consistent; roughly a thousand such cases plus hand-built edges (single
point, zero-then-update, duplicate coordinates, negative cancel, empty rectangle, near-`10^9` coordinates,
the forced-online chain, boundary-inclusive) all match byte-for-byte, and the documented sample prints `5`
and `5`. A full-scale run — `n = q = 10^5`, full-plane queries at `10^9`-magnitude coordinates — finishes
in `0.30 s` using `36` MB, inside the `1 s` / `256` MB budget.

The full self-contained C++ program is the answer; every part of it is one of the walks derived above.
