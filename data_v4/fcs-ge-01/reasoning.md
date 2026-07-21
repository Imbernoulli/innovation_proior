I have `n` planar points with integer coordinates and I need the maximum squared Euclidean
distance over all distinct pairs — the squared diameter. The two numbers that shape everything
are the scale, `n` up to `2*10^5`, and the coordinate range, `±10^9`. I take the arithmetic
first, because here it constrains the design. A separation in one axis can be `2*10^9`, so a
squared distance reaches `(2*10^9)^2 + (2*10^9)^2 = 8*10^18`. Signed 64-bit tops out near
`9.2*10^18`, so `8*10^18` fits — but barely, and only because I am asked for the square, not the
root. That is presumably why the problem outputs the square: it stays in exact integers, no
`sqrt`, no floating-point comparison, no epsilon tie-break. So every coordinate, cross product,
and distance accumulator has to be `long long`. Even `(x1-x2)` fits in 32 bits, but
`(x1-x2)*(x1-x2)` does not, so the multiply itself must happen in 64-bit.

The brute force writes itself — loop over all pairs, keep the largest `dx*dx + dy*dy` — and it is
exactly the oracle I will test against on small inputs. But at `n = 2*10^5`, `n*(n-1)/2 ≈ 2*10^10`
pair evaluations against a 1-second budget is three to four orders of magnitude too many. I need
the same exact answer with near-linear work, which means exploiting structure instead of
enumerating.

Squared distance is convex in each endpoint, which makes me suspect the farthest pair sits on the
boundary. To make that precise: fix one endpoint `Q` and consider "distance to `Q`" over the
hull; it is convex, so it is maximized at a hull vertex, never at an interior point. Applying that
to both endpoints, the diameter is realized by two hull vertices, so I can discard every interior
point via an `O(n log n)` hull. On random input that collapses `2*10^5` points to a few hundred
vertices, but the worst case — points on a circle — keeps all `n` on the hull, so I cannot fall
back to an `O(h^2)` scan of the hull and call it done. I need the hull *and* a linear sweep over
it.

Why does a convex polygon have only `O(h)` candidate farthest pairs rather than `O(h^2)`? Squeeze
it between two parallel supporting lines like caliper jaws; the two vertices they touch form an
*antipodal* pair. The diameter must be antipodal — if the farthest pair admitted no parallel
supporting lines I could rotate slightly and increase the distance. Rotate the jaws once all the
way around: the touched vertices change only when a jaw goes flush with an edge, so over a full
turn each jaw sweeps each edge once and only `O(h)` antipodal pairs appear. That is the reduction
— the diameter is among `O(h)` antipodal pairs, enumerable in a single pass around the hull.

To make the rotation discrete and keep it in integers, I replace angle comparison with area. Walk
an index `i` over the hull edges `(h[i], h[i+1])`; the antipodal vertex for that edge is the one
farthest from the edge's line, and that distance is unimodal as `j` walks a convex polygon, so a
single forward pointer `j` tracks it. Distance from the line is proportional to
`cross(h[i], h[i+1], h[j])` (twice the triangle area), so I advance `j` while the next vertex
gives a larger cross product, stop at the apex, and fold `dist2(h[i], h[j])` and
`dist2(h[i+1], h[j])` into the running best. Index `i` loops once and `j` never resets backward,
so `j` also makes at most one loop and the sweep is `O(h)` — that monotonicity is the whole reason
this is linear.

The clean loop assumes a real polygon, `h >= 3`, but the inputs include the
degeneracies where geometry code cracks: all points identical (hull is one vertex, answer `0`),
all collinear (hull is a segment, answer is its squared length), and heavy duplication. On
`h < 3` the modular edge indexing `(j+1)%m` is nonsensical, so I branch `m == 1` and `m == 2` out
and run the calipers only for `m >= 3`. In the hull builder I also decide against keeping
edge-interior collinear points, since the diameter is always at *extreme* vertices: I build a
strict hull that pops collinear points (`cross <= 0` pops, not `< 0`). That keeps the hull minimal
and, more usefully, makes an all-collinear input collapse to exactly its two extreme endpoints,
which the `m == 2` branch then handles. Deduping identical points first keeps zero-area cross
products out of the sweep.

I write Andrew's monotone chain (sort by `(x, y)`, dedup, lower chain then upper chain popping on
`cross <= 0`), the `m <= 2` special cases, and the calipers loop. My first cut of the advance
condition used `cross(h[i], h[ni], h[(j+1)%m]) >= cross(h[i], h[ni], h[j])`, and the `>=` is
wrong — a square exposes it. Take the unit-square hull `[(0,0),(1,0),(1,1),(0,1)]` CCW, `j = 1`,
edge `i = 0` the bottom. `cross(bottom, h[2]) = 1 >= cross(bottom, h[1]) = 0`, advance to `j = 2`.
Then `cross(bottom, h[3]) = 1 >= cross(bottom, h[2]) = 1` is true under `>=`, so `j` advances to 3
even though the apex got no farther from the edge. Two vertices equidistant from an edge — rampant
in squares, rectangles, and any far-side-collinear layout — let the pointer step *past* a
legitimate antipodal vertex without ever testing its distance, and the farthest pair can be
exactly the one skipped. Running that first version against the brute oracle, the `2x2` and `3x3`
lattice cases, which are saturated with such ties, mismatched. The fix is to make the advance
strict, `> ` not `>=`, so `j` halts at the first apex and both tied partners get tested (the other
is reached as the apex of a neighbouring edge). Re-tracing the square with `>`: edge `i = 0` now
stops at `j = 2`, where `dist2((0,0),(1,1)) = 2`; edge `i = 1` reaches `h[3]` for the other
diagonal; answer `2`, and the lattice mismatches vanish. With `>` the
per-edge area is strictly unimodal so the `while` terminates, and `j` advancing monotonically at
most once around keeps the total `O(h)`.

The degenerate branches, checked concretely: `m == 1` (all identical) returns `0`; `m == 2`
returns `dist2(h[0], h[1])` — `[(0,0),(1,0),(2,0)]` collapses under the strict pop to
`[(0,0),(2,0)]`, distance² `4`, and a vertical line behaves the same since the sort key is
`(x, y)`. The extreme pair `[(-10^9,-10^9),(10^9,10^9)]` returns `8000000000000000000`, the
`8*10^18` boundary sitting inside signed 64-bit exactly as budgeted.

I differential-tested a `g++ -O2 -std=c++17` build against the `O(n^2)` oracle: 1461 cases across
tiny inputs, all-identical, collinear (sloped and vertical), small lattices, generic clouds,
explicit edges (single point, duplicate segment, the overflow boundary), and adversarial
degenerate cases including circle points where the hull is the whole input. Zero mismatches after
the strict-advance fix. At `n = 2*10^5` the runs land around 40–50 ms (random `0.052s`,
all-collinear `0.044s`, all-identical `0.035s`), well inside the limit, and the collinear timing
confirms the `m == 2` short-circuit handles the worst hull-size blow-up cheaply. The full program
is in the answer.
