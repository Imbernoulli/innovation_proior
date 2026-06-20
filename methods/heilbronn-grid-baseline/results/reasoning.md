I want eleven points in the unit square whose smallest triangle is as large as possible, and
before I reach for any search I want a principled starting configuration — something I can write
down in closed form, that scores a real number, and that tells me what the floor looks like. The
whole difficulty of this problem is the *minimum*: my score is the worst triangle among all `165`
triples, so a single near-collinear trio sinks the whole configuration to almost nothing. So the
question for a baseline is really "what is the most evenly spread, least-collinear arrangement of
eleven points I can name without searching?"

My first instinct is a regular grid. Eleven is awkward — it does not factor into a tidy rectangle
— but a `4 × 3` grid has twelve nodes, and I can drop one to get eleven. A grid feels maximally
even, every point at a lattice site, nothing bunched up. But the moment I picture it I see the
problem: a grid is *built* out of collinear points. Every row of the grid is three or four points
on a perfectly straight line, and any three collinear points form a triangle of area exactly zero.
So a grid does not merely score low — it scores *zero*, the worst possible value, because the
minimum is dragged all the way down by the degenerate triples sitting on each grid line. The very
regularity I liked is fatal. A grid is the canonical example of a configuration full of zero-area
triangles. So that idea is dead on arrival, and it dies for an instructive reason: evenness in the
sense of "lattice-regular" is exactly the wrong kind of evenness here, because lattices are made
of straight lines.

Let me try to keep the spread-out intuition but kill the collinearity. What if I push all eleven
points onto the boundary of the square, equally spaced around the perimeter? Now no row of a grid
exists; the points sit on the four edges. But I have to be honest about what "on the edges" means:
each edge of the square is itself a straight line, and if two or more of my equally-spaced
perimeter points land on the same edge — which they must, since eleven points around four edges
puts two or three points on most edges — then those edge-mates are collinear with each other and
with any third point that also happens to lie on that line. Worse, points on the same edge plus a
corner are collinear. So the boundary ring, too, is riddled with straight-line triples, and I
expect it to collapse to zero or near-zero. The lesson repeats: any arrangement that puts three
points on a common straight line is disqualified, and both the grid and the equally-spaced
boundary do exactly that.

So what I actually need is a configuration with *no three points collinear anywhere*, and ideally
one where the points are spread so the smallest triangle is decently fat. The cleanest such object
I know is points on a **circle**. Place all eleven points on a common circle, equally spaced — a
regular eleven-gon — and inscribe that circle in the square so every point stays inside. On a
circle no three distinct points are ever collinear (a line meets a circle in at most two points),
so every one of the `165` triangles has strictly positive area; the degeneracy that killed the
grid and the boundary ring simply cannot happen. And the regular spacing makes the configuration
as symmetric as possible, which should keep the smallest triangle away from being a pathological
sliver. The smallest triangles on a regular polygon are the "thinnest" ones — three points that
are nearly adjacent around the rim, subtending a narrow arc — so the score will be set by those
near-adjacent triples, not zero, but a modest positive number.

How big a circle? I want the points spread as widely as the square allows, so I take the largest
circle that fits — the inscribed circle, radius `1/2`, centered at the square's center. That uses
the full extent of the box; a smaller circle would only shrink every triangle proportionally, and
a larger one would push points outside the square and violate the constraint. So the inscribed
regular eleven-gon is the natural parameter-free choice: largest circle, equal spacing, all points
legal, no collinear triples.

I should set expectations honestly. This is a *baseline*, and a circle is far from the known
optimum. The record at `n = 11` is `1/27 ≈ 0.037`, and that record configuration is nothing like a
circle — it is a structured arrangement with points on the boundary and interior of the square,
deliberately laid out so the worst triangle is fattened. A regular eleven-gon does not do that: it
spends its symmetry on rotational regularity, not on maximizing the *minimum* triangle, so its
smallest triangles (the near-adjacent rim triples) are thinner than they need to be. I expect the
inscribed eleven-gon to land somewhere around half of the record — a real, positive, principled
number that proves the configuration is non-degenerate and gives every later rung something
concrete to beat, but nowhere near `1/27`.

What I take away for the next rung is sharp. The structured baselines split cleanly into two
families: the lattice-like ones (grid, boundary) that score zero because they are built from
straight lines, and the circular one that scores a modest positive value because it has no
collinear triples but wastes its degrees of freedom on the wrong symmetry. Neither *searches*. The
record is a hand-tuned irregular configuration, and there is no closed form I can write down that
will reproduce it. So the entire distance from the circle's value to `1/27` has to be bought by
actually moving points around to enlarge the smallest triangle — and the very first, dumbest way
to do that, the thing that needs no gradient and no schedule, is simply to *sample many random
configurations and keep the best one*. That is the next rung: replace "name one clever
configuration" with "try a great many configurations and let the evaluator pick," and see how far
brute-force randomness alone climbs off the baseline.
