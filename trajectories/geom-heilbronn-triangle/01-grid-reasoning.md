I want eleven points in the unit square whose smallest triangle is as large as possible, and
before I reach for any search I want a principled starting configuration — something I can write
down in closed form, that scores a real number, and that tells me what the floor looks like. The
whole difficulty of this problem lives in one word: *minimum*. My score is the worst triangle among
all `C(11,3) = 165` triples, and the worst is a hard `min`, not an average. That changes everything
about what "good" means. If I were maximizing the *mean* triangle area I could get away with a
loose, roughly-even scatter, because one thin triangle would be diluted by a hundred fat ones. But
here a single near-collinear trio sinks the whole configuration to almost nothing, and three exactly
collinear points sink it to exactly zero, no matter how beautifully the other `164` triangles are
arranged. So the question for a baseline is not "how do I spread eleven points evenly?" — it is
"what is the most evenly spread arrangement I can *name* in which no three points ever come close to
lying on a common line?" Those are different questions, and conflating them is the trap I have to
avoid.

My first instinct is a regular grid, because a grid feels maximally even: every point on a lattice
site, nothing bunched up, translational symmetry everywhere. Eleven is awkward — it does not factor
into a tidy rectangle — but a `4 × 3` grid has twelve nodes, and I can drop one to get eleven. Let
me write the nodes down concretely: put them at `(i/3, j/2)` for `i ∈ {0,1,2,3}` and `j ∈ {0,1,2}`,
twelve points filling the square on a `1/3 × 1/2` mesh. The moment I have the coordinates in front of
me the problem is obvious. Each *row* of the grid — fix `j`, vary `i` — is four points sharing the
same `y = j/2`, so all four are collinear. Three collinear points span a triangle of area exactly
zero: the cross product `(b_x−a_x)(c_y−a_y) − (c_x−a_x)(b_y−a_y)` vanishes identically when the three
`y`-coordinates agree, because every factor carrying a `(·_y − a_y)` is zero. A single row of four
already contributes `C(4,3) = 4` zero-area triples, and there are three rows (twelve such triples),
four columns of three (each column another `C(3,3) = 1` zero, four more), and even the *diagonals*
line up: `(0,0), (1/3, 1/2), (2/3, 1)` all have slope `3/2` and are collinear, as are their several
translates and the anti-diagonals. Let me actually count the zero-area triples rather than wave at
them. The three rows of four give `3 · C(4,3) = 12`; the four columns of three give `4 · C(3,3) = 4`;
the slope-`+3/2` diagonals carry three points each along `(0,0),(1/3,1/2),(2/3,1)` and along
`(1/3,0),(2/3,1/2),(1,1)`, two lines, `2` more; and the slope-`−3/2` anti-diagonals `(0,1),(1/3,1/2),
(2/3,0)` and `(1/3,1),(2/3,1/2),(1,0)` add another `2`. That is `12 + 4 + 2 + 2 = 20` degenerate
triples before I even chase the steeper coincidences, and the grid has only `C(12,3) = 220` triples
total — so nearly a tenth of them are exactly zero. Dropping one node to get to eleven barely dents
that. Removing a single point touches exactly one row (thinning it from four points to three, which
still contributes `C(3,3) = 1` zero) and one column (thinning it from three to two, killing that
column's lone zero). So even in the best case the deletion clears at most the four zeros of one row
plus the one zero of one column — five of the twenty — and leaves the other two full rows intact
with their `2 · 4 = 8` zeros, plus the thinned row's surviving one, plus every diagonal not through
the deleted node. At least nine zero-area triples survive any single deletion, so the eleven-point
grid still scores exactly `0`.
The very regularity I liked is fatal. Evenness in the lattice sense is *built out of straight lines*,
and straight lines are precisely the thing the `min` objective punishes hardest.

Can I keep the grid's even coverage but break its collinearity? Let me actually try the obvious
repairs before I abandon the idea, because if one works it is the cheapest possible baseline. Shear
the grid — replace `(i/3, j/2)` with `(i/3 + s·j, j/2)` for some skew `s`. But a row is still all the
points with a fixed `j`; they still share `y = j/2`; they are still collinear. Shearing slides rows
sideways, it does not tilt them. Rotate the whole grid by some angle `θ`? Rotation is rigid, so it
carries each straight lattice line to another straight line — the rows become parallel slanted lines,
still perfectly collinear within themselves. Stretch it anisotropically? Same story: affine maps send
lines to lines, and a grid's defining feature is that its points sit on a small number of lines with
many points each. No affine deformation escapes, because collinearity is affine-invariant. The only
way to kill it is to move points *off* their lattice lines by unequal, non-affine amounts — and that
is no longer "naming a grid," that is searching. So the grid family is dead on arrival, and it dies
for an instructive reason I want to carry forward: any configuration whose points lie on a small
number of lines is disqualified, and lattices are the canonical such configuration.

Let me try to keep the spread-out intuition but attack the collinearity from a different direction.
What if I push all eleven points onto the *boundary* of the square, equally spaced around the
perimeter? Now there is no grid, no interior lattice — the points sit on the four edges, as far from
the center as possible, which intuitively gives each triangle a lot of room. But I have to be honest
about what "on the edges" means, and the way to be honest is to actually place the points and look.
The perimeter has length `4`; eleven equally-spaced points sit at arc-length `s_k = 4k/11` for
`k = 0,…,10`, spacing `4/11 ≈ 0.3636`. Walking the perimeter starting at the corner `(0,0)` and going
counterclockwise, the bottom edge is `s ∈ [0,1)`, the right edge `[1,2)`, the top `[2,3)`, the left
`[3,4)`. The first three points land at `s = 0, 4/11, 8/11`, all with `s < 1`, so all three sit on the
bottom edge: `(0,0)`, `(4/11, 0) ≈ (0.364, 0)`, `(8/11, 0) ≈ (0.727, 0)`. Three points on `y = 0` are
exactly collinear, and their triangle has area zero. I do not even have to look further — the score is
already zero. But it is worse than a single accident: continuing the walk, `s = 12/11, 16/11, 20/11`
land on the right edge (`x = 1` at heights `0.091, 0.455, 0.818`), another collinear trio; `s = 24/11,
28/11, 32/11` land on the top edge (`y = 1`), a third collinear trio; and the last two, `s = 36/11,
40/11`, sit on the left edge. So the equally-spaced ring puts three points on each of the bottom,
right, and top edges — three independent zero-area triangles right there. And it is worse than three,
because the corner `(0,0)` at `s = 0` sits on the bottom edge *and* on the left edge line `x = 0`,
where the two left-edge points at `s = 36/11, 40/11` also land: `(0,0)`, `(0, 0.727)`, `(0, 0.364)`
are a fourth collinear trio. That is at least four independent zero-area triples before I count the
further collinearities each corner spawns with its own two edge-neighbours. The boundary ring is not
"nearly degenerate," it is *thoroughly* degenerate. The lesson repeats and sharpens: putting many
points on the same straight edge is the same mistake as putting them on the same grid row.

Before I give up on lattices entirely I should ask whether a *non-square* lattice escapes, because
the triangular (hexagonal) lattice is the one people reach for when they want "even" without the
obvious square grid. But the triangular lattice fails for exactly the same reason, and it fails
worse: it has straight lines in *three* directions instead of two, and every one of those lines
carries several collinear points. Any lattice — by definition a set closed under integer combinations
of two basis vectors — has infinitely many lines each containing infinitely many lattice points, so
any finite patch of it that is dense enough to place eleven points will have three on a line. There is
no lattice, square or triangular or sheared, that avoids collinear triples while staying compact. That
closes the lattice family for good: the property I need is fundamentally *anti*-lattice.

It is worth pausing on why the `min` objective is so unforgiving here, because it tells me what a
baseline must guarantee. The number of triples is `C(11,3) = 165`, and that count grows like `n³/6`,
so as I add points the number of chances to accidentally form a near-collinear sliver grows cubically
while the room to place points grows only linearly in area. A configuration I place by hand without
*specifically* engineering against collinearity is very likely to have some trio nearly on a line —
and one such trio, being the minimum of `165`, sets the whole score. This is why "looks evenly spread"
is not enough: I need a construction that makes the no-collinear-triple guarantee structural, holding
for all `165` triples at once, not one I check and hope survives. That requirement is what a circle
supplies and a scatter does not.

So what I actually need is a configuration with *no three points collinear anywhere*, and among all
such configurations I want one whose points are spread widely enough that the smallest triangle is
decently fat rather than a sliver. I want a closed-form object that *guarantees* positivity, not one
that merely avoids collinearity by luck. The cleanest such object I know is a **circle**: place all
eleven points on a common circle. A line meets a circle in at most two points, so no three points on
a circle are ever collinear — every one of the `165` triangles is strictly positive by construction,
and the degeneracy that killed the grid and the boundary ring simply cannot occur. This is not a soft
"probably fine," it is a hard geometric guarantee, which is exactly what I want from a baseline I plan
to trust as a floor.

Before I commit to the circle let me at least consider the other continuous "spread" I know, a
phyllotaxis spiral — points at the golden angle, `θ_k = 2πk·φ` with radii growing like `√k`, the
sunflower packing. It spreads points beautifully and evenly across a disk, better *filling* than a
single ring. But it gives me no guarantee against collinearity: three spiral points can easily fall
near a common line, and I would only find out by scoring it. For a *baseline* — the one number I want
to be able to write down and trust without search — a probabilistic "usually no bad slivers" is worth
less than the circle's certainty. The spiral is a fine idea for a richer configuration, but it does
not earn its complexity here, so I set it aside and take the circle.

Now, which circle, and how does the score come out? I place the eleven points equally spaced on the
circle, a regular eleven-gon, at angles `θ_k = 2πk/11`. The smallest triangles on a regular polygon
are the "thinnest" ones — three points that subtend the narrowest total arc — and those are three
*consecutive* vertices. I can compute their area in closed form rather than guess it. For three points
on a circle of radius `R` at angles `θ₁, θ₂, θ₃`, the signed area is
`½R²[sin(θ₂−θ₁) + sin(θ₃−θ₂) + sin(θ₁−θ₃)]`. (I can sanity-check this identity on a case I know: an
equilateral triangle inscribed in the unit circle, angles `0, 120°, 240°`, gives
`½·1·[sin120° + sin120° + sin(−240°)] = ½·(3·0.8660) = 1.299`, and the direct equilateral-triangle
area for circumradius `1` is `3√3/4 = 1.299` — they agree, so the identity is right.) For three
consecutive vertices of the eleven-gon put them at `0, α, 2α` with `α = 2π/11 ≈ 0.5712`. Then the
bracket is `sin α + sin α + sin(−2α) = 2 sin α − sin 2α = 2 sin α − 2 sin α cos α = 2 sin α (1 − cos α)`,
so the consecutive-triple area is `R²·sin α·(1 − cos α)`. With `α = 2π/11`: `sin α ≈ 0.54064`,
`cos α ≈ 0.84125`, `1 − cos α ≈ 0.15875`, so the area is `R²·0.54064·0.15875 ≈ R²·0.085827`. I should
confirm consecutive vertices really are the thinnest and not some other triple, and I can do it
cleanly by parametrizing every triple by its three angular gaps. Any triple of vertices splits the
circle into three arcs of sizes `g₁α, g₂α, g₃α` with `g₁ + g₂ + g₃ = 11` and each `gᵢ ≥ 1`, and the
inscribed-triangle area is `½R²(sin g₁α + sin g₂α + sin g₃α)`. To make this small I want the gaps as
*unequal* as possible, pushing one gap toward the whole circle where its sine is small; the extreme
admissible partition is `(1, 1, 9)`, which is exactly three consecutive vertices. Its bracket is
`sin α + sin α + sin 9α = 2 sin(2π/11) + sin(18π/11) = 2(0.5406) + (−0.9096) = 0.1717`, giving area
`½R²·0.1717 = R²·0.08585` — matching the closed form above. Now let me walk the partition outward one
step at a time and watch the bracket climb, because I want to know not just that `(1,1,9)` is smallest
but whether it is *isolated* — whether the eleven consecutive triangles are alone at the floor with
everything else safely above. `(1, 2, 8)`: `sin α + sin 2α + sin 8α ≈ 0.5406 + 0.9096 − 0.9898 =
0.4605`, area `R²·0.2302` — already `2.68×` the `(1,1,9)` floor. `(1, 3, 7)`: `0.5406 + 0.9898 −
0.7557 = 0.7747`, area `R²·0.3874`, `4.5×`. `(2, 2, 7)`: `0.9096 + 0.9096 − 0.7557 = 1.0635`, area
`R²·0.5318`, `6.2×`. The bracket rises steeply the instant I move even one vertex off the tightest
packing, so the minimum is not a shallow valley among near-ties — it is a sharp, isolated floor with
the second-thinnest triple already `2.68×` fatter. So the consecutive triples set the minimum, and
any wider-spanning triple is fatter by a wide margin. Good: the score is governed by `R²·0.085827`,
and it is set cleanly by a single family of eleven equal triangles with real air between them and
everything else. And I can count how many triangles achieve it: a consecutive triple is
determined by its middle vertex together with that vertex's two immediate neighbors, so there are
exactly eleven of them, one per vertex, all equal by rotational symmetry. So eleven of the `165`
triangles tie at the minimum. That is a first, faint version of a pattern I expect to matter later —
that a configuration which is symmetric enough has many triangles pinned at the same value — though
here the symmetry is the *wrong* one and the shared value is only `0.0215`, not the record. It is also
worth seeing how far below "typical" that minimum sits, because it drives home why the `min` is the
enemy. The *largest* triangle inscribed in a circle is the equilateral one, area `(3√3/4)R² ≈
1.299·0.25 ≈ 0.325`; the eleven-gon cannot form a perfect equilateral (11 is not divisible by 3) but
comes close with widely-spaced vertices, so its fattest triangles are around `0.3`. Its thinnest are
`0.0215`. That is a spread of roughly fifteen-to-one *within a single configuration* — the objective
throws away the fourteen-fifteenths of area sitting in the fat triangles and reports only the sliver.
Every rung after this one is a fight to lift that sliver, and nothing else.

I should also check that I am not leaving an obvious improvement on the table within the circle family
itself, because if a two-parameter variant clearly beats the plain eleven-gon then "baseline" should
mean that variant. Two concentric rings — say eight points on an outer ring and three on an inner
one — would let the inner points break up the outer ring's near-adjacent triples. But now I have a
radius ratio and two angular offsets to choose, and choosing them to maximize the minimum *is*
optimization, not a parameter-free name; worse, three points on a small inner ring form tiny triangles
*among themselves*, and I can size the damage. Put three points on an inner ring of radius `r`; their
own thinnest triangle is a scaled copy of the three-consecutive story, of order `r²·0.4` for an
equilateral-ish inner trio (bracket `3√3/4 ≈ 1.3`, area factor `~0.65`, but I only have three points
so it is their single triangle at `~0.65 r²`... call it order `r²`). For that inner triangle merely to
*match* the outer ring's floor `R²·0.086 = 0.0215`, I already need `r² ≳ 0.033`, i.e. `r ≳ 0.18` —
more than a third of the outer radius `R = 1/2`. But an inner ring at `r ≈ 0.18` is not "inner" enough
to break the outer octagon's near-adjacent triples, which is the only reason I wanted it. So the
inner ring is trapped between two failures: small enough to disrupt the outer adjacencies means small
enough that its own trio is the new sliver; large enough to be safe means large enough to stop
helping. There is no free lunch here without *tuning* the radius, and tuning is search. The same
objection kills a "perturbed
eleven-gon" with per-vertex radial jitter: the jitter amplitudes are free parameters and setting them
well is search. A parameter-free baseline forbids all of this by definition, and that is the point —
I want the floor to be a single object I can write down with no knobs, so that every later rung's
improvement is unambiguously attributable to *searching* rather than to my having quietly tuned the
baseline. That leaves the plain inscribed eleven-gon as the honest floor.

That leaves the radius, and here the reasoning is a one-line scaling argument. Every triangle area
scales as `R²` under uniform scaling about the center, so to maximize the minimum I want the *largest*
circle that still fits inside the square. That is the inscribed circle, `R = 1/2`, centered at
`(1/2, 1/2)`; any smaller `R` shrinks every triangle by the factor `R²`, and any larger `R` pushes
points outside `[0,1]²` and violates the constraint. So the inscribed regular eleven-gon is forced —
largest legal circle, equal spacing, one parameter-free object — and it needs nothing tuned. With
`R = 1/2`, `R² = 1/4`, the minimum triangle area comes out to `0.085827/4 ≈ 0.021456`. Against the
record `1/27 = 0.037037` that is a fraction `0.021456/0.037037 ≈ 0.579`. So my honest prediction,
computed rather than hoped, is that the inscribed eleven-gon scores about `0.0215`, a little under
sixty percent of the record, and that the grid and the equally-spaced boundary ring both score exactly
`0` — the evaluator's number for the circle should land right on my closed-form `0.02146` if I have
made no error, and the two lattice-like configurations should read `0.000000`.

I should set expectations for what this *cannot* be. It is a baseline, and a circle is nowhere near
the known optimum. The record configuration at `n = 11` is nothing like a circle — it is a structured,
irregular arrangement with points on the square's boundary and interior, deliberately laid out so the
worst triangle is fattened toward `1/27`. A regular eleven-gon does not do that: it spends all of its
symmetry on *rotational* regularity, and rotational regularity is not the same as maximizing the
minimum triangle. The eleven-gon makes its consecutive-vertex triples as thin as `R²·0.0858` because
those triples subtend a narrow arc, and nothing in the construction pushes back on that thinness — the
symmetry that makes the polygon pretty is exactly the symmetry that leaves its worst triangles
thinner than they need to be. There is no closed form I can write down that reproduces the irregular
record; the whole distance from `0.0215` to `1/27` has to be bought by *moving points around* to
enlarge the smallest triangle, and no amount of cleverness in naming a symmetric shape will get there.

One more piece of orientation keeps me from over-reaching. The asymptotic side of this problem
conjectures that the best achievable minimum area decays like `1/n²` for large `n`, which at `n = 11`
would suggest a scale around `1/121 ≈ 0.008` — yet the actual record is `1/27 ≈ 0.037`, several times
larger. That gap is not a contradiction; it is the signature that `n = 11` sits firmly in the
*small-`n`* regime, where the answer is a specific hand-built configuration with a favorable constant,
not a manifestation of the asymptotic decay law. There is no formula I can evaluate to produce it, and
no reason the asymptotically-motivated arrangements would be near-optimal at eleven points. This is
the same conclusion the failed grid reached me by a different road: at this `n` the record is an
individual, irregular, boundary-heavy configuration, and the only general-purpose way to approach an
individual configuration I cannot name is to search for it. My baseline's job was only to establish a
legal, positive, principled floor and to prove that the naming-a-shape approach tops out around
half the record; both are done.

That is the sharp takeaway I carry to the next rung. The structured baselines split cleanly into two
families: the lattice-like ones (grid, boundary ring) that score zero because they are built out of
straight lines, and the circular one that scores a modest positive value because it forbids collinear
triples but wastes its degrees of freedom on the wrong symmetry. Neither of them *searches*. The
record is a hand-tuned irregular configuration with no closed form within my reach, so the entire gap
must be earned by actually trying configurations and letting the evaluator pick the best. And the
very first, dumbest way to do that — the thing that needs no gradient, no temperature, no schedule,
nothing but a sample count — is simply to draw a great many random configurations and keep the best
one. That is the next rung: replace "name one clever configuration" with "try a great many and let
the evaluator choose," and measure how far brute-force randomness alone climbs off this `0.0215`
floor. I expect it to reveal, quickly, both what the raw landscape offers for free and why raw
sampling alone will not be enough.
