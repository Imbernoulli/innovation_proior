I want a packing of `26` circles in the unit square with the sum of radii as large as possible,
and before I reach for any optimizer I want a baseline that is principled, guaranteed feasible,
and parameter-free — something that puts a concrete number on the board so every later, searched
rung has a floor to beat. The most natural such object is a regular grid of equal circles, and the
reason to start there is not laziness: a grid is the one arrangement whose value I can write down in
closed form and defend on paper, so it lets me understand the *objective* before I understand the
landscape.

Let me reason first about what makes this objective unusual, because it decides everything that
follows. In ordinary equal-circle packing the quantity being maximized is the single common radius
`r`, and there the winning move is to use *few* large circles — one circle fills the square with
`r = 0.5`, two do worse, and every extra circle you demand forces the common radius down. Here the
score is instead `Σ rᵢ`, a *sum*, and that flips the preference completely. To see the flip
concretely, put `n = k²` equal circles on a `k × k` grid, one per cell. Each cell has width `1/k`,
and the largest circle that stays inside its cell and just touches its four neighbours is the one
inscribed in the cell, radius `r = 1/(2k)`. The grid then holds `k²` circles of radius `1/(2k)`, so

    Σ rᵢ = k² · 1/(2k) = k/2.

For `k = 1` that sum is `0.5` — the single big circle that would *win* the equal-radius game — and
for `k = 5` it is `2.5`, five times larger. The mechanism is exactly the thing the sum rewards:
halving each radius while quadrupling the count multiplies the sum by two. So under a sum objective,
many small circles is not a weakness to be tolerated; it is the whole game, and a grid exposes that
immediately and cleanly. That single observation — that `Σ rᵢ` is monotone increasing in the grid
fineness `k` — is what tells me the good packings will be crowded with small circles rather than
dominated by a few large ones, and it is worth having on paper before I let any solver loose.

There is a ceiling on how far "more small circles" can go, and I want to know it now so I can read
every later number against it. The circles are disjoint and live inside a region of area `1`, so
their total area is at most `1`: `Σ π rᵢ² ≤ 1`, i.e. `Σ rᵢ² ≤ 1/π ≈ 0.3183`. By Cauchy–Schwarz,
`(Σ rᵢ)² ≤ 26 · Σ rᵢ² ≤ 26/π`, so

    Σ rᵢ ≤ √(26/π) ≈ 2.8768.

That is an absolute, first-principles ceiling for `n = 26`: no feasible packing, grid or otherwise,
can exceed about `2.877`. The frontier quoted in the background sits near `2.636`, which is `≈ 91.6%`
of this area bound — the gap between them is the unavoidable slack from the fact that disks cannot
tile the plane without leaving interstitial gaps. So the achievable target lives in a narrow band
just below `2.64`, well under the naive `2.877`, and my grid floor will land somewhere below even
that. Having the ceiling in hand keeps me honest about how much room each rung is actually buying.

Now I have to hit exactly `26` circles, and `26` is not a perfect square, so the clean `k/2` formula
does not apply directly and I have to choose the grid deliberately. The square grids that fit inside
a budget of `26` circles are `k = 1..5`, with counts `1, 4, 9, 16, 25` and sums `k/2 = 0.5, 1.0,
1.5, 2.0, 2.5`; `k = 6` needs `36 > 26` and does not fit. Since `k/2` is monotone increasing, the
largest grid that fits, `5 × 5`, is the best of the fitting grids outright, and it happens to leave
exactly one circle over — a tidy coincidence with `26 = 25 + 1`. Let me still confront the two
tempting escapes from `5 × 5` before I commit to it.

The neighbouring squares are `25 = 5²` and `36 = 6²`. A `5 × 5` grid gives `25` circles with
`r = 1/10 = 0.1` and sum `k/2 = 2.5`.
A `6 × 6` grid would give `36` circles with sum `3.0` — a larger `k/2`, which is tempting until I ask
whether I can actually realise it with only `26` circles. I cannot use all `36`; I am allowed exactly
`26`. If I keep `26` of the `36` grid positions and drop the other `10`, the kept circles still sit on
the `1/6`-spaced lattice with radius `1/12 ≈ 0.0833`, and their sum is `26 · 1/12 ≈ 2.1667` — *below*
the `5 × 5` value of `2.5`. Dropping circles from the finer grid throws away the very count that made
`k/2` large, and the survivors cannot grow to fill the vacated cells because they are still pinned by
their remaining neighbours and by the walls. So the `6 × 6` route, despite its higher full-grid `k/2`,
gives `2.167` at `26` circles and is strictly worse. Symmetrically, I cannot improve the `5 × 5` grid
by *removing* any of its `25` circles — removal only lowers the sum.

The second escape is more subtle and worth pricing out, because it is the closest thing the grid
family has to the irregular, multi-scale packings the frontier uses. Instead of one fine grid, use a
*coarse* grid and fill every one of its holes: a `4 × 4` grid of `16` circles at `r = 0.125` (sum
`2.0`), then drop a filler into each of its `3 × 3 = 9` interior interstices. Each such filler has
radius `(√2 − 1)·0.125 ≈ 0.0518`, so the nine of them add `9 · 0.0518 ≈ 0.466`, and the two-scale
packing reaches `2.0 + 0.466 ≈ 2.466` with `25` circles — already *below* the `5 × 5` grid's `2.5`
before I have even placed the `26`th. The reason it loses is quantitative and instructive: the
coarse grid's `16` circles are individually larger (`0.125` vs `0.1`) but there are too few of them,
and the interstitial fillers it opens up are small (`0.052`) and only nine in number, so the total
count-times-size product falls short of the fine grid's `25` uniform `0.1` circles. The uniform fine
grid wins among *structured* options precisely because the sum objective rewards count, and the fine
grid maximises count at a still-respectable per-circle radius. So the grid is forced: `5 × 5` full,
`25` circles of radius `0.1` contributing `2.5`, with exactly one extra circle left to place. (I
note the moral already: even the best two-scale *structured* packing underperforms the fine grid, so
whatever beats `2.5` substantially will not be structured at all — it will be irregular, which no
grid can produce.)

Those `25` circles are worth checking as a feasible object in their own right before I add the
`26`th. Their centres sit at `(i + 0.5)/5` for `i = 0..4`, i.e. at `0.1, 0.3, 0.5, 0.7, 0.9` in each
axis. Two horizontally adjacent centres are `0.2` apart and each circle has radius `0.1`, so the sum
of radii `0.2` exactly equals the centre distance — they are tangent, not overlapping. The outermost
centres are at `0.1` and `0.9`, one radius from the walls at `0` and `1`, so every boundary circle is
tangent to the wall. The whole grid therefore sits exactly on the feasibility boundary: no pair
overlaps, nothing pokes out of the square, and the only "violations" are the floating-point residues
of computing `0.1 + 0.1` versus `0.2`. That is precisely the kind of packing the tolerance is
designed to accept, so `2.5` is genuinely on the board.

Where does the `26`th circle go? It has to fit into whatever room the tiled grid leaves, and there
are three qualitatively different kinds of gap, so let me size each one rather than guess. The
cleanest is an *interior interstitial* gap. Four mutually adjacent grid circles sit on the corners
of a little `2r × 2r = 0.2 × 0.2` square; the centre of that square is equidistant from all four, at
half the diagonal, `√2 · r = √2 · 0.1 ≈ 0.1414`. A circle placed there can grow until it touches
those four, so its radius is the centre distance minus the neighbour radius, `√2·r − r = (√2 − 1)·r ≈
0.0414`. There are `4 × 4 = 16` such interior interstices in a `5 × 5` grid, all identical by the
lattice symmetry, so I have a free choice among them. The second kind is a *wall-edge* gap: a circle
tangent to the left wall and squeezed between two vertically adjacent grid circles at `(0.1, 0.1)`
and `(0.1, 0.3)`. By symmetry its centre is at `(r, 0.2)`; tangency to the grid circle at `(0.1, 0.1)`
requires `√((r − 0.1)² + 0.1²) = r + 0.1`, which squares to `(r − 0.1)² + 0.01 = (r + 0.1)²`, i.e.
`0.01 = 0.4 r`, giving `r = 0.025`. The third kind is a *corner* gap: a circle in the corner of the
square touching both walls, centre `(r, r)`, nearest grid circle at `(0.1, 0.1)`. Tangency needs
`√2 · (0.1 − r) = r + 0.1`, which solves to `r = (3 − 2√2)·0.1 ≈ 0.0172`. So the three options are
`0.0172` (corner), `0.025` (wall-edge), and `0.0414` (interior interstitial), and the interior gap
is the clear winner — about `2.4×` the corner circle and `1.65×` the wall-edge circle. It is worth
noticing *why* the ranking comes out this way: the corner and wall circles are penalised by having a
wall press in on them from one or two sides, which forces their centre near the boundary and leaves
less room before they hit the grid; the interior gap is symmetric on all four sides and so admits the
largest inscribed disk. There is an irony worth registering here, because it is the grid's waste
stated as a theorem: in a genuinely good sum-of-radii packing the *corner* is the prize location — a
circle there has two free walls and can grow large — yet in my grid the corner is already spent on an
ordinary `0.1` circle, so the only corner room *left over* is the `0.0172` sliver, the smallest of the
three. The grid has pre-committed its best real estate to a mediocre circle and left me scraps.

Priced as finished packings rather than loose radii, the three placements for the twenty-sixth circle
land at `2.5 + 0.0172 = 2.5172` (corner), `2.5 + 0.025 = 2.525` (wall-edge), and
`2.5 + 0.0414 = 2.5414` (interior interstice), so preferring the interior gap over the corner buys
`+0.0242` on the score and over the wall-edge `+0.0164` — a visible slice of the whole `0.0946` I am
trying to close, and all of it riding on where I drop a single circle, which is why the placement is
worth solving exactly rather than eyeballing. The corner value is the one worth deriving in full,
because it is the least obvious of the three. A corner circle sits at `(r, r)`, tangent to both near
walls, and its nearest grid circle is at `(0.1, 0.1)`; tangency to that neighbour reads
`√2·(0.1 − r) = r + 0.1`. Expanding, `0.1√2 − √2 r = r + 0.1`; collecting the `r` terms,
`0.1(√2 − 1) = r(√2 + 1)`; and rationalising, `r = 0.1(√2 − 1)/(√2 + 1) = 0.1(√2 − 1)² =
0.1(3 − 2√2) ≈ 0.01716`. The wall-edge value `0.025` and the interstitial `(√2 − 1)·0.1` fall out of
the same style of squared-distance equation, so all three are exact rather than fitted, and the
ordering `corner < wall-edge < interior` is a fact forced by the geometry: the more walls press in on
a gap, the smaller the disk it admits. I
drop the `26`th circle into one interior interstice, and the total is

    Σ rᵢ = 2.5 + (√2 − 1)·0.1 = 2.5414213562…

Before I trust that number let me check it is optimal within the one honest degree of freedom this
family still has: the grid radius. Suppose I shrink all `25` grid circles to a common radius `r ≤ 0.1`
while keeping their centres fixed on the `0.2`-spaced lattice. The interstice centre stays `0.1414`
from its four neighbours, so the filler that fits there grows to `0.1414 − r` as the grid circles
recede, and the total becomes `25 r + (0.1414 − r) = 24 r + 0.1414`. That is strictly increasing in
`r` (slope `+24`), so the sum is maximised by pushing `r` to its largest feasible value, `r = 0.1`,
the full inscribed grid — which returns exactly `2.4 + 0.1414 = 2.5414`. So there is no free lunch in
loosening the grid to open a bigger hole: the `24` circles I would shrink cost far more than the one
filler I would grow. The closed form is the best this family offers, and I did not have to guess it.

Let me trace the feasibility of that filler explicitly rather than assert it. Place it at the
interstice centre `(0.4, 0.4)`, the gap of the block with grid circles at `(0.3, 0.3)`, `(0.5, 0.3)`,
`(0.3, 0.5)`, `(0.5, 0.5)`. The distance from `(0.4, 0.4)` to `(0.3, 0.3)` is `√(0.1² + 0.1²) =
√0.02 ≈ 0.14142`, and the sum of the two radii is `0.0414 + 0.1 = 0.14142` — equal, so tangent, no
overlap. The same holds for the other three neighbours by symmetry. Its distance to the walls is
`0.4`, far larger than its radius `0.0414`, so it is safely interior. The whole `26`-circle
arrangement therefore violates no constraint beyond the floating-point floor, which is what I need
from a baseline: not merely feasible in spirit but feasible when the checker actually computes the
distances.

It is worth being explicit about where this packing sits relative to the tolerance, because that is
the currency later rungs will spend. Every contact here is an *exact* tangency by construction:
neighbour distance minus radius sum is zero in real arithmetic, and in floating point it is a residue
of order `10⁻¹⁶`, roughly `10⁹` times inside the accepted `atol = 10⁻⁷`. The grid therefore lives
deep in the feasible interior in tolerance terms even though it is geometrically pressed to the
boundary — there is no sliver of overlap I am relying on the checker to forgive. That distinction
matters: a packing can be "on the boundary" geometrically (tangent) yet carry essentially zero
constraint violation, and that is the honest, robust kind of feasibility. When later rungs push radii
outward to grow the sum, they will begin to consume that tolerance headroom, and I want the floor to
be established with none of it spent, so that any climb above `2.5414` is a real geometric gain and
not a tolerance artefact.

One dimensional sanity check, because a closed form that scales wrong is a closed form with a bug. If
I rescale the square to side `L`, the grid radius becomes `r = L/(2k)` and the sum `k² · L/(2k) =
kL/2`, linear in `L`; the interstitial filler `(√2 − 1)·r` also scales linearly with `L`. Radii carry
units of length, and a sum of `26` radii is still a length, so `Σ rᵢ ∝ L` is exactly the right
homogeneity. The unit-square numbers are the `L = 1` case, and nothing in the derivation hides a
stray square or square-root of `L`. The arithmetic is self-consistent.

Now the honest appraisal of where this lands, because the point of starting here is to measure the
floor precisely. The grid gives `≈ 2.5414`, the frontier band sits near `2.636`, and the absolute
area ceiling is `2.877`. So the grid leaves roughly `0.0946` on the table relative to the frontier —
and to feel how large that gap is, compare it to the *width* of the frontier band itself: the quoted
values `2.63586`, `2.635983`, `2.635988` span only about `1.3 × 10⁻⁴`, so the distance from my floor
to the frontier is on the order of `700×` the spread among the best known packings. That is an
enormous gap in a problem whose whole competitive action happens in the sixth decimal place, and it
is exactly the gap I expect from a rigid symmetric layout. I can name the three ways the grid wastes
that `0.0946`. First, the equal-radius constraint: every circle is locked to `0.1`, so the objective
cannot make the trade it obviously wants — grow a few circles past `0.1` at the cost of shrinking
others below it, which under a *sum* can be net positive because the sum does not care about equality.
Second, the corners and edges: a circle pressed into a corner has two free walls and can grow much
larger than an interior circle before it collides with anything, yet the grid puts an ordinary
`0.1` circle there and squanders the room. Third, the lattice rigidity: a genuinely good
sum-of-radii packing is *irregular*, mixing a few large circles with many unequal gap-fillers whose
sizes are tuned to the local geometry, and a single-lattice, single-radius grid has no vocabulary for
irregularity at all.

I can locate that waste in the area budget precisely, which turns the vague word "wasteful" into a
number. The grid covers area `25 · π · 0.1² + π · 0.0414² ≈ 0.791`, so about `79%` of the square is
disk and `21%` is interstitial void. Any `k × k` inscribed-circle grid covers exactly
`k² · π/(2k)² = π/4 ≈ 0.785` of its square, *independent of `k`* — refining the grid shrinks every
gap in proportion, so the density never improves. Against the Cauchy–Schwarz ceiling this is sharper
still: a `k²`-circle grid attains `Σ rᵢ = k/2` versus a ceiling `√(k²/π) = k/√π`, a ratio of
`√π/2 ≈ 0.886` that is *also* independent of `k`. So a square grid is structurally pinned at about
`88.6%` of the area-efficiency ceiling for its circle count, and my `5 × 5`-plus-filler sits at
`2.5414/2.8768 ≈ 0.883` of the `n = 26` ceiling — essentially that same `88.6%`. The frontier band
near `2.636` reaches `≈ 91.6%`. So the whole searched improvement I am about to chase, from this
floor up to the frontier, is a push in area-efficiency from `~88%` to `~92%`, and that three-point
gain is buyable *only* by irregularity — unequal radii and corners used as prime locations. A grid,
at any `k`, is capped at `√π/2` by construction and cannot make it.

There is an apparent paradox in that claim that I should resolve now, because getting it wrong would
mislead the next rung. Cauchy–Schwarz says `Σ rᵢ ≤ √(n · Σ rᵢ²)` with *equality exactly when all
radii are equal* — so for a fixed number of circles and a fixed total area `Σ rᵢ²`, equal radii
*maximise* the sum. That seems to argue *for* the grid's equal radii and against the irregular
packings I keep insisting on. The resolution is that the packing constraint does not hold `Σ rᵢ²`
fixed. There are two levers in the bound: how much area the circles cover (`Σ rᵢ²`, wanting to reach
the budget `1/π ≈ 0.318`) and how equal they are (making the inequality tight). The grid wins the
equality lever outright — its `25` equal circles give `Σ rᵢ² = 0.25` and `√(25 · 0.25) = 2.5`
exactly, Cauchy–Schwarz *tight* — but it loses the area lever badly, covering only `0.25` of the
`0.318` budget. The frontier trades the other way. Its `Σ rᵢ² ≥ (Σ rᵢ)²/n = 2.636²/26 ≈ 0.267` is
strictly above the grid's `0.25`, so a frontier packing provably covers more area (at least
`π · 0.267 ≈ 0.84` of the square versus the grid's `0.79`) — and I can conclude that *without ever
seeing the frontier configuration*, purely from its reported sum and `n = 26`. It pays for that
extra coverage with unequal radii, accepting some Cauchy–Schwarz slack, and the geometric gain
dominates the equality loss. So the grid is not wrong to use equal radii; it is wrong to leave a
fifth of the square empty. The next rung has to fill that void, and it can only do so by giving up
the very equality the grid was clinging to.

There is a subtlety in that equality claim I want to get exactly right, because it is easy to
overstate. Cauchy–Schwarz is *tight* for the grid only on the `25`-circle subset, where every radius
is literally the same: `Σ rᵢ = 2.5`, `Σ rᵢ² = 0.25`, and `√(25 · 0.25) = 2.5` with no slack at all.
The instant I add the `26`th circle the radii stop being equal — `25` of them are `0.1` and one is
`0.0414` — and the bound loosens. For the full `26`-set, `Σ rᵢ² = 25·0.1² + 0.0414² =
0.25 + 0.001716 = 0.251716`, so Cauchy–Schwarz now promises only
`Σ rᵢ ≤ √(26 · 0.251716) = √6.5446 ≈ 2.5582`, while the actual sum is `2.5414` — a slack of about
`0.0168` opened purely by that one unequal filler. That is a miniature of the whole problem living in
a single circle: even the sliver of inequality the grid is *forced* into already spends some of its
Cauchy–Schwarz headroom, and the frontier spends far more, accepting a much larger equality slack in
exchange for a much larger area gain. The grid sits at the extreme equal-radius corner of that
trade-off; every rung after it has to walk away from that corner deliberately, and this one involuntary
filler is the first step off it.

So I am not proposing this as a strong answer; I am proposing it as the right starting rung, and its
rigidity is precisely what makes it diagnostic. It realises the "many small circles raise the sum"
insight in the cleanest possible form, it is guaranteed feasible, it is parameter-free once `k = 5`
is fixed, and it produces a concrete `Σ rᵢ` that every searched rung must beat. What it cannot do is
break its own symmetry, and that names the next move without ambiguity. If my diagnosis is right —
that the loss is unequal radii and wasted corners, not anything subtle — then the very first thing
that lets the radii differ and the centres move should clear this floor and clear it by a visible
margin on the `n26` score, buying back the easy part of the `0.0946` gap; how much of it, and where
it then stalls, is what the next rung's measurement will tell me. That means handing the problem to a
constrained nonlinear optimizer: write the wall and non-overlap conditions as inequalities, let the
solver make the radii unequal and slide circles into the corners and edges, and see how far a single
local optimum climbs above `2.5414` — knowing already that one local descent will find *a* good
irregular packing but not necessarily the best basin, which is where the rungs after it will have to
go.
