I want a packing of `26` circles in the unit square with the sum of radii as large as possible,
and before I reach for any optimizer I want a baseline that is principled, guaranteed feasible, and
parameter-free — a concrete number on the board that every later, searched rung must beat. The
natural such object is a regular grid of equal circles, because its value can be written in closed
form, so it lets me understand the *objective* before I understand the landscape.

The objective is unusual, and it decides everything that follows. In ordinary equal-circle packing
the quantity maximized is the single common radius `r`, and there the winning move is *few* large
circles — one circle fills the square at `r = 0.5`, and every extra circle you demand forces the
common radius down. Here the score is `Σ rᵢ`, a *sum*, and that flips the preference completely.
Put `n = k²` equal circles on a `k × k` grid, one per cell: each cell has width `1/k`, and the
largest circle inscribed in it and tangent to its four neighbours has radius `r = 1/(2k)`. The grid
then holds `k²` circles of radius `1/(2k)`, so

    Σ rᵢ = k² · 1/(2k) = k/2.

For `k = 1` that is `0.5` — the single big circle that *wins* the equal-radius game — and for
`k = 5` it is `2.5`, five times larger. Halving each radius while quadrupling the count doubles the
sum. So under a sum objective many small circles is not a weakness to be tolerated; it is the whole
game, and the good packings will be crowded with small circles rather than dominated by a few large
ones.

There is a ceiling on how far "more small circles" can go, and I want it now so I can read every
later number against it. The disks are disjoint inside a region of area `1`, so `Σ π rᵢ² ≤ 1`, i.e.
`Σ rᵢ² ≤ 1/π ≈ 0.3183`; by Cauchy–Schwarz `(Σ rᵢ)² ≤ 26 · Σ rᵢ² ≤ 26/π`, so

    Σ rᵢ ≤ √(26/π) ≈ 2.8768.

That is an absolute first-principles ceiling for `n = 26`. The frontier quoted in the background
sits near `2.636`, `≈ 91.6%` of it — the gap is the unavoidable slack from the fact that disks
cannot tile the plane without leaving interstitial voids. So the achievable target lives just below
`2.64`, well under the naive `2.877`, and my grid floor will land below even that.

Now I have to hit exactly `26`, and `26` is not a perfect square, so the clean `k/2` formula does
not apply directly. The square grids that fit a budget of `26` are `k = 1..5`, with counts
`1, 4, 9, 16, 25` and sums `0.5..2.5`; `k = 6` needs `36 > 26`. Since `k/2` is monotone, `5 × 5` is
the best of the fitting grids, and it leaves exactly one circle over. Two escapes need ruling out
before I commit. A `6 × 6` grid has a higher full-grid `k/2 = 3.0`, but I can keep only `26` of its
`36` positions: those survivors still sit on the `1/6` lattice at radius `1/12 ≈ 0.0833`, summing
`26 · 1/12 ≈ 2.167` — *below* `2.5`, because dropping circles throws away the very count that made
`k/2` large and the survivors stay pinned by their remaining neighbours and the walls. The subtler
escape is a two-scale packing: a `4 × 4` grid of `16` circles at `r = 0.125` (sum `2.0`) plus a
filler in each of its `3 × 3 = 9` interior interstices, each of radius `(√2 − 1)·0.125 ≈ 0.0518`,
adding `9 · 0.0518 ≈ 0.466` — total `≈ 2.466` with `25` circles, already below `2.5` before the
`26`th. The coarse grid's circles are individually larger (`0.125` vs `0.1`) but too few, and its
fillers too small (`0.052`) and too few, so its count-times-size falls short of `25` uniform `0.1`
circles. The uniform fine grid wins among *structured* options because the sum rewards count. So the
grid is forced: `5 × 5` full, `25` circles of radius `0.1` contributing `2.5`, one circle left. The
moral registers already — even the best two-scale structured packing underperforms the fine grid,
so whatever beats `2.5` substantially will not be structured at all; it will be irregular, which no
grid can produce.

The `25` circles sit at `(i + 0.5)/5 = 0.1, 0.3, 0.5, 0.7, 0.9` on each axis. Adjacent centres are
`0.2` apart with radii `0.1`, so tangent, not overlapping; the outermost centres at `0.1` and `0.9`
lie one radius from the walls. The whole grid sits exactly on the feasibility boundary, with only
the floating-point residue of computing `0.1 + 0.1` versus `0.2` — precisely what the tolerance
accepts, so `2.5` is genuinely on the board.

Where does the `26`th circle go? There are three qualitatively different gaps, worth sizing. The
*interior interstitial*: four mutually adjacent circles sit on a `0.2 × 0.2` square whose centre is
`√2·r ≈ 0.1414` from all four, so a disk there grows to `√2·r − r = (√2 − 1)·r ≈ 0.0414`, and there
are `4 × 4 = 16` such gaps. The *wall-edge*: a circle tangent to the left wall between two
vertically adjacent grid circles at `(0.1, 0.1)` and `(0.1, 0.3)` has centre `(r, 0.2)`; tangency
gives `(r − 0.1)² + 0.1² = (r + 0.1)²`, i.e. `0.01 = 0.4 r`, `r = 0.025`. The *corner*: a circle at
`(r, r)` tangent to both walls with nearest grid circle at `(0.1, 0.1)` needs `√2·(0.1 − r) = r + 0.1`,
which rationalises to `r = 0.1(√2 − 1)² = 0.1(3 − 2√2) ≈ 0.0172`. So the three are `0.0172`, `0.025`,
`0.0414`, and the interior gap wins: symmetric on all four sides, it admits the largest inscribed
disk, while the corner and edge circles are pressed by walls into less room. There is an irony
worth registering — in a good sum-of-radii packing the corner is the *prize* location, two free
walls letting a disk grow large, yet the grid has already spent its corner on an ordinary `0.1`
circle and left only the `0.0172` sliver. Priced as finished packings the three land at `2.5172`,
`2.525`, `2.5414`, so the interior gap buys `+0.0242` over the corner — a visible slice of the
`0.0946` I am trying to close, riding entirely on where I drop one circle.

I drop the `26`th into the interior interstice at `(0.4, 0.4)`: its distance to each of the four
neighbours at `(0.3, 0.3), (0.5, 0.3), (0.3, 0.5), (0.5, 0.5)` is `√0.02 ≈ 0.14142`, equal to the
radius sum `0.0414 + 0.1`, so tangent; its distance to the walls is `0.4`, far above `0.0414`. The
total is

    Σ rᵢ = 2.5 + (√2 − 1)·0.1 = 2.5414213562…

Is that optimal within the one honest degree of freedom left — the grid radius? Shrink all `25` grid
circles to a common `r ≤ 0.1` on the fixed lattice; the interstice centre stays `0.1414` from its
neighbours, so the filler grows to `0.1414 − r`, and the total is `25 r + (0.1414 − r) = 24 r + 0.1414`,
strictly increasing in `r` (slope `+24`). The sum is therefore maximised at `r = 0.1`, the full
inscribed grid: the `24` circles I would shrink cost far more than the one filler I would grow. No
free lunch in loosening the grid.

Now where this lands. The grid gives `≈ 2.5414`, the frontier band near `2.636`, the area ceiling
`2.877` — so the grid leaves `≈ 0.0946` relative to the frontier, an enormous gap in a problem whose
competitive action is in the sixth decimal (the quoted frontier values span only `~10⁻⁴`). I can
locate the waste in the area budget. Any `k × k` inscribed-circle grid covers exactly
`k² · π/(2k)² = π/4 ≈ 0.785` of its square, *independent of `k`* — refining shrinks every gap in
proportion, so density never improves. Against the ceiling, a `k²`-circle grid attains `Σ rᵢ = k/2`
versus `√(k²/π) = k/√π`, a ratio `√π/2 ≈ 0.886`, also independent of `k`. So a square grid is
structurally pinned near `88.6%` of the area-efficiency ceiling, and my `5 × 5`-plus-filler sits at
`2.5414/2.8768 ≈ 0.883`; the frontier reaches `≈ 91.6%`. The whole searched improvement I am about
to chase is a push in area-efficiency from `~88%` to `~92%`, buyable only by irregularity — unequal
radii and corners used as prime locations — which a grid at any `k` cannot make.

There is an apparent paradox to resolve, because getting it wrong would mislead the next rung.
Cauchy–Schwarz has equality exactly when all radii are equal, so for a *fixed* `Σ rᵢ²` equal radii
maximise the sum — which seems to argue *for* the grid. The resolution is that the packing does not
hold `Σ rᵢ²` fixed. There are two levers: how much area the circles cover (`Σ rᵢ²`, wanting to reach
`1/π ≈ 0.318`) and how equal they are. The grid wins the equality lever outright — its `25` equal
circles give `Σ rᵢ² = 0.25` and `√(25·0.25) = 2.5`, Cauchy–Schwarz tight — but loses the area lever,
covering only `0.25` of the `0.318` budget. The frontier trades the other way: its
`Σ rᵢ² ≥ 2.636²/26 ≈ 0.267` exceeds `0.25`, so it provably covers more area (`≈ 0.84` of the square
versus the grid's `0.79`) — a conclusion I can draw from its reported sum alone, without seeing its
configuration. It pays for that coverage with unequal radii, accepting Cauchy–Schwarz slack, and the
geometric gain dominates. So the grid is not wrong to use equal radii; it is wrong to leave a fifth
of the square empty, and the next rung can fill that void only by giving up the equality the grid
clings to.

So this is the right *starting* rung, not a strong answer — its rigidity is what makes it
diagnostic. It realises the "many small circles raise the sum" insight cleanly, is guaranteed
feasible, is parameter-free once `k = 5` is fixed, and puts a concrete floor on the board. What it
cannot do is break its own symmetry, and that names the next move: hand the problem to a constrained
nonlinear optimizer — write the wall and non-overlap conditions as inequalities, let the solver make
the radii unequal and slide circles into the corners and edges, and see how far a single local
optimum climbs above `2.5414`.
