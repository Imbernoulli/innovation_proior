The grid baseline did its job: it put `2.5414` on the board, feasible with the tolerance entirely
unspent (its tangencies sit at the `10⁻¹⁶` floating-point floor), and then told me by its rigidity
what is wrong. Every circle is locked to `0.1` on a lattice, the corners and edges are wasted, and
the sum-of-radii objective wants what the grid cannot give — unequal radii and freely-placed
centres. I priced the loss: the grid reaches `≈ 88.3%` of the Cauchy–Schwarz ceiling `√(26/π) ≈ 2.877`,
covering only `79%` of the square, while the frontier near `2.636` reaches `≈ 91.6%` by covering
more area with unequal circles. The `0.0946` gap is that missing area-efficiency. So the move is to
stop constructing by hand and start *optimizing*: write the feasibility constraints as nonlinear
inequalities and hand them to a constrained optimizer.

Let me be precise about the problem, because its structure dictates the method. The variables are
the `26` centres `(xᵢ, yᵢ)` and `26` radii `rᵢ` — `52 + 26 = 78` in all. The objective `Σ rᵢ` is
linear. The wall constraints `xᵢ ≥ rᵢ, xᵢ ≤ 1 − rᵢ, yᵢ ≥ rᵢ, yᵢ ≤ 1 − rᵢ` are `104` linear
inequalities, convex. The pairwise non-overlap `√((xᵢ−xⱼ)² + (yᵢ−yⱼ)²) ≥ rᵢ + rⱼ` runs over all
`C(26,2) = 325` pairs, so the full problem carries `429` inequalities on `78` variables. The trouble
is all in the second family: requiring a convex distance to stay *above* a linear quantity puts the
feasible side on the *outside* of a disk in configuration space, which is nonconvex. Geometrically
the feasible region is the complement of a union of `325` "too-close" slabs — highly nonconvex, many
disconnected components, one per way of choosing which circles touch which. So this is a nonconvex
QCQP with no global convexity to exploit; the realistic goal is a strong local optimum, and its
quality is decided by where I start. On top of the geometric nonconvexity there is a large discrete
symmetry: the square's `8` symmetries and the free relabelling of similar-sized circles give every
good packing an orbit of mirror-and-permutation copies, each in its own basin. A single descent
slides to the bottom of whichever basin its start falls into and stops — the precise sense in which
"where I start" decides everything.

The standard strong tool for a smooth objective with smooth nonlinear inequality constraints is
SLSQP: at each iterate it linearizes the constraints, builds a quadratic model of the Lagrangian,
solves that QP for a step, maintains an active set, and converges to a KKT point. The geometry argues
for it specifically. At a locally maximal packing every circle is pressed until it touches a wall or
neighbour, so many constraints are *active* at the solution and the optimizer must ride the boundary
— native to an active-set SQP, awkward for a path-following interior-point method that needs a
strictly interior iterate. SLSQP linearizes each distance constraint into the half-plane "don't move
these two circles closer," exactly the right local model, and its QP subproblem selects which
contacts are active. So SLSQP is the method whose mechanics match a boundary-pressed nonconvex
packing optimum.

There is a design decision about the radii worth getting right, because it fixes how I initialize
and read off the answer. For fixed centres the optimal radii are the solution of a linear program:
maximize `Σ rᵢ` subject to `rᵢ + rⱼ ≤ dᵢⱼ` and `0 ≤ rᵢ ≤ wallᵢ = min(xᵢ, yᵢ, 1−xᵢ, 1−yᵢ)`. Every
constraint is linear in the radii, so this is convex and solved exactly and cheaply. That is tempting
because it says the *radii* half is easy — all the difficulty lives in the centres. But test the LP
on the one configuration whose radii I already know: feed it the grid centres. Each interior circle
sits `0.2` from four neighbours (`rᵢ + rⱼ ≤ 0.2`); to grow one past `0.1` it must shrink a neighbour
below `0.1`, a zero-sum swap, and any boundary circle it tries to enlarge is capped by `wallᵢ = 0.1`
anyway. So the LP returns the grid's own `0.1` everywhere and recovers exactly `2.5` — it buys
nothing while the centres are frozen. The radii LP is powerful but *impotent without centre motion*;
the entire climb above the grid has to come from moving centres so that new tangency patterns open.

That coupling is sharp on the grid's corner circle at `(0.1, 0.1)`. It is tangent to both near walls
and to its two edge neighbours at `(0.3, 0.1)` and `(0.1, 0.3)` — pinned from four sides at `r = 0.1`
— yet a corner disk with no neighbours, tangent to two walls at `(r, r)`, is stopped only by the far
walls at `r = 0.5`. To cash any of that gap the solver must *move those neighbours*: slide `(0.3, 0.1)`
rightward and `(0.1, 0.3)` upward while pulling the corner disk into `(r, r)` as they retreat, and
sliding those edge circles compresses *their* neighbours in turn, so the displacement ripples down the
whole bottom row and left column. That is the move — a coordinated shove of a chain of circles that
lets one corner disk grow. A scheme that optimises radii and centres in separate blocks cannot begin
it: freezing centres leaves the corner at `0.1`, and freezing radii to grow the corner instantly
overlaps the fixed neighbours. Only a solver carrying all `78` variables in one quadratic model can
price the corner's radius growth against the small centre displacements of the whole chain.

So I let SLSQP optimize centres and radii *jointly* rather than split them to an inner LP. The
objective gradient is trivial analytically — `∇(−Σ rᵢ)` is `0` on every centre coordinate and `−1`
on every radius — so I supply the exact objective jacobian and let SLSQP finite-difference only the
constraint jacobians. I keep the LP, but only as a final polish: SLSQP stops at a KKT point where the
radii may sit a hair inside the boundary, so after it converges I re-tighten the radii to their exact
LP optimum for the returned centres. That polish can only help, and it is one line: the SLSQP radii
are themselves feasible for the fixed-centre LP, so the LP optimum's sum is `≥` the SLSQP sum by
definition, equal only when SLSQP already sat at the radius vertex. Re-tightening is monotone — it
never lowers the sum and generically raises it by the microscopic stopping slack — and it disposes of
the one failure mode I would otherwise watch for, a circle floating a hair short of its contacts, by
growing every under-tight radius out to the boundary its centres allow.

Initialization has to be feasible and reasonable. A random scatter with tiny radii is already
feasible, so I draw `26` centres uniformly in a slightly inset square `[0.05, 0.95]²` and set the
initial radii to their LP optimum for that scatter. The inset is not a fudge: drawn from the full
square, some centres would land a whisker from a wall, giving `wallᵢ ≈ 0`, so the radii LP would pin
those radii to essentially zero — a degenerate LP basis and a poor SLSQP seed jammed against active
wall constraints with no room for its first QP step. Insetting to `[0.05, 0.95]` guarantees every
`wallᵢ ≥ 0.05` and a non-degenerate spread of positive radii; it costs nothing in reachable
configurations, since SLSQP is free to slide any circle back against a wall during the descent.

The rest of the budget follows the geometry. I bound the centres to `[0, 1]` and the radii to
`[0, 0.5]` — no circle in the unit square exceeds `0.5`, so that is the true bound, not a guess. I set
`maxiter = 300`, generous headroom since a jammed optimum of `78` variables against a few dozen active
contacts is reached in far fewer outer iterations. I set `ftol = 1e-10` because the whole contest
lives in the sixth decimal of `Σ rᵢ`, and a looser tolerance would stop while the last thousandths of
radius are still being squeezed out. The seed is fixed (default `0`) so the reported number is the
deterministic output of a stated draw — which also means this single-start result is honestly *just
one draw*, with all the basin-lottery risk that implies. A single descent costs a few million
pairwise-distance evaluations, a fraction of a second to a couple of seconds in vectorised numpy, so a
later rung could afford dozens to hundreds of independent starts.

What do I expect? A single SLSQP run from one random scatter will find *a* local optimum, and it will
clear the grid, because it can do the two things the grid could not: make the radii unequal, and slide
circles into the corners and edges where a disk braced by two free walls grows past `0.1`. In area
terms it pushes `Σ rᵢ²` above the grid's `0.25` by filling some of the void, which is exactly how a
packing earns sum. So I expect it to land comfortably above `2.5414`, up into the high `2.5`s — a real
jump confirming that unequal radii and corner use are where the easy gains live. But I expect it to
fall well short of `2.636`, and I can name why before running it: one initialization drops into one
basin of a landscape whose basins are different contact topologies of very different quality, and a
single random draw has no bias toward the good ones. If it stalls in the high `2.5`s, that gap is not
the optimizer's fault — SLSQP will have returned a clean KKT point — it is the single start's.

I can sketch the optimum's shape as a consistency check. At any locally maximal packing every circle
is pinned — if one touched nothing it could grow — so each radius is held by at least one active wall
or contact, and the active contacts form a rigid graph the `78` variables cannot relax. In the
language of frictionless rigidity a jammed packing of `N` disks is roughly isostatic: on the order of
`2N ≈ 52` contacts, drawn from the `429` constraints. So I should see a jammed, irregular arrangement
— a few large corner/edge circles and many gap-fillers, every one touching something — with a few
dozen of the `429` constraints active and the rest slack. That small active set is part of why SLSQP
stays affordable here even with a large constraint list. And it makes the basin structure
combinatorial: each distinct rigid contact graph is a distinct basin, the number of such graphs is
enormous, and a single random start commits me to whichever one its scatter falls into. That is the
concrete meaning of "one basin out of many," and why the next rung has to wrap this same solver in
many restarts and keep the best.
