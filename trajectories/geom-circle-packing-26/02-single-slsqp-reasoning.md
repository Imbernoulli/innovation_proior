The grid baseline did exactly what a baseline should: it put `2.5414` on the board, feasible with
the tolerance entirely unspent (its tangencies sit at the `10⁻¹⁶` floating-point floor, `10⁹` inside
`atol`), and then it told me, by its own rigidity, what is wrong. Every circle is locked to radius
`0.1` on a lattice, the corners and edges are wasted, and the sum-of-radii objective clearly wants
something the grid cannot give — unequal radii and freely-placed centres. I even priced the loss: the
grid attains about `88.3%` of the Cauchy–Schwarz area ceiling `√(26/π) ≈ 2.877`, covering only `79%`
of the square, while the frontier near `2.636` reaches `≈ 91.6%` by covering more area with unequal
circles. The `0.0946` gap from `2.5414` to the frontier is that missing area-efficiency. So the move
now is to stop constructing the packing by hand and start *optimizing* it: write the feasibility
constraints as nonlinear inequalities and hand the whole thing to a constrained optimizer, letting it
find the irregular arrangement no grid can express.

Let me be precise about the optimization problem, because its structure dictates the method. The
variables are the `26` centres `(xᵢ, yᵢ)` and the `26` radii `rᵢ` — `52 + 26 = 78` variables in all.
The objective `Σ rᵢ` is linear. The constraints come in two families. The wall constraints
`xᵢ ≥ rᵢ`, `xᵢ ≤ 1 − rᵢ`, `yᵢ ≥ rᵢ`, `yᵢ ≤ 1 − rᵢ` are four per circle, `104` linear inequalities,
and they are convex. The pairwise non-overlap constraints `√((xᵢ−xⱼ)² + (yᵢ−yⱼ)²) ≥ rᵢ + rⱼ` run over
all `C(26,2) = 325` pairs, so the full problem carries `104 + 325 = 429` inequality constraints on
`78` variables. The trouble is entirely in that second family: `√((xᵢ−xⱼ)²+(yᵢ−yⱼ)²)` is a convex
function of the centre differences, so requiring it to be *≥* a linear quantity is requiring a convex
function to stay *above* a value — the feasible side is the *outside* of a disk in configuration
space, which is nonconvex. Geometrically the feasible region is the complement of a union of `325`
"too-close" slabs, a highly nonconvex set with many disconnected components (one per way of choosing
which circles touch which). So this is a nonconvex QCQP with no global convexity to exploit, and there
is no hope of a global guarantee; the realistic goal is a strong local optimum, and the quality of
the local optimum I reach will be decided by where I start.

The nonconvexity is worth making concrete on a single pair, because it explains the basin structure I
will be fighting for the rest of the ladder. Freeze the two radii of circles `i` and `j` and ask
where centre `i` may sit: the constraint `‖cᵢ − cⱼ‖ ≥ rᵢ + rⱼ` says `cᵢ` must lie *outside* a disk of
radius `rᵢ + rⱼ` around `cⱼ` — the exterior of a disk, which is the textbook nonconvex set (the
midpoint of two feasible points, one on each side of the disk, passes straight through the forbidden
interior). Multiply that by `325` pairs and the feasible region becomes a thin, twisting manifold of
jammed configurations. On top of that geometric nonconvexity there is a large discrete symmetry:
the square has `8` symmetries (rotations and reflections), and any relabelling of similarly-sized
circles gives another distinct configuration with the same score, so every good packing comes with a
whole orbit of mirror-and-permutation copies, each sitting in its own basin. A single SLSQP descent
cannot see across those barriers; it slides to the bottom of whichever basin contains its start and
stops. That is the precise sense in which "where I start" decides everything.

The standard strong tool for exactly this shape — a smooth nonlinear objective with smooth nonlinear
inequality constraints — is SLSQP, Sequential Least-Squares Quadratic Programming. At each iterate it
linearizes the `429` constraints, builds a quadratic model of the Lagrangian, solves that QP for a
step, and iterates, maintaining an active set and converging to a KKT point of the constrained
problem. Let me check that choice against the alternatives rather than take it on faith, because the
geometry of *this* problem argues for it specifically. An interior-point method wants to travel down
the centre of the feasible region and needs a strictly interior iterate, but the optimum here is
exactly the opposite kind of point: at a locally maximal packing every circle is pressed until it
touches a wall or a neighbour, so a large number of constraints are *active* at the solution and the
optimizer must ride the boundary, which is native to an active-set SQP method and awkward for a
path-following interior-point one. A pure penalty or augmented-Lagrangian method could handle the
constraints but would need careful penalty scheduling across `429` constraints of two very different
scales and tends to converge slowly to the boundary. A projected-gradient scheme has no clean
projection onto a nonconvex non-overlap set. SLSQP linearizes each distance constraint into a
half-plane, which is exactly the right local model of "don't move these two circles closer," and its
QP subproblem naturally selects which contacts are active. So SLSQP is not just a default; it is the
method whose mechanics match a boundary-pressed nonconvex packing optimum.

There is a design decision about the radii that I want to get right from the start, because it
determines how I initialize and how I read off the answer. I have two genuinely different options.
One is to treat the radii as free SLSQP variables alongside the centres — optimize all `78` together.
The other exploits a structural fact worth stating carefully: *for fixed centres, the optimal radii
are the solution of a linear program.* With the centres held, maximize `Σ rᵢ` subject to
`rᵢ + rⱼ ≤ dᵢⱼ` for every pair and `0 ≤ rᵢ ≤ wallᵢ` where `wallᵢ = min(xᵢ, yᵢ, 1−xᵢ, 1−yᵢ)`. Every
constraint is linear in the radii, so this is a convex LP, solved exactly and cheaply to a global
optimum. That decomposition is tempting because it says the *radii* half of the problem is easy — all
the difficulty lives in the centres — so I could optimize only the `52` centre coordinates and snap
the radii to their LP optimum for each configuration.

Before I let that tempt me into a centre-only scheme, let me test how much the LP alone can do, by
running it in my head on the one configuration whose radii I already know: the grid. Feed the `25`
grid centres to the radii LP. Each interior circle sits `0.2` from four neighbours, giving
`rᵢ + rⱼ ≤ 0.2`; the corner and edge circles are additionally capped by `wallᵢ = 0.1`. Could the LP
do better than the uniform `0.1`? To grow one circle past `0.1` it must shrink a neighbour below `0.1`
to keep `rᵢ + rⱼ ≤ 0.2`, a zero-sum swap that leaves the pair's contribution unchanged — and any
circle it tries to enlarge on the boundary is blocked by `wallᵢ = 0.1` anyway. So the LP returns the
grid's own `0.1` everywhere and recovers exactly `2.5`; it buys nothing while the centres are frozen
on the lattice. That little trace is decisive: the radii LP is powerful but *impotent without centre
motion*. The entire climb above the grid has to come from *moving centres* so that new tangency
patterns open up — precisely what a center-only-plus-LP loop cannot initiate on its own without an
outer optimizer to move the centres, and precisely what makes the coupling between centres and radii
the crux.

It pays to trace, on the grid I already understand, the very first move a joint solver will want to
make, because it shows in one picture why centres and radii cannot be optimised apart. Take the
bottom-left corner circle at `(0.1, 0.1)`. Its wall slack is `min(0.1, 0.1, 0.9, 0.9) − 0.1 = 0`, so
it is tangent to both near walls; and its distance to each edge neighbour, `(0.3, 0.1)` and
`(0.1, 0.3)`, is exactly `0.2 = 0.1 + 0.1`, so it is tangent to those two as well. That circle is
pinned from four sides at `r = 0.1` — two walls and two neighbours — and yet a corner is geometrically
the place a disk *should* be able to grow largest, because two of its four confining sides are free
walls that recede as it enlarges rather than pushing back. The unrealised potential is enormous: a
corner disk with no neighbours at all, tangent to two walls at `(r, r)`, is stopped only by the far
walls, `r ≤ 1 − r`, i.e. `r = 0.5`. So the grid's corner circle sits at `0.1` against a ceiling of
`0.5`, held down entirely by its two edge neighbours. To cash in any of that gap the solver has to
*move those neighbours* — slide `(0.3, 0.1)` rightward along the bottom row and `(0.1, 0.3)` upward
along the left column while pulling the corner disk into `(r, r)` as they retreat — and sliding those
edge circles compresses *their* neighbours in turn, so the displacement ripples down the whole bottom
row and left column at once. That is the shape of the move: a coordinated shove of a chain of circles
that lets one corner disk grow. It is exactly the move a scheme that optimises radii and centres in
separate blocks cannot begin — freezing the centres and running the radii LP leaves the corner at
`0.1` (I just checked that the LP recovers the grid's own radii), and freezing the radii to grow the
corner would instantly overlap the still-fixed neighbours. Only a solver carrying all `78` variables
in one quadratic model can price the corner's radius growth against the small centre displacements of
the whole chain and take the step. That is the coupling, made concrete on an example I can compute by
hand.

That coupling is why I let SLSQP optimize centres and radii *jointly* for this rung rather than
freeze the radii to an inner LP. The value of moving a centre is exactly how much radius that move
unlocks across all of its neighbours at once, and only a solver that carries centres and radii in the
same quadratic model can trade a small centre displacement against the radius growth it enables. A
split scheme optimizes one block with the other frozen and misses those coordinated moves; to
recover them it would have to differentiate the LP optimum with respect to the centres (its dual
prices) or finite-difference through `52` centre perturbations per gradient, both more fragile than
just handing all `78` variables to one SQP. The objective gradient is trivial to supply analytically
— `∇(−Σ rᵢ)` is `0` on every centre coordinate and `−1` on every radius — so I give SLSQP the exact
jacobian of the objective and let it finite-difference only the constraint jacobians.

I then keep the LP, but only as a final polish. SLSQP stops at a KKT point where the radii may sit a
hair inside the feasible boundary — the stopping tolerance leaves microscopic slack — so after it
converges I *re-tighten* the radii to their exact LP optimum for the returned centres. That step is
free (one convex LP), it recovers whatever slack the local solver left, and it guarantees the
reported radii are maximal for those centres. It cannot change the centres or the basin; it just
squeezes the last drops of radius out of the configuration SLSQP found.

That the polish can only help is not a hope but a one-line fact worth stating, because it fixes what I
am entitled to claim about the reported number. Whatever radii SLSQP returns are themselves feasible
for the fixed-centre LP: at the final centres they satisfy `rᵢ + rⱼ ≤ dᵢⱼ` and `rᵢ ≤ wallᵢ`, because
the packing is feasible. So the SLSQP radii are a feasible point of the very LP I then solve to
optimality, and the LP optimum's sum is therefore `≥` the SLSQP sum by definition, with equality
exactly when SLSQP already sat at the radius vertex. Re-tightening is thus monotone: it never lowers
the reported sum and generically raises it by the microscopic slack the stopping tolerance left, at
the cost of one convex solve. It also disposes of the one failure mode I would otherwise have to watch
for — a returned configuration with some circle floating a hair short of its contacts — by growing
every under-tight radius out to the boundary its centres allow.

Initialization has to be feasible and reasonable. A purely random scatter of centres with tiny radii
is already feasible, so I draw `26` centres uniformly in a slightly inset square `[0.05, 0.95]²` — the
inset keeps the initial circles off the walls so the LP has a non-degenerate feasible start and SLSQP
does not begin pinned at a constraint corner where its first QP step can stall — and I set the initial
radii to their LP optimum for that scatter. That start respects every constraint, so SLSQP begins
inside the feasible region and climbs.

The `0.05` inset is not arbitrary, and it is worth pricing so I do not mistake it for a fudge. If I
drew centres in the full square, some would land within a whisker of a wall, giving
`wallᵢ = min(xᵢ, yᵢ, 1−xᵢ, 1−yᵢ)` near zero; the radii LP would then be forced to set those circles'
radii to essentially zero, and a start with several radii pinned at zero is a degenerate LP basis and
a poor SLSQP seed — the solver would begin jammed against a cluster of active wall constraints with
almost no room for its first QP step. Insetting to `[0.05, 0.95]` guarantees every initial
`wallᵢ ≥ 0.05`, so every circle can carry a radius up to `0.05` before its wall binds and the LP
returns a non-degenerate spread of strictly positive radii. The inset costs nothing in reachable
configurations, since SLSQP is free to slide any circle back against a wall during the descent; it
only keeps the *start* off the boundary corner where the first step would stall.

The solver budget follows from what I just established about the geometry. I bound the centre
coordinates to `[0, 1]` and the radii to `[0, 0.5]` — no circle in the unit square can have radius
above `0.5`, so `0.5` is the true variable bound, not a guessed one, and giving SLSQP the honest box
keeps its QP subproblems well-posed. I set `maxiter = 300`: `78` variables against a few-dozen active
contacts is a modest QP to resolve, and a jammed local optimum is reached in far fewer than `300`
outer iterations, so the cap is generous headroom rather than a limiter. I set `ftol = 1e-10` because
the whole contest lives in the sixth decimal of `Σ rᵢ`; a looser objective tolerance would stop the
descent while the last thousandths of radius are still being squeezed out, and those thousandths are
the entire point. The one seed is fixed (default `0`) so the reported number is the deterministic
output of a stated draw, not a lucky sample I cannot reproduce — which also means the single-start
result is honestly *just one draw*, with all the basin-lottery risk that implies.

Let me sanity-check that a single start is cheap enough to lean on, because whether it is decides how
aggressively a later rung could multiply it. One constraint evaluation computes `325` pairwise
distances plus `104` linear wall expressions — call it `325` square roots, trivially vectorised.
SLSQP needs the constraint jacobian, and since I supply only the objective gradient analytically, it
finite-differences the `429` constraints across all `78` variables: about `78 × 325 ≈ 2.5×10⁴`
distance evaluations per jacobian. A jammed optimum is reached in on the order of `10²` outer
iterations, so a single start costs a few million distance evaluations — a fraction of a second to a
couple of seconds of wall clock in vectorised numpy. That figure matters beyond this rung: it is small
enough that I could afford *dozens to hundreds* of independent starts inside a modest budget, which is
exactly the lever a multi-start version would pull. And the isostatic count below is what keeps it
affordable even as the constraint list grows — with only a few dozen contacts active at any iterate,
the QP subproblems SLSQP solves stay small, so riding the boundary of `429` constraints does not cost
`429` constraints' worth of work.

What do I expect, and what will the `n26` number tell me? A single SLSQP run from one random scatter
will find *a* local optimum, and it will certainly clear the grid, because it can do the two things
the grid could not: make the radii unequal, and slide circles into the corners and edges where a
circle bounded by two free walls can grow past `0.1`. In area-budget terms it will push `Σ rᵢ²` above
the grid's `0.25` by filling some of the void, which is exactly how a packing earns sum. So I expect
the run to land comfortably above `2.5414` — into the high `2.5`s, plausibly a little past `2.59` — a
real jump that would confirm the diagnosis that unequal radii and corner use are where the easy gains
live. I can pin that expectation to the area lever to keep it honest: a sum near `2.59` forces
`Σ rᵢ² ≥ 2.59²/26 ≈ 0.258` by Cauchy–Schwarz, just above the grid's `0.25`, so the run will have
filled only a sliver of the `21%` void the grid left, while the frontier's `≈ 0.267` demands
noticeably more void-filling than one random basin will manage.

I can push that lever one step further into an actual void-fraction, because it turns "a sliver" into
a number I can check against the result. The grid covers area `25·π·0.1² + π·0.0414² ≈ 0.791`, leaving
`0.209` of the square empty. A packing that reaches `Σ rᵢ ≈ 2.59` carries `Σ rᵢ² ≈ 0.258` of covered
radius-squared, hence disk area `≈ π·0.258 ≈ 0.811`, so it will have filled roughly
`(0.811 − 0.791)/0.209 ≈ 9%` of the void the grid left. A frontier packing at `Σ rᵢ² ≈ 0.267` covers
`π·0.267 ≈ 0.839`, filling about `(0.839 − 0.791)/0.209 ≈ 23%` of that void. So the honest picture of
what one start can do is close roughly the first `9%` of the empty space, against the `~23%` the
frontier reaches — a real bite, but only the easy corner of the problem. If the `n26` number lands
near `2.59`, that is the void fraction it will have bought; if it lands higher, the same area identity
tells me exactly how much more void it must have filled to get there.

But I expect it to fall well short of
the `2.636` frontier, and I can name why before I run it:
one initialization drops into one basin of a landscape whose basins correspond to different contact
topologies of very different quality, and a single random draw has no way to prefer the good ones. If
the single run clears the grid but stalls in the high `2.5`s, that gap is not the optimizer's fault —
SLSQP will have returned a clean KKT point — it is the single start's fault.

I can even sketch the shape of the optimum SLSQP will report, as a consistency check on the method.
At any locally maximal packing every circle must be pinned: if some circle touched nothing and no
wall, it could grow and raise the sum, contradicting local maximality. So at the KKT point each of
the `26` radii is held by at least one active constraint — a wall or a neighbour contact — and the
active contacts form a rigid graph that the `78` variables cannot relax further. That is the
structural signature I should see in the returned configuration: a jammed, irregular arrangement with
a few large corner/edge circles and many gap-fillers, every one of them touching something. If
instead SLSQP returned a configuration with a slack circle floating free, I would distrust the run;
the LP re-tightening exists precisely to eliminate that failure mode by growing any under-tight
radius to its contact.

The rigidity count sharpens this into something I can use as a sanity gauge. A jammed packing of
disks is, in the language of frictionless rigidity, roughly isostatic: to remove all internal
freedoms of `N` disks you need on the order of `2N` contacts (each contact kills one relative
degree of freedom, and there are about `2N` translational freedoms after fixing global motions). For
`N = 26` that is a contact network of order `50`, drawn from the `104` wall and `325` pair
constraints. So I should expect the returned optimum to activate a few dozen of the `429`
constraints — a sparse but rigid web — and the rest to sit comfortably slack. This matters for two
reasons. It tells me the active set SLSQP maintains stays small and tractable even though the
constraint list is large, which is part of why SLSQP is affordable here. And it tells me the basin
structure is *combinatorial*: each distinct rigid contact graph is a different basin, the number of
such graphs is enormous, and a single random start commits me to whichever one its scatter falls
into. That is the concrete meaning of "one basin out of many," and it is why one start cannot be
enough.

So this rung is deliberately the simplest honest version of the optimization: one initialization, one
joint SLSQP descent over all `78` variables, LP-polished radii, feasibility verified against the real
constraints. It will show that the optimizer is the right engine — the climb above `2.5414` is real
geometric gain, not a tolerance artefact, since the run should land far inside `atol`. And it will
expose that the limitation is not the engine but the *single start*: one basin out of many. The next
rung has to wrap this same solver in many restarts and keep the best, converting compute into a
better basin than one random draw can offer — and I expect that to help, but only up to the point
where uniform scattering stops finding rarer and better basins.
