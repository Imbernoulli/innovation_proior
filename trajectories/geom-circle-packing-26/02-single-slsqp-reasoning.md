The grid baseline did exactly what a baseline should: it put `2.5414` on the board and then told
me, by its own rigidity, what is wrong. Every circle is locked to radius `0.1` on a lattice, the
corners and edges are wasted, and the sum-of-radii objective clearly wants something the grid
cannot give — unequal radii and freely-placed centers. So the move now is to stop constructing the
packing by hand and start *optimizing* it: write the feasibility constraints as nonlinear
inequalities and hand the whole thing to a constrained optimizer, letting it find an irregular
arrangement no grid can express.

Let me be precise about the optimization problem, because the structure dictates the method. The
variables are the `26` centers `(xᵢ, yᵢ)` and the `26` radii `rᵢ` — `78` variables in all. The
objective `Σ rᵢ` is linear. The constraints are: the wall constraints `xᵢ ≥ rᵢ`, `xᵢ ≤ 1 − rᵢ`,
`yᵢ ≥ rᵢ`, `yᵢ ≤ 1 − rᵢ` (linear in the variables), and the pairwise non-overlap constraints
`√((xᵢ−xⱼ)² + (yᵢ−yⱼ)²) ≥ rᵢ + rⱼ` for all `325` pairs. The wall and objective parts are convex,
but the pairwise distance constraints are *nonconvex* — the feasible region is the complement of a
union of "too close" regions, which is not convex — so this is a nonconvex QCQP with no global
convexity to exploit. There is no hope of a global guarantee; the realistic goal is a strong local
optimum.

The standard strong tool for exactly this shape — a smooth nonlinear objective with smooth
nonlinear inequality constraints — is SLSQP, Sequential Least-Squares Quadratic Programming. It
linearizes the constraints and builds a quadratic model of the Lagrangian at each iterate, solves
that QP for a step, and iterates; it handles the `325` distance inequalities and the wall
inequalities directly, and it converges to a KKT point of the constrained problem. So the plan for
this rung is the simplest honest version: pick *one* initialization, run SLSQP on the full
`(centers, radii)` problem, and report what it reaches.

There is a design decision about the radii that I want to get right from the start, because it
determines how I initialize and how I read off the answer. I have two options. One is to treat the
radii as free SLSQP variables alongside the centers — optimize all `78` together. The other is to
exploit a structural fact: *for fixed centers, the optimal radii are the solution of a linear
program* — maximize `Σ rᵢ` subject to `rᵢ + rⱼ ≤ dᵢⱼ` and `rᵢ ≤ wallᵢ`, all linear in the radii.
That LP is convex and solved exactly and cheaply. So I could optimize only the centers and snap the
radii to their LP-optimal values for each center configuration. For this first SLSQP rung I will do
both in a mild way: let SLSQP optimize centers and radii jointly (so the optimizer can trade center
positions against radius growth in a single smooth descent), and then, at the end, *re-tighten* the
radii to their exact LP optimum for the returned centers. The re-tightening matters because SLSQP
stops at a KKT point where the radii may sit slightly inside the feasible boundary; the LP recovers
the last bit of slack for free and guarantees the radii are maximal for those centers.

Initialization. I need a feasible, reasonable starting point. A purely random scatter of centers
with tiny radii is feasible and gives SLSQP somewhere to start, so I draw `26` centers uniformly in
a slightly inset square (to keep them off the walls) and set the initial radii to their LP-optimal
values for that scatter. That LP start already respects every constraint, so SLSQP begins inside
the feasible region and climbs.

What do I expect? A single SLSQP run from one random scatter will find *a* local optimum — it will
certainly beat the grid, because it can make the radii unequal and slide circles into the corners
and edges the grid wasted, and the linear objective with a good local solver reliably tightens a
random feasible packing into a locally maximal one. But it will land in whatever basin its random
initialization happened to sit in, and the landscape of this nonconvex problem has many basins of
quite different quality. So I expect the single run to clear the grid comfortably — into the high
`2.5`s, maybe just past `2.59` — but to fall well short of the `2.636` frontier, because one
initialization simply cannot explore the basin structure. The limitation this rung will expose is
therefore not the optimizer but the *single start*: SLSQP is the right local engine, and the next
rung has to wrap it in many restarts to find a better basin than one random draw can offer.
