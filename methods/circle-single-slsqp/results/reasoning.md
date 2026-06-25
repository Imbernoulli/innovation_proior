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

What kind of solver does this shape call for? I have a smooth nonlinear objective (linear here, but
that is just an easy special case) and a pile of smooth nonlinear inequality constraints, all
differentiable wherever no two centers coincide. That rules out the convex-only solvers and the
purely-combinatorial ones; what fits is a sequential quadratic programming method, which at each
iterate linearizes the constraints, builds a quadratic model of the Lagrangian, solves that QP for
a step, and repeats until it reaches a KKT point. SLSQP is the workhorse implementation of exactly
that. It handles the `325` distance inequalities and the wall inequalities directly as nonlinear
constraints, and it does not need the problem to be convex — it just needs smoothness, which I
have. So the plan for this rung is the simplest honest version of that idea: pick *one*
initialization, run SLSQP on the full `(centers, radii)` problem, and report what it reaches.

There is a design decision about the radii that I want to get right from the start, because it
determines how I initialize and how I read off the answer. I have two options. One is to treat the
radii as free SLSQP variables alongside the centers — optimize all `78` together. The other rests
on an observation: if I *freeze* the centers, the problem in the radii alone is linear. Maximize
`Σ rᵢ` subject to `rᵢ + rⱼ ≤ dᵢⱼ` (the distances `dᵢⱼ` are now constants) and `rᵢ ≤ wallᵢ` — a
linear objective with linear inequality constraints, i.e. a linear program, which is convex and
solved exactly and cheaply.

Before I lean on that observation, let me actually convince myself it gives sensible radii, on a
case small enough to do by hand. Put three circles on the midline of the square at `x = 0.2, 0.5,
0.8`, all at `y = 0.5`. The wall limit for the left circle is `min(0.2, 0.5, 0.8, 0.5) = 0.2`; for
the right circle, likewise `0.2`; for the middle circle, `0.5`. The center-to-center distances are
`d₀₁ = d₁₂ = 0.3` and `d₀₂ = 0.6`. Now reason about what the LP should do. The two outer circles
each want to grow; their only binding limits are their walls, so each should sit at `r = 0.2`.
Once they are at `0.2`, the middle circle is squeezed: `r₀ + r₁ ≤ 0.3` forces `r₁ ≤ 0.1`, and
`r₁ + r₂ ≤ 0.3` forces the same, so `r₁ = 0.1`. Its own wall limit `0.5` is slack and never binds.
Sum `= 0.2 + 0.1 + 0.2 = 0.5`. I ran the LP on these three centers and it returned `[0.2, 0.1,
0.2]`, sum `0.5` — matching the hand computation, and confirming the LP genuinely maximizes against
the binding constraints rather than, say, splitting growth evenly. Good: the "fixed-center radii
are an LP" claim is real, and the LP picks out exactly the constraints that should bind.

So I could optimize only the centers and snap the radii to their LP-optimal values for each center
configuration. For this first SLSQP rung I will combine the two ideas. Let SLSQP optimize centers
and radii *jointly* — so the optimizer can trade center positions against radius growth in a single
smooth descent, which a fixed-center scheme cannot do — and then, at the end, re-tighten the radii
to their exact LP optimum for the returned centers. The reasoning for the re-tightening is that
SLSQP stops at a KKT point where, for numerical reasons, the radii may sit a hair inside the
feasible boundary; the LP can only ever return radii that are at least as large, since it maximizes
the same objective over a superset-feasible set with the centers held at SLSQP's final values. So
re-tightening can never hurt and might recover a sliver of slack. I will check after the run how
much it actually recovers — I should not assume it grabs anything substantial.

Initialization. I need a feasible, reasonable starting point. A purely random scatter of centers
with radii set by the LP is the natural choice: I draw `26` centers uniformly in a slightly inset
square `[0.05, 0.95]²` (to keep them off the walls) and set the initial radii to their LP-optimal
values for that scatter. I want to be sure this start is actually feasible before handing it to
SLSQP, because the LP enforces `rᵢ + rⱼ ≤ dᵢⱼ` and `rᵢ ≤ wallᵢ` for *every* pair and wall, which
is precisely the packing's feasibility — but only if my LP rows really cover all `325` pairs.
Computing the worst constraint violation of the seed-`0` LP start, I get a maximum pairwise
violation of `≈1.7e-17` and zero wall violation: feasible to machine precision, as intended. The
LP start sums to `1.309` — far below where I expect to end up, but that is fine; it just gives SLSQP
a feasible interior point to climb from.

What do I expect this single run to reach? It will find *a* local optimum, and it will certainly
beat the grid: it can make the radii unequal and slide circles into the corners and edges the grid
wasted, and a linear objective with a competent local solver reliably tightens a random feasible
packing into a locally maximal one. But it will land in whatever basin its random initialization
happened to sit in, and the landscape of this nonconvex problem has many basins of quite different
quality. So I expect a comfortable clearance of the grid — somewhere in the high `2.5`s — but a
finish well short of the `2.636` frontier, because one initialization cannot explore the basin
structure.

Running it on seed `0`, SLSQP returns a packing summing to `2.5949`. Let me verify it before
trusting the number. The maximum pairwise overlap violation is `≈1.2e-14` and the maximum wall
violation is `0` — feasible to machine precision, comfortably inside the acceptance tolerance. The
radii are genuinely unequal, ranging from `0.068` to `0.149`, which is the irregularity the grid
could not produce. Against the grid's `2.5414` that is a gain of `0.054`; against the `2.636`
frontier it falls short by `0.041`. So the expectation holds quantitatively: clear of the grid,
short of the frontier.

Two checks pin down the lessons. First, the re-tightening: the LP-retightened sum exceeds the raw
SLSQP radii sum by `≈ −5.5e-14` — i.e. nothing measurable; SLSQP had already converged tight, so
on this run the re-tightening earns essentially zero. That is worth knowing honestly: it is cheap
insurance that happened not to be needed here, not a source of the result. Second, the single-start
limitation. If the ceiling on this rung were the optimizer, different random starts would converge
to about the same value; if it is the lone initialization, they should scatter. Re-running with
seeds `1, 2, 3, 7` gives `2.581, 2.581, 2.611, 2.592` — a spread of three hundredths, all above the
grid and all below the frontier, none reaching seed `0`'s `2.595` from above. That scatter is the
diagnosis made concrete: SLSQP is the right local engine, but a single random basin out of many is
what caps the score. The next rung has to wrap this exact solver in many restarts to find a better
basin than one draw can offer.
