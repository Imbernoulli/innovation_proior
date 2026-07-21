The single SLSQP run did its job and then named its own limitation: it lifted the sum from the
grid's `2.5414` to `2.5949`, clearly the right kind of move — unequal radii, circles pushed into
the corners — but it stopped at the local optimum of whatever basin its one random start happened
to fall into, well short of the `2.636` frontier. The optimizer is not the problem; the *single
initialization* is. This is a nonconvex landscape with many basins of very different quality, and
one draw samples one basin.

So the lever to pull is the start, not the solver. Whether throwing more random starts at the same
SLSQP actually moves the best, and by how much, I can only answer by running it — but first I want
to pin down where the per-start improvement comes from, since that decides what each restart
actually has to repeat.

A tempting cheaper idea: the radii subproblem is an exact LP for fixed centers, convex and fast.
Maybe I don't need SLSQP at all — just scatter centers many times, solve the LP for each, and keep
the best radii sum. Here is what the LP alone buys on a raw uniform scatter. On `15` random
scatters the LP-optimal radii sum averages about `1.30` (individual draws `1.16`–`1.46`), while
running SLSQP from the *same* scatters lifts each to roughly `2.4`–`2.6`:

```
scatter LP 1.36 -> SLSQP 2.61   (+1.24)
scatter LP 1.35 -> SLSQP 2.56   (+1.21)
scatter LP 1.46 -> SLSQP 2.59   (+1.13)
scatter LP 1.24 -> SLSQP 2.58   (+1.34)
scatter LP 1.16 -> SLSQP 2.61   (+1.45)
```

That settles it: the radii LP is doing almost none of the work. A raw uniform scatter is a terrible
packing — circles scattered without regard to each other leave huge slack — and tightening radii on
fixed bad centers cannot recover it; the gain of `+1.2` to `+1.4` per start comes entirely from
SLSQP *moving the centers* into a coherent arrangement. So LP-per-scatter is out; each start has to
be a full center+radius refinement. The LP keeps its role, but as a sub-step: a feasible warm start
for the radii and a clean final re-tightening, not the search itself.

The mechanics then carry over unchanged from the single-start run. For each start I draw `26` centers
uniformly in a slightly inset square, set the initial radii to their LP optimum for that scatter
(so SLSQP begins feasible), run SLSQP jointly over centers and radii to a KKT point, then
re-tighten the radii to their exact LP optimum for the final centers. I check feasibility against
the real constraints and, if the packing is legal and its sum exceeds the running best, I keep it.
At the end I return the best packing seen across all starts. I fix a single master seed so the whole
multi-start run is reproducible — the reported number is the deterministic output of that seed, not
a lucky draw I cannot reproduce.

Two choices here matter enough to be deliberate about. First, the inset for the random centers: drawing
in `[0.04, 0.96]²` rather than the full square keeps the initial circles off the walls, which gives
the LP a feasible non-degenerate start and stops SLSQP from beginning at a constraint corner where
it can stall. Second, the per-start iteration budget: I do not need each SLSQP run to converge to
machine precision, because a start heading toward a mediocre basin is not worth polishing — I want
to spend the budget on *more starts* rather than over-refining each one. So I cap each run at a
moderate `maxiter=250` and rely on the LP re-tightening to recover the radii cleanly at the end. The
bet is that more, slightly coarser refinements beat fewer perfectly-converged ones when the goal is
to find a good basin.

Now the question I can't reason my way past: does best-of-many actually climb, and where does it
land? I run `K=120` starts from the master seed `12345` and record every per-start sum and the
running best. All `120` refinements come back feasible. The per-start sums are spread out — mean
`2.519`, median `2.580`, the worst a degenerate collapse to `0.0`, the best `2.622`. That spread is
exactly what multi-start feeds on; it is the maximum of these draws that matters, not the average.
Tracking the running best as starts accumulate:

```
after   1 start   2.5914
after  10 starts  2.6062
after  20 starts  2.6077
after  60 starts  2.6116
after 100 starts  2.6221
after 120 starts  2.6221
```

So the answer to the first half is yes: from the single-start `~2.59` the best-of-many climbs past
`2.61` and reaches `2.62210204667146` — a real step toward the frontier, bought entirely by
sampling more basins. The final packing checks out as legal: its worst constraint violation is
`3.2e-15`, machine-noise level, so the sum is honest and not riding a tolerance. Its radii are
genuinely unequal — sorted, they run from `0.140` down to `0.066`, a handful of large circles with
the rest filling gaps — which is the irregular structure I'd expect a good basin to have.

The second half of the answer is the more interesting one, and it is visible in the same numbers:
the running-best table plateaus. The best is flat from start `100` to start `120` — the last `20`
refinements add nothing. The reason is in the
distribution: of the `120` feasible starts, `19` cleared `2.60`, only `4` cleared `2.61`, and
exactly `1` cleared `2.62`. The top basins are that rare under uniform scattering, so each
additional uniform start is overwhelmingly likely to land somewhere I've already matched or beaten.
The order statistic is saturating in front of me — the gap from `2.622` to the `2.636` frontier is
not going to close by adding more of the same blind draws, because the basins that reach the top
correspond to specific irregular arrangements that uniform scatter hits with vanishing probability.

That is the limitation this plateau exposes: pure random multi-start finds good basins but not the
*best* ones, because it has no structure and no memory. Doing better than blind restarts means
seeding the search with structured layouts that resemble known good packings and, crucially,
*exploiting the best packing found so far* by perturbing it and re-refining — iterated local search
rather than throwing every start away. That is the direction the frontier band has to come from, and
it is this random-restart plateau that motivates moving there next.
