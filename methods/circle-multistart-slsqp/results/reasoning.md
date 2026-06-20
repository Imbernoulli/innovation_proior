The single SLSQP run did its job and then named its own limitation: it lifted the sum from the
grid's `2.5414` to `2.5949`, clearly the right kind of move — unequal radii, circles pushed into
the corners — but it stopped at the local optimum of whatever basin its one random start happened
to fall into, well short of the `2.636` frontier. The optimizer is not the problem; the *single
initialization* is. This is a nonconvex landscape with many basins of very different quality, and
one draw samples one basin. The obvious fix is the standard one for nonconvex QCQPs: run the same
SLSQP from *many* random starts and keep the best feasible result.

Let me think about why this should work and what governs how well it works. Each random center
scatter, refined by SLSQP, converges to a distinct local optimum; the sum-of-radii values of those
optima are spread out, and the *maximum over many draws* is an order statistic that climbs as I add
starts. The more basins I sample, the better the best one I find — with diminishing returns, since
the high-quality basins are rarer and harder to hit by uniform random scattering. So this rung is
fundamentally a compute-for-quality trade: the only knobs are how many starts I can afford and how
diverse they are. There is no new algorithmic idea here beyond "restart," and that honesty matters
— this rung buys its improvement with breadth, not with a cleverer search.

The mechanics carry over unchanged from the previous rung. For each start I draw `26` centers
uniformly in a slightly inset square, set the initial radii to their LP optimum for that scatter
(so SLSQP begins feasible), run SLSQP jointly over centers and radii to a KKT point, and then
re-tighten the radii to their exact LP optimum for the final centers. I check feasibility against
the real constraints and, if the packing is legal and its sum exceeds the running best, I keep it.
At the end I return the best packing seen across all starts. I fix a single master seed so the
whole multi-start run is reproducible — the reported number is the deterministic output of that
seed, not a lucky draw I cannot reproduce.

A couple of choices I want to be deliberate about. First, the inset for the random centers: drawing
in `[0.04, 0.96]²` rather than the full square keeps the initial circles off the walls, which gives
the LP a feasible non-degenerate start and stops SLSQP from beginning at a constraint corner where
it can stall. Second, the per-start iteration budget: I do not need each SLSQP run to converge to
machine precision, because a start that is heading toward a mediocre basin is not worth polishing —
I want to spend the budget on *more starts* rather than over-refining each one. So I cap each run at
a moderate `maxiter` and rely on the LP re-tightening to recover the radii cleanly at the end. The
trade is: more, slightly coarser refinements beat fewer, perfectly-converged ones when the goal is
to find a good basin.

What do I expect? With on the order of a hundred random restarts I expect the best-of-many to clear
the single-start value comfortably and climb into the low `2.62`s — a real step toward the frontier,
bought entirely by sampling more basins. But I also expect it to *plateau* below `2.636`, and I can
see why in advance. Uniform random center scatters are a blunt way to seed: the basins that reach
the very top of the frontier correspond to specific irregular arrangements (a few large circles in
a particular pattern with the rest filling gaps), and those are a vanishingly small target for
uniform scattering, so adding more uniform starts yields ever-smaller gains — the order statistic
saturates. That saturation is exactly the limitation this rung will expose: pure random multi-start
finds good basins but not the *best* ones, because it has no structure and no memory. The next rung
has to do better than blind restarts — seed the search with structured layouts that resemble known
good packings, and, crucially, *exploit the best packing found so far* by perturbing it and
re-refining (iterated local search / perturbation chains) rather than throwing every start away.
That is where the frontier band lives, and where the random-restart plateau stops.
