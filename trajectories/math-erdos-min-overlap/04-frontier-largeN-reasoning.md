The hierarchical lifts kept paying — `0.381240` at `24` cells, `0.381076` at `120` — and each rung
confirmed the same recipe: upscale the optimized profile for free, kick to break the block plateau, refine
with annealed-soft-max search keeping the best true overlap. The feedback said the cap was still
resolution, that the published frontier lives at several hundred cells, so the natural next move is to lift
once more, to the `~600`-cell scale the records use (AlphaEvolve's `95`-step `0.380924`, AutoEvolver's
`~600`-step record `0.38086945`), and grind there with a much longer, sharper refinement plus an exact
minimax polish. That band — a few hundred to several hundred cells — is exactly where careful gradient
refinement should approach the step-function frontier, so that is the target for this endpoint rung: push
to and, if I can, through the `0.3810` mark toward `~0.3809`, knowing the absolute record was bought with
orders of magnitude more compute than I have here.

I take the optimized `120`-cell profile and upscale `×5` to `600`. As before the upscale is free — same
step function, same overlap, same `C` — and as before the upscaled point is a degenerate plateau of
repeated blocks, so I kick it with a small perturbation to break the block symmetry and give the optimizer
traction. Then I refine. But here I have to confront a scaling wall I have been deferring, and it changes
the whole shape of the endpoint.

The wall is the solver. The coarse and middle rungs leaned on constrained SLSQP against the soft-max
surrogate, and that was the right tool at two dozen and a hundred-odd cells. At `600` cells SLSQP becomes
the bottleneck: its internal quadratic program is super-linear in the number of variables, and with `600`
heights plus the equality constraint a single annealed-SLSQP ladder takes a couple of minutes — and worse,
from a good starting point it barely moves, because the surrogate optimum it chases has essentially
coincided with where the profile already sits. So leaning on SLSQP at this resolution buys almost nothing
for a large cost. I have to switch the large-`n` optimizer to something that scales: a projected-gradient
method on the smooth soft-max bound with an *analytic* gradient, run as `β`-annealed Adam. Each step is a
correlation plus a gradient assembly — cheap and scalable — so I can afford tens of thousands of steps at
`600` cells in seconds rather than minutes, exactly the move the analogous step-function ladder used for
the autocorrelation problem. So the endpoint's optimizer is annealed-Adam on the analytic soft-max
gradient, with periodic kicks during the long run to keep it out of shallow traps, and a final exact
subgradient polish on the *true* minimax to clean up the binding shifts the surrogate slightly mis-tracks.

Three things change at `600` cells. First, the `β` schedule has to be pushed *much* sharper. The soft-max
stand-in for the worst overlap is only faithful to the hard `max` when `β` is large relative to the spread
of the cross-correlation values; at `600` cells with a spiky near-binary profile the binding shifts are
many and closely tied, and a `β` that was sharp enough at `120` lets the surrogate's peak sit below the
true peak, so the optimizer chases a slightly wrong objective and the true overlap lags. So I anneal `β`
into the thousands and beyond in the late passes, so the surrogate genuinely tracks the hard worst-overlap.
Second, periodic kicks during the long grind, not just at the upscale: over many passes the optimizer can
settle into a shallow basin, and a small shrinking kick between passes acts like a mild restart that keeps
it exploring while the late phase stays pure refinement. Third, the exact subgradient polish at the end:
the soft-max and the true `max` diverge slightly, so I finish by descending the genuine minimax —
distributing a step across the active (near-worst) shifts — to shave the last fraction the surrogate leaves
on the table. Throughout I keep the best *true* overlap ever seen, because the surrogate-best and the
true-best are not the same vector and I want the genuinely best one.

What do I expect? The first reorganizing passes should hold the `120`-cell value — upscaling is free and
the lift only adds freedom — and the long sharp-`β` grind is where any endpoint gain comes from, carving
the finer irregular structure the published constructions rely on. I expect to push toward and ideally
through `0.3810`, into the `0.38087`–`0.3809` band that the AlphaEvolve / AutoEvolver step functions
occupy. I am honest, though, that I will likely land a *hair above* the records: the AutoEvolver record
`0.38086945` was found by an evolutionary coding-agent search run to `n=750` over `~12` hours, and
AlphaEvolve's `0.380924` likewise came from a large evolutionary search; the gap from the low `0.3810`s
to `0.38087` is the part of the problem that a single bounded gradient-refinement run on a few hundred
cells does not close. So the endpoint of this ladder is the step-function frontier that careful hierarchical
gradient refinement can actually reach — close to, and reading honestly against, the best published
results — with the AutoEvolver / AlphaEvolve records standing just above as the still-open distance, and
White's provable lower bound `0.379005` below as the floor the true constant cannot cross. The honest
measured number I report is the one the evaluator returns on the returned `600`-cell vector, and the
remaining gap to `0.38087` is the measure of how tightly squeezed — and how much still-contested at the
fifth decimal — this seventy-year-old problem remains.
