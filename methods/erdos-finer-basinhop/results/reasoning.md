The coarse rung landed at `0.381240` with two dozen cells, and the feedback said the cap was resolution —
the wide steps cannot resolve the fine structure of a near-optimal profile. So the natural next move is to
lift the optimized coarse profile to many more pieces and refine it there. But I want to be careful about
*how* I lift, because the coarse profile already encodes most of the gross structure and I do not want to
throw that away by re-randomizing.

The clean way to add resolution for free is to *upscale*: replace each coarse cell by several identical
finer cells of the same height. This is exactly the same step function — same `h`, same overlap, same
bound — just expressed on a finer grid. So the upscaled point starts at the same `C` the coarse profile
achieved, and now has many more degrees of freedom to spend. That is the move: take the best coarse vector,
upscale it, and refine on the finer grid. The upscaled point is a degenerate plateau of repeated blocks,
so it has the *same* value but a different, richer neighborhood — the optimizer can now break the blocks
apart and carve structure the coarse grid could not represent.

There is a wrinkle, though, and I have seen it before with upscaled initializations: the repeated-block
point is *flat* in many directions — perturbing two cells of the same original block in opposite ways often
leaves the overlap unchanged to first order — so a gradient or SLSQP step from the exact plateau can stall.
The remedy is a small kick: perturb the upscaled vector slightly (and re-project to feasibility) to break
the block symmetry and give the optimizer traction. A tiny multiplicative or additive jolt is enough; it
barely moves `C` but it unsticks the plateau.

The bigger question is the optimizer itself at this larger `n`. The annealed-SLSQP ladder from the coarse
rung still works — same smooth soft-max surrogate, same box-plus-equality constraints — but a single
SLSQP ladder from one (upscaled, kicked) start is not enough, because the finer landscape has *more* local
minima, not fewer. The extra degrees of freedom that let me carve better structure also create more ways
to get stuck. So I wrap the SLSQP ladder in a **basin-hopping** loop: solve the ladder to a local optimum,
then perturb the best-so-far vector and re-solve, accepting only improvements, repeating for a budget of
hops. Each hop is a constrained restart near the current best — far enough to jump basins, near enough to
keep the good gross structure. I shrink the perturbation as the hops proceed, so early hops explore and
late hops refine. This is precisely the "perturbation search + basin-hopping" recipe the agentic-search
record reports for this problem, and it is the right tool: a non-convex minimax where local descent is
cheap but the basins are many.

I also push the `β` ladder *sharper* at this resolution than at the coarse level. With more cells the
optimal profile is spikier and the overlap envelope has a finer set of near-binding shifts; a `β` that was
sharp enough at `24` cells is too soft at `120`, letting the surrogate's max sit below the true max so the
optimizer chases a slightly wrong objective. So I anneal `β` up further — into the thousands — in the late
hops, so the surrogate genuinely tracks the hard overlap I report.

How far to lift? I do not jump straight to the hundreds of cells the records use; I go to a middle
resolution — around a hundred-odd cells — for two reasons. First, I want to confirm that lifting actually
helps before spending a long run at high `n`. Second, this middle rung is where I learn the right
basin-hopping schedule — perturbation scale, number of hops, `β` ceiling — so the endpoint rung can spend
its budget on resolution rather than on re-tuning. I lift the coarse `24`-cell profile through an
intermediate multistart at a moderate `n` and then upscale-and-basin-hop to `~120` cells.

What do I expect? Resolution helped the literature monotonically, so I expect the finer profile to clear
the coarse `0.381240`, but only modestly — the coarse profile already captured most of the gross structure,
so the gain from `24 → 120` cells is the fine shaving, not a big jump. I expect to land a little below
`0.3812`, into the `0.38107`–`0.38108` band, closing most of the remaining distance to the Haugland /
AlphaEvolve landmarks (`~0.380924`–`0.380927`) but not yet reaching them. The honest number I report is the
true hard-max overlap of the best returned `~120`-cell vector.

The limitation this rung will expose is, again, resolution — but now at the *fine* end. A hundred-odd cells
resolves the profile better than two dozen, yet the published frontier lives at several hundred cells, where
the worst-overlap envelope can be made to tie across many shifts at a still-lower level. The endpoint rung
has to lift once more, to the `~600`-cell scale the records use, and refine there with a longer
basin-hopping budget plus an exact-minimax polish — pushing toward `~0.3809` and reading off, honestly, how
close a single bounded constructor gets to the AutoEvolver record `0.38086945`.
