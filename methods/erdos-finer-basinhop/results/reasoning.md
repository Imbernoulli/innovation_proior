The coarse rung landed at `0.381240` with two dozen cells, and the feedback said the cap was resolution —
the wide steps cannot resolve the fine structure of a near-optimal profile. So the natural next move is to
lift the optimized coarse profile to many more pieces and refine it there. But I want to be careful about
*how* I lift, because the coarse profile already encodes most of the gross structure and I do not want to
throw that away by re-randomizing.

The cleanest way I can think of to add resolution without paying for it is to *upscale*: replace each
coarse cell by several identical finer cells of the same height. Intuitively that is the same step function
written on a finer grid — same `h`, same overlap — but I should not just assert that, because the evaluator
works on the discrete convolution `Σ_i v_i(1−v_{i−k})` and it is not obvious to me that repeating cells
leaves the discrete max-over-shifts untouched. Let me actually check it. I take a random feasible 24-cell
vector (projected so `Σv = 12`), compute `C`, then repeat each cell twice and three times and recompute:

```
C at n=24            : 0.5075993178236563
C at n=48 (repeat ×2): 0.5075993178236563
C at n=72 (repeat ×3): 0.5075993178236562
```

Identical to the last printed digit (the ×3 difference is rounding in the last place). So upscaling really
is a free no-op on the bound: the upscaled point starts at exactly the same `C` the coarse profile achieved
and now carries many more degrees of freedom to spend. That settles the lifting move — take the best coarse
vector, repeat each cell, and refine on the finer grid.

There is a wrinkle I want to look at before I trust this start. The upscaled point is a plateau of repeated
blocks, and I suspect it is *flat* in many directions: perturbing the two copies of one original block in
opposite ways keeps the block average fixed, so it might leave the overlap unchanged and give the optimizer
nothing to descend. Rather than guess, I take a profile, upscale ×2, and push the two copies of the first
block apart by `±ε`:

```
base C = 0.5418928281317413
ε=0.001 : C=0.5418928281  Δ=+0.00e+00
ε=0.01  : C=0.5418928281  Δ=+0.00e+00
ε=0.03  : C=0.5418928281  Δ=+0.00e+00
```

The hard-max does not move at all, even at `ε=0.03`. So the plateau really is exactly flat along these
intra-block antisymmetric directions — a gradient or SLSQP step launched from the exact repeated-block
point has no first-order signal there and can stall. The fix is a small kick: perturb the upscaled vector
slightly and re-project to feasibility, breaking the block symmetry so the optimizer has traction. A tiny
jolt is enough; the check above tells me it barely moves `C`, it just lifts the point off the flat ridge.

The bigger question is the optimizer itself at this larger `n`. The annealed-SLSQP ladder from the coarse
rung still works — same smooth soft-max surrogate, same box-plus-equality constraints — but I do not think a
single SLSQP ladder from one (upscaled, kicked) start will be enough, because the finer landscape has *more*
local minima, not fewer. The extra degrees of freedom that let me carve better structure also create more
ways to get stuck. The shape of the problem — a non-convex minimax where local descent is cheap but the
basins are many — is exactly the regime where you want to solve to a local optimum, perturb, and re-solve,
keeping only improvements. So I wrap the SLSQP ladder in a basin-hopping loop: solve the ladder to a local
optimum, then perturb the best-so-far vector and re-solve, accepting only improvements, repeating for a
budget of hops. Each hop is a constrained restart near the current best — far enough to jump basins, near
enough to keep the good gross structure. I shrink the perturbation as the hops proceed, so early hops
explore and late hops refine.

I also want to push the `β` ladder sharper at this resolution than at the coarse level, and I can justify
that with the surrogate gap rather than by assertion. The soft-max `_smooth_bound` overestimates the true
hard-max; the question is by how much, and whether sharpening `β` actually closes the gap at a profile like
the one I will be optimizing. I evaluate both at the upscaled 120-cell start:

```
beta=300   smooth=0.38142442  hard=0.38118973
beta=800   smooth=0.38127135  hard=0.38118973
beta=1800  smooth=0.38122188  hard=0.38118973
beta=3600  smooth=0.38120306  hard=0.38118973
```

At `β=300` the surrogate sits `2.3e-4` above the true overlap — a gap larger than the entire improvement I
am hoping to win at this rung — so an optimizer chasing that soft objective is chasing a slightly wrong
target. As `β` climbs to `3600` the gap falls to `1.3e-5`, and the surrogate genuinely hugs the hard
overlap I report. With more cells the optimal profile is spikier and the overlap envelope has a finer set of
near-binding shifts, so a `β` that was sharp enough at `24` cells is too soft here; the numbers above are
why I anneal `β` up further — into the thousands — in the late hops.

How far to lift? I do not jump straight to the hundreds of cells the records use; I go to a middle
resolution — around a hundred-odd cells — for two reasons. First, I want to confirm that lifting actually
helps before spending a long run at high `n`. Second, this middle rung is where I learn the right
basin-hopping schedule — perturbation scale, number of hops, `β` ceiling — so the endpoint rung can spend
its budget on resolution rather than on re-tuning. I lift the coarse `24`-cell profile through an
intermediate multistart at a moderate `n` and then upscale-and-basin-hop to `~120` cells.

What do I expect, and what do I actually get? Resolution helped the literature monotonically, so I expect
the finer profile to clear the coarse `0.381240`, but only modestly — the coarse profile already captured
most of the gross structure, so the gain from `24 → 120` cells should be a fine shaving, not a big jump. My
guess going in is the `0.38107`–`0.38108` band. So I run it. The multistart at `n=60` returns `0.3811897`,
and upscaling to `120` reproduces it to the digit (`0.3811897`), as the no-op check promised. Then the
basin-hopping best-so-far runs:

```
hop 0 : 0.381154
hop 5 : 0.381130
hop 6 : 0.381127
hop 7..20: 0.381127   (no further improvement)
```

Final `C = 0.3811266` at `n=120`, feasible (`|Σv − 60| = 0`). That is a real improvement over the coarse
`0.381240`, and it confirms lifting helps — but it lands at `0.38113`, a touch *above* the `0.38107`–`0.38108`
band I guessed. My expectation was a little too optimistic: `24 → 120` cells buys about `1.1e-3` off the
coarse number, not the `~1.6e-3` I was hoping for, and the hops stop improving after hop 6. The honest
number I report is this hard-max overlap of the best returned `~120`-cell vector, `0.3811266` — closer to
the Haugland / AlphaEvolve landmarks (`~0.380924`–`0.380927`) than the coarse rung was, but with real
distance still to go.

That gap is itself the limitation this rung exposes: resolution again, now at the fine end. A hundred-odd
cells resolves the profile better than two dozen — the `0.38113` I measured proves that — yet it sits well
short of the `~0.38092` landmarks, and the basin-hopping flatlining after six hops tells me the `120`-cell
representation, not the search budget, is what is binding. The published frontier lives at several hundred
cells, where the worst-overlap envelope can be made to tie across many shifts at a still-lower level. The
endpoint rung has to lift once more, to the `~600`-cell scale the records use, and refine there with a
longer basin-hopping budget plus an exact-minimax polish — pushing toward `~0.3809` and reading off,
honestly, how close a single bounded constructor gets to the AutoEvolver record `0.38086945`.
