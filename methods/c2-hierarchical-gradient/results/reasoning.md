The coarse annealing gave me a `20`-piece profile at `0.8848`, just under the published `0.88922`, and the
feedback was clear about why it stops there: `20` pieces is too coarse a grid to render a sufficiently
flat-topped autoconvolution, and the cap on how flat that cap can get is the cap on `R`. The optimized
shape itself looks right — a tall spike plus a tapering shoulder, with several heights pinned at zero — so I
do not want to throw it away and search a long vector from scratch. I want to *keep this shape* and give it
more resolution, so the autoconvolution can develop the fine structure that a coarse vector simply cannot
represent. That is the idea of this rung: a hierarchical lift.

The lift itself is mechanical and, importantly, ratio-preserving. If I take the `20` heights and replace
each one by `k` copies of itself, I get a `20k`-piece step function that is *the same function* — same
graph, same autoconvolution, same `R`. So upscaling costs nothing and risks nothing; it just hands the
optimizer a finer canvas. The whole question is what to do *after* the lift, because at the upscaled point
the function is a fixed point of any symmetric refinement — every block of `k` copies is flat, and a search
that respects that flatness will not move. I have to break the block structure and let neighbouring pieces
within a block differ, which is where the new degrees of freedom — and the new gains — live.

For that I need a real optimizer on the heights, and at `N` in the hundreds a coordinate-wise annealing
that perturbs one height at a time is hopelessly slow: tens of thousands of acceptance tests to make one
coherent change across a shoulder. I want to move *all* the heights together along a good direction, which
means I need a gradient. So I will compute the gradient of `R` with respect to the height vector directly.
The obstruction is the `max` in the denominator — `||f*f||_inf = max_j L_j` is not differentiable at the
peak. I handle it the standard way: replace the hard max by a smooth softmax with a sharpness parameter
`β`, so the whole objective becomes differentiable, and then *anneal* `β` upward over the run — starting
soft, where the surrogate is smooth and the gradient points broadly toward better shapes, and ending sharp,
where the surrogate is a faithful stand-in for the true `max` and the optimum of the surrogate is the
optimum of `R`. The gradient of the surrogate factors cleanly through the autoconvolution: the chain rule
takes the derivative of `R` with respect to each node value `L_j`, and then each `L_j` depends on the
heights through the self-convolution, so the derivative with respect to the heights is itself a convolution
of the node-gradient with the height vector. Everything is `O(N log N)` with FFTs, so even a few thousand
gradient steps at `N = 500` is seconds, not minutes.

The optimizer I run on top of this gradient is Adam, for a specific reason. The heights span a wide dynamic
range — a spike near `28` times the smallest shoulder values in the eventual solution — and a plain
fixed-step gradient ascent either crawls on the large coordinates or blows up the small ones. Adam's
per-coordinate adaptive scaling is exactly the right tool: it normalizes each height's step by its own
recent gradient magnitude, so the spike and the thin shoulder advance on comparable terms. After every step
I clip to non-negative to stay legal, and I track the best *true* `R` ever seen (not the surrogate),
because the surrogate and the true ratio differ slightly and I want to return the genuinely best vector.

There is one more ingredient that turns out to matter: a small multiplicative perturbation right after each
upscale. The upscaled point is a degenerate plateau — flat blocks, zero gradient *within* a block in the
symmetric directions — and Adam started exactly there can sit still. A tiny random kick breaks the block
symmetry, gives the gradient something to grab, and lets the refinement carve asymmetric fine structure.
I keep the kick small so it does not destroy the good coarse shape; it just unsticks the plateau.

So the rung is a ladder of lifts: take the optimized `20`-piece profile, upscale `×5` to `100` and refine
with `β`-annealed Adam, upscale `×5` again to `500` and refine. Each lift is free; each refinement adds
fine structure the previous resolution could not hold. What do I expect? The first lift to `100` should
already clear the `20`-piece value, because `100` pieces can flatten the autoconvolution's cap further than
`20` can — I expect to land around `0.89`, into the AlphaEvolve `50`-step band (`0.89628`). The second lift
to `500` should add a bit more, into the high `0.89`s. I do *not* expect to reach the `0.9016` of the
Boyer–Li / Jaech–Joseph `~575`-step constructions from a few thousand Adam steps per level: those used far
more compute — Boyer–Li ran on the order of `10^6` gradient trajectories — and the last fraction of a
percent on this problem is notoriously expensive, because the optimal autoconvolution is a deliberately
irregular shape that only emerges after enormous refinement. So I expect this rung to confirm the lift
principle — that resolution plus a `β`-annealed gradient buys a steady climb from `0.8848` into the high
`0.89`s — and then to taper, limited by how many refinement steps I can afford at this resolution. The
taper is the opening for the endpoint: if more resolution and a longer, more carefully annealed gradient run
keep paying, then the move is to lift once more to a few thousand pieces and spend a long, kicked,
sharpening Adam schedule on it, pushing toward the `~0.9016` frontier of the published step-function
results.
