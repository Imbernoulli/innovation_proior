Annealing reached `0.0356`, about `0.962` of the record, and then plateaued — and its own analysis
named why: near a good configuration several triangles are simultaneously near-tight, and a random
single-point move almost always grows one while shrinking another, so the minimum hovers just below the
optimum. The wall is no longer exploration — annealing found the right basin — it is *coordination*: I
need to nudge all the points so that every near-tight triangle grows at once. That is a smooth
optimization problem calling for a gradient, not a random kick, and a gradient moves all twenty-two
coordinates simultaneously — exactly the coordinated move the single-point move set structurally could
not make.

The obstacle is that the objective, `min` over `165` areas, is not differentiable: the `min` has a kink
wherever which triangle is smallest changes, and its gradient sees only the *single* currently-smallest
triangle, ignoring the others nearly as small. A gradient on the raw `min` would push one triangle up,
immediately make another the smallest, and chatter — the single-triangle myopia that defeated
annealing, in continuous clothes. What I want is a *smooth surrogate* for the minimum that feels all the
near-tight triangles at once.

The natural surrogate is the soft-minimum: `softmin(areas) = −(1/β)·log Σ_t exp(−β·area_t)`. As
`β → ∞` it converges to the true minimum; at finite `β` it is smooth and dominated by the smallest
areas but weighted across all near-tight ones. Differentiating shows the weighting exactly:
`∂softmin/∂area_t = exp(−β area_t) / Σ_s exp(−β area_s)` — the softmax weight `w_t`, large for tight
triangles, tiny for slack ones, summing to one. So maximizing the soft-min moves the configuration
along `Σ_t w_t ∇(area_t)`, a weighted average of the "grow triangle `t`" directions dominated by
exactly the near-tight triangles I need to inflate together. That is annealing's missing coordinated
push, written as one gradient step.

Each area is a smooth function of the coordinates, so the chain rule gives the whole gradient in closed
form with no autodiff. With `cross_t = (b_x−a_x)(c_y−a_y) − (c_x−a_x)(b_y−a_y)` and `area_t =
½|cross_t|`, the six partials of the cross product are `∂/∂a_x = b_y−c_y`, `∂/∂a_y = c_x−b_x`,
`∂/∂b_x = c_y−a_y`, `∂/∂b_y = a_x−c_x`, `∂/∂c_x = a_y−b_y`, `∂/∂c_y = b_x−a_x` (the `x`-partials sum to
zero, as translation-invariance of the cross product requires). Then
`∂area_t/∂(·) = ½ sign(cross_t)·∂cross_t/∂(·)`, and the full soft-min gradient
`Σ_t w_t·½ sign(cross_t)·∇cross_t` is assembled by scattering each triangle's contribution onto its
three points — twenty-two numbers from `165` short contributions.

One numerical point is fatal at the top of the `β` ladder: raw `exp(−β·area)` underflows. At
`β = 120000` and `area ≈ 0.035`, `exp(−4200)` is zero in double precision for *every* triangle, so the
normalizer is `0` and the weights are `0/0`. The log-sum-exp stabilization fixes it exactly: subtract
the minimum area first, `w_t = exp(−β(area_t − a_min))`, so the smallest triangle contributes
`exp(0) = 1` and the rest lie in `[0,1]`, no underflow, and the normalized weights are identical
because the common factor `exp(−β a_min)` cancels. The soft-min value is recovered as
`a_min − log(Σ w)/β`.

There is a kink where `cross = 0` — a degenerate triangle — but it is harmless here: near the record
every triangle has `|cross| ≈ 0.07`, nowhere near zero, so `sign(cross_t)` is locally constant and the
objective is genuinely smooth in the neighbourhood the optimizer explores.

Now the optimizer. A smooth objective with an analytic gradient and box constraints (coordinates in
`[0,1]`) is exactly the setting for L-BFGS-B: it builds curvature from the gradient history with no
Hessian to form, and handles the box by gradient projection and an active set. The box handling is
load-bearing, not incidental — the record configuration puts several points *on* the boundary, so the
optimizer must drive a coordinate to `0` or `1` and hold it there, which plain unconstrained descent
could not without clipping that fights the gradient. (A first-order method like Adam has no business on
a `22`-variable smooth problem where a quasi-Newton method converges in a handful of iterations.) I
feed it the *negative* soft-min and gradient and let it climb.

The sharpness `β` needs care, and the right move is to anneal it upward, for a reason the soft-min
formula makes quantitative. Relative to the true minimum, `softmin = a_min − (1/β) log Σ_t
exp(−β(a_t − a_min))`; if `k` triangles are essentially tied at the minimum, the sum is `≈ k` and
`softmin ≈ a_min − (log k)/β` — the surrogate *underestimates* the true minimum by `(log k)/β`. If I
set `β` huge from the start, that bias is negligible but the gradient is winner-take-all: the weight
ratio between two triangles differing by `δ` is `exp(βδ)`, so at `β = 120000` and `δ = 10⁻⁴` the ratio
is `exp(12) ≈ 1.6×10⁵` and the second triangle is invisible — from a configuration whose near-tight
triangles are not yet tied to within `1/β ≈ 8×10⁻⁶`, a huge-`β` gradient sees one triangle and
chatters, reinventing the raw `min`. If `β` is small the gradient spreads nicely across all near-tight
triangles but the bias `(log k)/β` is large and its maximizer is a soft proxy, not the hard objective.

A concrete area vector shows the transition. Take four triangles at `(0.0350, 0.0352, 0.0360, 0.0500)`.
At `β = 200`, `exp(−β·area)` normalizes to weights `(0.35, 0.34, 0.29, 0.018)`: the three near-tight
triangles share the gradient almost equally and a step inflates all three at once — the coordination I
want. Sharpen to `β = 120000` on the same vector: relative to `0.0350` the offsets are
`(0, 2×10⁻⁴, 10⁻³, 1.5×10⁻²)` and the weights collapse to `(1, ≈0, ≈0, ≈0)` — the `0.0352` triangle, a
mere `2×10⁻⁴` away, is frozen out because that dwarfs the window `1/β ≈ 8×10⁻⁶`. That is both the
winner-take-all failure and its cure: to keep the `0.0352` triangle in the gradient at high `β`, it
must first be grown to within `8×10⁻⁶` of the minimum, which is exactly what a broader `β = 200` stage
does before I sharpen.

So the ladder climbs `β` in stages, each warm-started from the last: `[200, 500, 1000, 3000, 8000,
20000, 50000, 120000]`, a `600×` total sharpening read as a shrinking *coordination window* `1/β`. At
`β = 200` the window is `0.005`, co-inflating everything within about `14%` of the current minimum into
a broad cluster grown toward near-equality; each raise narrows the window, and because the triangles
that entered the previous window are already tied finer than the new `1/β`, the winner-take-all
degeneracy never triggers. The final `β = 120000` makes the surrogate track the true minimum to within
`(log k)/β`; with a couple dozen triangles tied, `log k ≈ 3`, so the residual bias is `≈ 2.5×10⁻⁵`.
That is a falsifiable prediction: the polish should land not exactly on the record but within a few
parts in `10⁵` of it, the last sliver being surrogate/optimizer tolerance against a target a
floating-point objective can only approach.

After the ladder I always recompute the *exact* minimum over all `165` triangles — a genuine guard, not
a formality, because maximizing the soft-min is not identical to maximizing the hard min. At moderate
`β` a step can raise the weighted cluster's average while one triangle inside it dips, momentarily
lowering the true minimum; the ladder keeps that gap small but not zero. So I use the surrogate only to
*propose* configurations and the exact `min` to *judge* them, which also lets me rank fresh-restart
polishes against a seeded polish on the honest objective rather than on surrogate values a larger
cluster could inflate.

Seeding is as important as the polish, because soft-min ascent is *local* — it climbs to the nearest
good configuration, so it only reaches the record if it starts in the record's basin. The max-min
objective has many local maxima, each a different combinatorial "binding web," and most are inferior
arrangements with a floor below `1/27`; polish climbs to whatever basin its seed sits in. I do two
things. First, I run the same multi-restart annealing engine, heavier now (`48` restarts of `300,000`
steps), and polish each result — the heavier budget is justified *because* there is now a polish behind
it: each restart only needs to deliver a configuration at the *edge* of a good basin and the polish
extracts the rest, so deeper anneals and more restarts convert directly into final quality. Fresh SA
basins land a little short of the record, so I expect this cluster to sit a hair below the annealing
quality, near `0.96` of the record. Second, and load-bearing, I also polish the *rung-3 annealing best
at `0.0356`* itself: it already sat at `0.962`, right at the edge of the optimum's basin, near-tight
triangles and all — the single configuration in hand most likely to be *in* the record basin, so a
soft-min polish from there has the best chance of snapping onto `1/27`. So I predict a two-tier
outcome: fresh SA→polish restarts clustering near `0.96`, and `polish(rung-3 best)` reaching
essentially the record, within my predicted `~10⁻⁵` of `1/27`. This is the single-machine analogue of
the search-then-refine recipe — a stochastic global search to find the basin, a differentiable local
polish to land on it.

One more layer for robustness: basin-hopping around the best configuration found. After the polish, I
perturb it slightly (`σ = 0.03`, the annealing step scale) and re-polish, keeping the best exact
min-area over `80` rounds. A `0.03` kick is large enough to hop across the small barriers between
near-equivalent optima — mirror images and relabellings of a symmetric record configuration sit in
nearby basins — but small enough that the polish falls back into the record basin. I expect this to
*confirm* rather than improve: if `polish(rung-3 best)` already reached the basin floor, finding nothing
better is the useful signal that I am at the best of the cluster.

So overall I expect the polish to close essentially all the gap annealing left, pushing into the very
high `0.03`s and onto `1/27` to within a few parts in `10⁵`. This *matches*, and does not beat, the
record: `1/27` is the conjectured optimum at `n = 11`, and matching it to floating-point precision is
the ceiling of a single-machine search-plus-polish. And the predicted residual is itself a flag for
what comes next: if the polish
lands a hair short of `1/27`, whether that hair is geometry or mere floating-point tolerance is a
question a smooth optimizer on a float objective cannot answer, and only exact arithmetic can.
