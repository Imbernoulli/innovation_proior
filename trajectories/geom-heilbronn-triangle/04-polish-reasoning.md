Annealing reached `0.0356`, about `0.962` of the record, and then plateaued — and the feedback told
me precisely why. Near a good configuration several triangles are simultaneously near-tight, all
roughly equal to the current minimum, and a random single-point Gaussian move almost always grows
one of them while shrinking another, so the minimum hovers just below the optimum and refuses to
close the last few percent. The wall is not exploration anymore — annealing found the right basin —
it is *coordination*: I need to nudge all the points so that every near-tight triangle grows at
once. That is a smooth optimization problem, and it calls for a gradient, not a random kick. The
previous rung's own analysis already named the obstruction: at a good configuration the binding
triangles are spread across many of the eleven points, so no single-point move can lift them all, and
the fraction of moves that could shrinks toward zero as the configuration improves. A gradient, by
contrast, moves *all twenty-two coordinates at once*, so it is exactly the kind of coordinated move
the annealing move set structurally could not make.

The obstacle is that my objective, the minimum over `165` triangle areas, is not differentiable: the
`min` has a kink wherever which triangle is smallest changes, and its gradient (where it exists) only
sees the *single* currently-smallest triangle, ignoring all the others that are nearly as small. A
gradient on the raw `min` would push only the one tightest triangle up, immediately make a different
triangle the tightest, and chatter between them — exactly the single-triangle myopia that defeated the
annealing move, just dressed up in continuous clothes. What I want instead is a *smooth surrogate* for
the minimum that feels all the near-tight triangles at once and whose gradient pushes them all up
together, so that "grow the worst triangle" becomes "grow the whole near-tight cluster."

The natural surrogate is the soft-minimum, the log-sum-exp form: with a sharpness parameter `β`,
define `softmin(areas) = −(1/β) · log Σ_t exp(−β · area_t)`. As `β → ∞` this converges to the true
minimum; at finite `β` it is a smooth, differentiable function dominated by the smallest areas but
weighted across *all* the near-tight ones. I can see the weighting exactly by differentiating. The
derivative of the soft-min with respect to one triangle's area is `∂softmin/∂area_t = −(1/β) ·
(−β exp(−β area_t)) / Σ_s exp(−β area_s) = exp(−β area_t) / Σ_s exp(−β area_s)` — the softmax weight
`w_t`. That weight is large for the tight triangles (small area, large `exp(−β area)`) and tiny for the
slack ones, and the weights sum to one. So maximizing the soft-min moves the configuration in the
direction `Σ_t w_t ∇(area_t)` — a *weighted average* of the individual "grow triangle `t`" directions,
dominated by exactly the near-tight triangles I need to inflate together. That is the coordinated push
annealing could not make, written as a single gradient step.

And each triangle's area is itself a smooth function of the coordinates, so the chain rule gives me the
whole gradient in closed form with no autodiff. Write `cross_t = (b_x−a_x)(c_y−a_y) − (c_x−a_x)(b_y−
a_y)` for the triple `(a,b,c)`, and `area_t = ½|cross_t|`. The derivatives of the cross product are
short, and I should write them all out and check them, because a sign slip here would send the
optimizer downhill. Differentiating `cross` in each of its six coordinate arguments: `∂cross/∂a_x =
−(c_y−a_y) + (b_y−a_y) = b_y − c_y`, and `∂cross/∂a_y = −(b_x−a_x) + (c_x−a_x) = c_x − b_x`; then
`∂cross/∂b_x = c_y − a_y` and `∂cross/∂b_y = −(c_x−a_x) = a_x − c_x`; and `∂cross/∂c_x = −(b_y−a_y) =
a_y − b_y` and `∂cross/∂c_y = b_x − a_x`. Six partials, one for each of the triangle's three points'
two coordinates. As a check they must sum correctly: the cross product is invariant under translating
all three points together, so `∂cross/∂a_x + ∂cross/∂b_x + ∂cross/∂c_x` should vanish — and indeed
`(b_y−c_y) + (c_y−a_y) + (a_y−b_y) = 0`, and likewise the `y` partials `(c_x−b_x) + (a_x−c_x) +
(b_x−a_x) = 0`. The translation-invariance check passes, so the signs are right. Then `∂area_t/∂(·) =
½ · sign(cross_t) · ∂cross_t/∂(·)`, and the full soft-min gradient is `Σ_t w_t · ½ sign(cross_t) ·
∇cross_t`, which I assemble by scattering each triangle's contribution (a factor `f = ½ · sign · w_t`
times the appropriate coordinate differences) onto its three points. Twenty-two numbers, assembled from
`165` short contributions — no black-box differentiation anywhere.

One numerical point I have to get right before any of this runs, because at the top of the `β` ladder it
is fatal: the raw `exp(−β·area)` underflows. At `β = 120000` and `area ≈ 0.035`, `exp(−β·area) =
exp(−4200)`, which is zero in double precision — and it is zero for *every* triangle, so the normalizing
sum is `0` and the weights are `0/0`, a nan gradient that kills the optimizer. The standard log-sum-exp
stabilization fixes it exactly: I subtract the minimum area first and compute `w_t = exp(−β(area_t −
a_min))`, so the smallest triangle contributes `exp(0) = 1` and every other a number in `[0,1]`, no
underflow anywhere, and the softmax weights `w_t / Σ w_s` are identical to the unstabilized ones because
the common factor `exp(−β·a_min)` cancels top and bottom. The soft-min value itself is then recovered as
`a_min − log(Σ w)/β`. This is why the earlier bias analysis was written relative to `a_min` in the first
place — the shift that makes the algebra transparent is the same shift that makes the arithmetic safe.

There is a kink to worry about — the absolute value in `area = ½|cross|` is non-differentiable where
`cross = 0`, i.e. at a degenerate triangle — but it is harmless here, and I can say why concretely.
Near the record every triangle has area around `0.035`, so `|cross| ≈ 0.07`, nowhere near zero; the
polish operates far from any degeneracy, `sign(cross_t)` is locally constant for every triangle, and
the objective is genuinely smooth in the neighbourhood the optimizer explores. The kink is a real
feature of the global landscape but it is not in the room where I am working. Before I trust the
assembled gradient I would finite-difference it: perturb each of the `22` coordinates by `h ≈ 10⁻⁶`,
form the central difference of the soft-min, and compare to the analytic value. I expect agreement to
about `10⁻⁸` — limited by the `O(h²)` truncation of the central difference and floating-point noise,
not by the formula — and if it were only good to `10⁻³` I would know I had a sign or a scatter-index
wrong. I will not skip that check; the whole method rests on the gradient being exactly this.

Now the optimizer. I have a smooth objective with an analytic gradient and box constraints
(coordinates in `[0,1]`), which is exactly the setting for a bounded quasi-Newton method like
L-BFGS-B. It builds curvature information from the history of gradients — no Hessian to form or
invert — and handles the `[0,1]²` box by gradient projection and an active set, so points that want to
sit on an edge of the square are held there correctly. That box handling is not incidental: the record
configuration puts several points *on* the boundary of the square, so the optimizer must be able to
drive a coordinate to `0` or `1` and hold it there, which a plain unconstrained method could not do
without ad-hoc clipping that fights the gradient. The alternatives I would otherwise reach for are worse
fits — a bare projected-gradient step wastes the curvature information that makes the ill-conditioned
near-tied landscape tractable; a general nonlinear-constrained solver like SLSQP is overkill for simple
box bounds and slower per iteration; a first-order method like Adam has no business on a `22`-variable
smooth problem where a quasi-Newton method converges in a handful of iterations. In `22` dimensions
L-BFGS-B is tiny; each stage converges in far fewer than the `4000` iterations I allow. I feed it the
*negative* soft-min (since the optimizer descends and I want to ascend) and the negative gradient, and
let it climb.

But the sharpness `β` needs care, and the right move is to *anneal it upward*, for a reason I can make
quantitative from the soft-min formula itself. Rewrite the surrogate relative to the true minimum:
`softmin = a_min − (1/β) log Σ_t exp(−β(a_t − a_min))`. The sum is at least `1` (the minimizing triangle
contributes `exp(0) = 1`) and, if `k` triangles are essentially tied at the minimum while the rest are
far above, the sum is `≈ k`, so `softmin ≈ a_min − (log k)/β`. The surrogate *underestimates* the true
minimum by about `(log k)/β`. That single expression tells me everything about the `β` ladder. If I set
`β` huge from the start, the bias `(log k)/β` is negligible but the gradient is nearly winner-take-all:
the weight ratio between two triangles differing in area by `δ` is `exp(βδ)`, so at `β = 120000` and
`δ = 10⁻⁴` the ratio is `exp(12) ≈ 1.6×10⁵` — the second triangle gets six millionths of the weight and
is invisible. From a generic configuration whose near-tight triangles are *not* yet tied to within
`1/β ≈ 8×10⁻⁶`, a huge-`β` gradient sees essentially one triangle, pushes it up, makes another the
smallest, and chatters — I have reinvented the raw `min` and its myopia. If instead I set `β` small, the
gradient spreads its weight nicely across all near-tight triangles (at `β = 200`, `δ = 10⁻⁴` gives ratio
`exp(0.02) ≈ 1.02`, near-equal weights), so it coordinates well — but now the bias `(log k)/β` is large,
the surrogate sits well below the true minimum, and its maximizer is the maximizer of a *soft* proxy,
not of the hard objective I actually care about.

Let me watch the weights on a concrete little area vector so the transition is not just abstract. Take
four triangles with areas `(0.0350, 0.0352, 0.0360, 0.0500)` — three near-tight and one slack. At
`β = 200`, `exp(−β·area)` gives `(9.1, 8.8, 7.5, 0.45)×10⁻⁴`, which normalize to weights `(0.35, 0.34,
0.29, 0.018)`: the three near-tight triangles share the gradient almost equally and the slack one is
nearly ignored, so a step inflates all three at once — exactly the coordination I want. Now sharpen to
`β = 120000` on the *same* vector: relative to the minimum `0.0350`, the offsets are `(0, 2×10⁻⁴,
10⁻³, 1.5×10⁻²)`, and `exp(−β·offset)` is `(1, e⁻²⁴, e⁻¹²⁰, ≈0) = (1, 3.8×10⁻¹¹, ≈0, ≈0)`. All the
weight collapses onto the single smallest triangle; the `0.0352` triangle, a mere `2×10⁻⁴` away, is
frozen out because `2×10⁻⁴` dwarfs the window `1/β ≈ 8×10⁻⁶`. That is the winner-take-all failure in
one line — and it also shows the cure: to keep the `0.0352` triangle *in* the gradient at `β = 120000`,
it must first be grown to within `8×10⁻⁶` of the minimum, which is precisely what the earlier, broader
`β = 200` stage does before I sharpen. The ladder is nothing but "tie them at coarse resolution, then
tighten the resolution."

The ladder resolves the tension by climbing `β` in stages, each warm-started from the last:
`[200, 500, 1000, 3000, 8000, 20000, 50000, 120000]`, roughly `2.5×` per stage over eight stages, a
`600×` total sharpening. The way to read it is as a shrinking *coordination window* `1/β`: at `β = 200`
the window is `1/200 = 0.005`, so the gradient co-inflates every triangle within `0.005` of the current
minimum — at a score of `0.035` that is everything within about `14%`, a broad cluster the optimizer can
grow together into near-equality. Once those are nearly tied, I raise `β` so the window narrows —
`0.002`, `0.001`, ... down to `1/120000 ≈ 8×10⁻⁶` — and at each new sharpness the triangles that
entered the previous window are already tied to finer than the new `1/β`, so the winner-take-all
degeneracy never triggers; the gradient stays spread over the genuinely-binding set and keeps
coordinating. Each stage hands a well-tied warm start to the next, and the final `β = 120000` makes the
surrogate track the true minimum to within a bias of about `(log k)/β`. With a couple dozen triangles
near-tied, `log k ≈ 3`, so the residual bias is `≈ 3/120000 ≈ 2.5×10⁻⁵`. That is a concrete,
falsifiable prediction: I expect the polish to land not *exactly* on the record but within a few parts
in `10⁵` of it — the last sliver of gap being optimizer/surrogate tolerance against a target the
floating-point objective can only approach, not a geometric deficiency. After the ladder I always
recompute the *exact* minimum over all `165` triangles; the surrogate guides the search, but the number
I report is the hard `min`, never the soft one.

That exact recompute is a genuine guard, not a formality, because maximizing the soft-min is not
identical to maximizing the hard min and the two can briefly disagree. At moderate `β` the gradient
grows the *weighted cluster* of near-tight triangles, and it is possible for a step to raise the
soft-min — lifting the cluster's weighted average — while one particular triangle inside the cluster
actually dips, momentarily *lowering* the true minimum. The `β` ladder keeps that gap small (the higher
`β` climbs, the closer the weighted cluster is to the single true minimum), but "small" is not "zero,"
so I cannot trust the surrogate's own value as the score. Recomputing the honest `min` over all `165`
triples after every polish, and again after every basin-hop, means the configuration I finally return is
selected on the real objective — I use the smooth surrogate only to *propose* good configurations and
the exact `min` to *judge* them. This is also what lets me compare fresh-restart polishes against the
`polish(rung-3 best)` on an even footing: every candidate is ranked by its true minimum triangle, so the
two-tier outcome I predict is a comparison of hard scores, not of surrogate values that a larger cluster
could inflate.

The seeding matters as much as the polish, and here the previous rungs pay off directly, because the
soft-min ascent is *local* — it climbs to the nearest good configuration, so it only reaches the record
if it starts in the record's basin. I do two things. First, I run the same multi-restart annealing
engine as before, heavier now (`48` restarts of `300{,}000` steps versus the previous `30 × 200{,}000`),
and polish each annealed result. The heavier budget is justified precisely *because* there is now a
polish behind it: in the previous rung a restart's value was capped at whatever its own random single-
point moves could reach, so spending more steps had diminishing returns against the plateau; now each
restart only needs to deliver a configuration sitting at the *edge* of a good basin, and the polish
extracts the rest, so a deeper `300{,}000`-step anneal (better-seated basin edge) and more restarts
(`48`, more independent basins sampled, more chances one is the record basin) both convert directly into
final quality. These fresh restarts find their own basins, and since a fresh SA basin lands a little
short of the record, I expect the fresh-restart-then-polish cluster to sit around the annealing quality
— call it `~0.958` of the record — with the soft-min closing the *coordination* gap within each basin
but not teleporting between basins. Second, and this is the load-bearing choice, I also polish the *rung-3
annealing best at `0.0356`* itself: that configuration already sat at `0.962` of the record, right at the
edge of the optimum's basin, near-tight triangles and all — it is the single configuration in hand most
likely to be *in* the record basin, so a soft-min polish from there has the best chance of snapping onto
`1/27`. I therefore predict a two-tier outcome: the fresh SA→polish restarts clustering near `0.958`,
and `polish(rung-3 best)` reaching essentially the record, up in the very high `0.03`s within my
predicted `~10⁻⁵` of `1/27`. This is the single-machine analogue of the search-plus-refine recipe that
scales up to a global evolutionary search feeding a smooth local polish: a stochastic global search to
find the basin, then a differentiable local polish to land on the record.

Why do I expect the fresh restarts to cluster *below* the record while the seeded polish reaches it?
Because the max-min objective has many distinct local maxima, not one. Each corresponds to a different
combinatorial "binding web" — a different pattern of which triangles end up tight — and most of those
webs are genuinely inferior arrangements with a floor below `1/27`. Polish is local: it climbs to the
floor of whatever basin its seed sits in, so a fresh SA restart that landed in an inferior basin gets
polished up to that basin's inferior floor, not to the record. Only a seed already inside the record
basin polishes onto `1/27`. Fresh SA has some modest probability of hitting the record basin on any one
restart, so across `48` of them the *best* fresh-restart-polish reflects the deepest basin sampled —
which I expect to be a near-record basin around `0.958`, not the record itself, because the record basin
is a small target for a random start. The rung-3 annealing best, by contrast, is a configuration that
the previous rung *already* walked to `0.962` — it is, empirically, sitting at the lip of the record
basin — so polishing it is not a gamble on hitting the basin but a near-certainty of descending into it.
That asymmetry is the whole reason I carry the previous rung's best forward as an explicit seed rather
than trusting fresh restarts to rediscover it.

One more layer for robustness: basin-hopping around the best configuration found. After the polish
identifies a strong configuration, I perturb it slightly (`σ = 0.03`, about the annealing step scale)
and re-polish, keeping the best exact min-area over `80` rounds. A `0.03` kick is large enough to hop
across the small barriers between near-equivalent local optima — the record configuration has
symmetries, so mirror images and relabellings sit in nearby basins — but small enough that the polish
falls back into the record basin rather than wandering off. I expect this to *confirm* rather than
improve: if `polish(rung-3 best)` already reached the basin floor, basin-hopping should find no better
configuration, which is itself the useful signal that I am at the best of the cluster and not stopped a
hair early in one arbitrary member of it.

What do I expect overall? The polish should close essentially all of the gap the annealing left open,
because the soft-min gradient does the one thing the random move could not — grow all the near-tight
triangles together — and the seeding puts it in the right basin. Seeded from the annealing best at
`0.0356`, I expect it to push into the very high `0.03`s, onto the record `1/27 = 0.037037` to within a
few parts in `10⁵`. I am honest that this *matches*, and does not *beat*, the record: `1/27` is the
conjectured optimum at `n = 11` in the square, and matching it to within floating-point precision is the
ceiling of what a single-machine search-plus-polish can do here. Going *beyond* tabulated records
requires the same kind of large-scale evolutionary search on far more compute, attacking cases where the
tabulated value is *not* already optimal; at `n = 11` in the square the value is believed optimal, so
the honest endpoint of this ladder is the record itself, reached and confirmed by genuine measurement,
with the proof of optimality standing as the still-open part of the problem. And the predicted residual
`~10⁻⁵` gap is itself a flag for the rung after this one: if the polish lands a hair short of `1/27`, the
question becomes whether that hair is geometry or mere floating-point tolerance — a question a smooth
optimizer on a float objective cannot answer, and only exact arithmetic can.
