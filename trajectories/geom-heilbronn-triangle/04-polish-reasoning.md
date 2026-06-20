Annealing reached `0.0356`, about `0.962` of the record, and then plateaued — and the feedback told
me precisely why. Near a good configuration several triangles are simultaneously near-tight, all
roughly equal to the current minimum, and a random single-point Gaussian move almost always grows
one of them while shrinking another, so the minimum hovers just below the optimum and refuses to
close the last few percent. The wall is not exploration anymore — annealing found the right basin —
it is *coordination*: I need to nudge all the points so that every near-tight triangle grows at
once. That is a smooth optimization problem, and it calls for a gradient, not a random kick.

The obstacle is that my objective, the minimum over `165` triangle areas, is not differentiable: the
`min` has a kink wherever which triangle is smallest changes, and its gradient (when it exists) only
sees the *single* currently-smallest triangle, ignoring all the others that are nearly as small. A
gradient on the raw `min` would push only the one tightest triangle up, immediately make a different
triangle the tightest, and chatter — exactly the single-triangle myopia that defeated the annealing
move. What I want instead is a *smooth surrogate* for the minimum that feels all the near-tight
triangles at once and whose gradient pushes them all up together.

The natural surrogate is the soft-minimum, the log-sum-exp form: with a sharpness parameter `β`,
define `softmin(areas) = -(1/β) · log Σ exp(-β · area_t)`. As `β → ∞` this converges to the true
minimum; at finite `β` it is a smooth, differentiable function that is dominated by the smallest
areas but weighted across *all* the near-tight ones. Its gradient with respect to each triangle's
area is the softmax weight `exp(-β·area_t) / Σ exp(-β·area_s)` — large for the tight triangles,
small for the slack ones — so maximizing the soft-min simultaneously inflates every triangle that is
near the current minimum, which is exactly the coordinated push annealing could not make. And each
triangle's area is itself a smooth (in fact bilinear) function of the point coordinates via the
cross product, so the chain rule gives me a clean analytic gradient of the soft-min with respect to
all `22` coordinates. I do not need autodiff machinery; the cross-product derivatives are short
closed forms (`∂cross/∂a = (b_y - c_y, c_x - b_x)` and its cyclic partners), and I can assemble the
full gradient by scattering each triangle's contribution onto its three points.

Now the optimizer. I have a smooth objective with an analytic gradient and box constraints
(coordinates in `[0,1]`), which is exactly the setting for a bounded quasi-Newton method like
L-BFGS-B. I hand it the negative soft-min and its gradient and let it climb. But the sharpness `β`
needs care, and the right move is to *anneal* it. If I set `β` huge from the start, the surrogate is
nearly the true `min` — sharp kinks, a near-degenerate gradient, and the optimizer stalls just like
a gradient on the raw `min` would. If I set `β` small, the surrogate is smooth and easy to optimize
but it is a *soft* minimum that sits well below the true minimum, so its optimum is not the optimum I
want. The fix is a ladder: start at a moderate `β` where the landscape is smooth, optimize to
convergence, then raise `β` and re-optimize from the result, repeating up to a very large `β`. Each
stage hands a good warm start to the next, and the final large-`β` stage makes the surrogate track
the true minimum closely, so the configuration it lands on genuinely maximizes the hard minimum, not
a soft proxy. After the ladder I always recompute the *exact* minimum over all `165` triangles — the
surrogate guides the search, but the reported number is the real thing.

The seeding matters as much as the polish. The soft-min gradient ascent is a *local* method: it
climbs to the nearest good configuration, so it only reaches the record if it starts in the record's
basin. That is precisely what the previous rungs are for. I seed the polish from heavy annealing —
the same multi-restart annealing engine as before, run with a larger per-restart budget and many
restarts — and polish each annealed configuration. Crucially I also polish the *best configuration
the previous rung already found*: the annealing best at `0.0356` is presumably sitting right at the
edge of the record's basin, near-tight triangles and all, so a soft-min polish from there has the
best chance of snapping onto the exact optimum. This is the small-scale analogue of the
search-plus-refine recipe that AlphaEvolve used on the sibling Heilbronn containers: a stochastic
global search to find the basin, then a smooth local polish to land on the record.

One more layer for robustness: basin-hopping around the best configuration found. After the polish
identifies a strong configuration, I perturb it slightly and re-polish, keeping the best exact
min-area — a few dozen rounds. This shakes the configuration across the small barriers between
near-equivalent local optima so the final answer is the best of a cluster, not a single polish that
happened to stop a hair early.

What do I expect? The polish should close most of the gap the annealing left open, because the
soft-min gradient does the one thing the random move could not — grow all the near-tight triangles
together. Seeded from the annealing best at `0.0356`, I expect it to push into the very high `0.03`s,
essentially onto the record `1/27 = 0.037037`. I would not be surprised to land within a fraction of
a percent of `1/27`, because the record configuration is a genuine local optimum of this very
objective and the polish is built to find exactly such optima. I am honest that this *matches*, and
does not *beat*, the record: `1/27` is the conjectured optimum at `n = 11`, and matching it to within
floating-point precision is the ceiling of what a single-machine search-plus-polish can do here.
Going *beyond* the tabulated records — as AlphaEvolve did for the triangle and convex-region
containers — requires the same kind of large-scale evolutionary search on far more compute, attacking
the cases where the tabulated value is *not* already optimal; at `n = 11` in the square, the value is
believed optimal, so the honest endpoint of this ladder is the record itself, reached and confirmed
by genuine measurement, with the proof of optimality standing as the still-open part of the problem.
