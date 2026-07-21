The global-scalar control answered the question I posed and then drew the boundary of its own
usefulness. Plain temperature scaling posted worst-group ECE of 0.4844 / 0.4850 / 0.3098 (mean 0.4264),
against the subgroup method's 0.4938 / 0.4972 / 0.3138 (mean 0.4349). So the single global scalar is
*better* than the per-group machinery on the metric I am graded on — by about a point of worst-group ECE
on Adult and COMPAS — confirming my earlier suspicion: the per-group temperatures were leaking
small-group estimation noise into the worst group, and stripping them out helped. `subgroup_auroc` came
back identical (0.8590 / 0.8819 / 0.7486), as monotone-map invariance demanded, and Brier moved only
slightly (0.435 / 0.364 / 0.321). The control is clean.

The two feedback tables together decompose the Stein tradeoff. Using `best-subgroup ECE = worst_group_ece
− max_subgroup_gap`: for the group method the best cells were `0.058` (Adult), `0.356` (COMPAS), `0.119`
(Law School); for the global scalar, `0.066`, `0.365`, `0.142`. On every dataset the scalar's *worst*
cell improved (Adult `−0.009`, COMPAS `−0.012`, Law School `−0.004`) while its *best* cell got slightly
worse (Adult `+0.008`, COMPAS `+0.010`, Law School `+0.023`). That is the tradeoff caught in the act:
the per-group machinery spent its degrees of freedom driving the large well-determined cell down to
0.058 on Adult, and paid with noise on the small worst cell at 0.494. The scalar refuses to help the big
cell that extra bit and in exchange stops hurting the small cell — and because I am graded on the maximum
over cells, giving back `0.008` on the best to buy `0.009` on the worst is a straight win.

But the more important reading is the *level* the control bottomed out at. Even the lowest-variance map
I have leaves Adult and COMPAS near 0.48–0.49. A single scalar has exactly one degree of freedom
— it can soften uniformly but cannot *bend*. If the true score→probability distortion is steep in one
region and flat in another, or asymmetric between low and high scores, temperature scaling fits the best
single slope and leaves the rest as residual, which on the worst subgroup is the 0.484 I am still
reading. The best-subgroup decomposition says which datasets carry that kind of distortion: Adult's huge
internal spread — a cell at 0.066 next to one at 0.484 — is the signature of a distortion that changes
across the score range, exactly what one slope cannot chase and where a bend should pay; COMPAS's small
spread is uniform badness that smells more like irreducible shift. So the bottleneck the control exposes
is *shape*, not parameter count and not the per-group split.

The lesson from the first two rungs is firm: keep the low-variance, group-agnostic posture that just
paid off under shift, but give the *global* map a richer, still-monotone shape than a single division
can produce. What shape, and how much shape can I afford on a small shifted split?

What I actually believe about the true map, with confidence and independent of any dataset, is that it
is *non-decreasing* — not a guess but the one thing every run confirms: the classifier ranks well
(`subgroup_auroc` ~0.86/0.88/0.75), so a higher raw score means a genuinely higher, or at least not
lower, calibrated probability. That is the whole prior. A fixed richer parametric shape (add the
intercept back, fit a two-parameter sigmoid) commits to an S-curve I have no evidence the true
distortion is; histogram binning is shape-free but its bin count is a hyperparameter needing
cross-validation I cannot do on a shifted split. So instead of committing to one shape or to none,
restrict the map to *all* non-decreasing functions and ask for the best fit to the calibration labels
within that class. Monotonicity is weaker than any fixed parametric form — it lets the curve bend either
direction — but strong enough to be a real regularizer, forbidding the wiggling pure point estimation
would do. This is order-constrained regression.

Sort the calibration pairs by raw score, carrying binary targets `g_i ∈ {0,1}`; find non-decreasing
fitted values by weighted least squares, `minimize Σ_i w_i (g_i − ĝ_i)² s.t. ĝ_1 ≤ … ≤ ĝ_n`. Does that
produce a *probability*? Drop the constraint: the constant `c` minimizing `Σ_i w_i (g_i − c)²` is the
weighted mean `(Σ w_i g_i)/(Σ w_i)`, which for 0/1 targets is exactly the empirical positive rate. So
wherever the constraint forces the fit constant on a block, it takes that block's empirical positive
rate, which lives in `[0,1]` and *is* the calibrated probability I want — squared error is the choice
that makes each constant piece an honest probability. And cross-entropy has the same minimizer (both are
Bregman losses for the mean), so the order-restricted fit is identical under log-loss.

The structure hands me the algorithm. At the optimum any maximal run where `ĝ` is constant — a block,
free at both ends — takes its unconstrained minimizer, the weighted mean of its `g`'s; between blocks
`ĝ` strictly increases. So the solution is piecewise constant, each piece the weighted average of the
raw targets it spans, and the problem reduces to finding the partition. Pool-adjacent-violators computes
it in one pass: each point starts its own block with value `g_i`; whenever an adjacent pair violates
monotonicity (`v_L > v_R`) pool them into one block with weighted-mean value
`(w_L v_L + w_R v_R)/(w_L + w_R)`, back-merging left while pooling creates a fresh upstream violation —
the same unique optimum as the min-max envelope, `O(n)` after the `O(n log n)` sort. Run it on
`g = [0, 1, 0, 1, 1]`, unit weights: appending the third point pools `{1,0}` to `0.5`, and the final fit
is `[0, 0.5, 0.5, 1, 1]`. Every value is an empirical rate, and the two middle examples — distinct,
ordered raw scores — now carry the *same* `0.5`: the flat block collapsed their ordering. That is the
one place a monotone map preserves rank and PAVA does not, so I should expect `subgroup_auroc` to take a
small hit here that no temperature caused. The freedom PAVA grants slides between two poles:
already-sorted labels pool nothing and the fit is the raw step (as many degrees of freedom as levels,
most overfit-prone), perfectly reversed labels pool everything to the grand rate (one degree of
freedom). Real data sits between, and PAVA sets the effective degrees of freedom itself — which means I
cannot bound the variance in advance the way I could for the scalar.

The fitted intervals are a *binning whose boundaries are chosen by the data*: coarse where the
classifier ranks poorly, fine where it ranks well — binning's arbitrariness repaired with no bin count
to tune, which is why I prefer it to histogram binning on a split I cannot cross-validate. A test score
outside the calibration range carries no information, so the safe monotone choice is to clip the input
to the fitted score domain and bound the fitted values to `[0,1]` — exactly
`IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)`. The task is binary end to end, so one
isotonic map of the 0/1 label on the positive-class score suffices; `groups` is accepted and ignored,
keeping the posture that just won. The full module is in the answer.

The cost is the crux of whether this beats temperature scaling *under the shift*. The map is
non-parametric — its effective degrees of freedom are the number of blocks, a large fraction of `n` on a
well-ranked score versus the scalar's one — and variance is what the shift punishes. A block of `m`
points reports a rate with sampling standard deviation `√(p(1−p)/m) ≤ √(0.25/m)`; a ten-point block
carries a `±0.16` wobble in probability, stamped onto every test point in its range — more than five
times noisier per unit of the map than the global temperature's `±3%`, paid worst where the sample is
thinnest. The clip is the second half, and the shift activates it: if the calibration scores top out
near `0.95` but the shifted tail concentrates at `0.99`, every one of those test points is clipped back
to the terminal block's rate, and under a distribution shift there is no reason that rate matches the
tail — a whole slab of test mass systematically miscalibrated. That is the mechanism behind my worry
about Law School, whose split is the thinnest and whose scalar fit was already good enough that the
extra blocks have only noise to chase.

There is a deeper reason a group-agnostic reshaping can beat the group-aware fit on a per-group metric,
and it is the quiet thesis of this whole line of work. Subgroups can differ in the *shape* of their score→truth map
or merely in *where along the score axis their examples concentrate*. If a minority cell's scores pile
up at the extremes while a majority cell spreads across the middle, a single reshaping that pulls the
extremes back harder than the middle corrects the minority cell more — without ever being told which
cell is which — purely because the cells occupy different regions of the axis it bends. A per-group
temperature buys that same differentiation by fitting a number per cell and pays the small-cell
variance; a global bend buys it for free with one low-variance fit, as long as the between-group
difference is mostly *positional* rather than *shape*. Adult's decomposition is suggestive — a worst
cell at 0.484 next to a best at 0.066 is what a positional story predicts, one cell sitting where the
map is honest and another where it is blown out.

So the a-priori expectation splits by dataset. On Adult, where temperature scaling bottomed out at 0.484
with that tell-tale spread, the data-chosen binning should bend where the global slope could not and
clearly help. On Law School, where temperature scaling already reached 0.3098 because the distortion was
nearly a pure scale, the extra degrees of freedom have little real signal and a lot of room to overfit
the small shifted split, so I expect isotonic to do *worse* and to drag `subgroup_auroc` down a touch as
flat blocks collapse orderings. COMPAS I expect near the 0.48 the scalar posted — a marginal move either
way, since its uniform badness offers little non-uniform shape and it may lose the most AUROC to
pooling, having the most violations. If that is what the numbers say — isotonic winning where the
distortion is non-uniform, losing where it was already a pure scale, the *mean* roughly tying
temperature scaling while relocating where the error lives — then the next move writes itself: keep the
ability to bend but inside a tightly *parametric* family of only a few parameters, so I get the shape
without the variance.
