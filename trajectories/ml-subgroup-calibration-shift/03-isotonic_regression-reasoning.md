The global-scalar control answered the question I posed and then drew the boundary of its own usefulness.
Plain temperature scaling posted worst-group ECE of 0.4844 / 0.4850 / 0.3098 (mean 0.4264), against the
subgroup method's 0.4938 / 0.4972 / 0.3138 (mean 0.4349). So the single global scalar is *better* than
the per-group machinery on the metric I am graded on — by about a point of worst-group ECE on Adult and
COMPAS and a hair on Law School — and that confirms the suspicion from the last rung: the per-group
temperatures were not buying worst-group calibration under this shift, they were leaking small-group
estimation noise into the worst group, and stripping them out helped. The `subgroup_auroc` came back
identical to the group method, 0.8590 / 0.8819 / 0.7486, exactly as the monotone-map invariance
demanded, so no bug. And the Brier moved only slightly (0.435 / 0.364 / 0.321, mean 0.373, basically the
group method's 0.370). Good — the control is clean.

Before I move on I want to squeeze the comparison for its mechanism, because the two feedback tables
together decompose the Stein tradeoff I predicted, and the decomposition is more instructive than the
headline. Use `best-subgroup ECE = worst_group_ece − max_subgroup_gap` on both runs. For the group
method the best cells were `0.494 − 0.436 = 0.058` (Adult), `0.497 − 0.142 = 0.356` (COMPAS), `0.314 −
0.195 = 0.119` (Law School); for the global scalar they are `0.484 − 0.419 = 0.066`, `0.485 − 0.120 =
0.365`, `0.310 − 0.168 = 0.142`. Now lay the two side by side. On every dataset the global scalar's
*worst* cell improved (Adult `−0.009`, COMPAS `−0.012`, Law School `−0.004`) while its *best* cell got
slightly *worse* (Adult `+0.008`, COMPAS `+0.010`, Law School `+0.023`). That is the Stein tradeoff
caught in the act: the per-group machinery spent its degrees of freedom on the large, well-determined
cell — driving its ECE down to 0.058 on Adult — and paid for it with noise on the small worst cell, which
sat at 0.494. The global scalar refuses to help the big cell that little bit extra and in exchange stops
hurting the small cell, and because I am graded on the maximum over cells, that is a straight win: it
gave back `0.008` on the best cell to buy `0.009` on the worst, and only the worst counts. The
prediction that per-group noise inflates a maximum-over-cells objective landed exactly, and in the
direction the max operator forces.

But look at the *level* the control bottomed out at, because that is the more important reading. Even the
lowest-variance map on this ladder leaves Adult and COMPAS worst-group ECE sitting near 0.48–0.49. A
single scalar that just divides every logit by one number cannot do better, because it has exactly one
degree of freedom — it can soften uniformly, but it cannot *bend*. If the true score→probability
distortion is not a uniform scale — if it is steep in one region of the score range and flat in another,
or asymmetric between the low and high scores — temperature scaling fits the best single slope and leaves
the rest as residual miscalibration, which on the worst subgroup is exactly the residual I am still
reading at 0.484. And the best-subgroup decomposition tells me which datasets carry that kind of
distortion. Adult's internal spread is huge — a cell at 0.066 next to one at 0.484 — the signature of a
distortion that changes across the score range, honest in one band, blown out in another; that is exactly
what one slope cannot chase, and exactly where a bend should pay. COMPAS's spread is small — everyone
between 0.365 and 0.485 — uniform badness that smells more like irreducible shift than like a shape a
richer map could remove. So the bottleneck the control exposes is *shape*, not parameter count and not
the per-group split, and it is likely to be a shape worth fixing on Adult and a shape that may resist
fixing on COMPAS.

The lesson from the first two rungs is therefore firm: keep the low-variance, group-agnostic posture that
just paid off under shift — do not go back to per-group fitting — but give the *global* map a richer,
still-monotone shape than a single division can produce. The question becomes: what is the right shape,
and how much shape can I afford on a small, shifted calibration split?

What do I actually believe about the true map, with confidence, independent of any dataset? I believe it
is *non-decreasing*. That belief is not a guess — it is the one thing every run has confirmed: the
classifier ranks well (`subgroup_auroc` ~0.86/0.88/0.75 on every method), so a higher raw score means a
genuinely higher, or at least not lower, calibrated probability. That is the whole prior. Now weigh the
ways to spend it. I could commit to a fixed richer parametric shape — add the intercept back and fit a
two-parameter sigmoid — but a sigmoid is a *committed* shape, an S-curve, and I have no evidence the true
distortion is an S; if it bends the other way the sigmoid cannot follow it, and I would have swapped one
rigid form for a slightly less rigid one without knowing it is the right form. I could go shape-free with
histogram binning, but its boundaries fall at arbitrary equal-count cuts and the bin count is a
hyperparameter that needs cross-validation I cannot do reliably on a shifted split — the same knob-tuning
I refused on the last rung. So instead of committing to one shape (a scalar division, or a sigmoid) or to
none (arbitrary bins), restrict the map to the class of *all* non-decreasing functions and ask for the
best fit to the calibration labels within that class. Monotonicity is weaker than any fixed parametric
form — it lets the curve bend however the data wants, in either direction — but strong enough to be a
real regularizer, because it forbids the wiggling that pure point estimation would do. This is
order-constrained regression. Write it down.

Sort the calibration pairs by raw score `p_1 ≤ p_2 ≤ … ≤ p_n`, carrying the binary targets
`g_1, …, g_n ∈ {0,1}`. I want fitted values `ĝ_1, …, ĝ_n`, non-decreasing and close to the labels, by
weighted least squares: `minimize Σ_i w_i (g_i − ĝ_i)²  s.t.  ĝ_1 ≤ … ≤ ĝ_n`. Before solving, check the
objective produces a *probability*, because I am fitting to 0/1 targets. Drop the constraint and ask what
single constant `c` minimizes `Σ_i w_i (g_i − c)²`: differentiate, `−2 Σ w_i (g_i − c) = 0`, so `c =
(Σ w_i g_i)/(Σ w_i)`, the weighted mean — and for 0/1 targets that is exactly the empirical positive
rate. So wherever the constraint forces the fit to be constant on a group of examples, the value it takes
is the empirical positive rate of that group, which lives in `[0,1]` and *is* the calibrated probability
I want. Squared error is not an arbitrary metric here — it is the choice that makes each constant piece
come out as an honest probability. And it is not even special: cross-entropy `Σ w_i [−g_i log c − (1−g_i)
log(1−c)]` has the same minimizer `c = (Σ w_i g_i)/(Σ w_i)` (both are Bregman losses for the mean), so the
order-restricted fit is identical under log-loss — I can solve the clean quadratic and get the log-loss
calibration for free.

How to solve the order-constrained least squares? The structure hands me the algorithm. At the optimum,
look at any maximal run of indices where `ĝ` is constant — a block. Within it the constraint holds with
equality (active); between blocks `ĝ` strictly increases (slack). A block free at both ends must take its
unconstrained minimizer, the weighted mean of its `g`'s. So the solution is piecewise constant, each
piece equal to the weighted average of the raw targets it spans, and the whole problem reduces to finding
the right partition into blocks — the values are then forced. The closed form is the min-max envelope
`ĝ_i = min_{ℓ≥i} max_{k≤ℓ} (Σ_{j=k}^ℓ w_j g_j)/(Σ_{j=k}^ℓ w_j)`, but computing it literally is `O(n²)`.
The streaming version is pool-adjacent-violators: walk left to right, each point its own block with value
`g_i`; whenever an adjacent pair violates monotonicity (`v_L > v_R`), pool them into one block with the
weighted-mean value `(w_L v_L + w_R v_R)/(w_L + w_R)` (the projection of the two-block optimum onto
`z_L ≤ z_R` lands on its boundary), and back-merge with the left neighbor while pooling creates a fresh
upstream violation. By a prefix induction — the stack is the exact isotonic fit for every processed
prefix, appending a point can only break the last boundary, and the pooling step is exactly the two-block
projection — this single pass computes the same unique optimum as the min-max formula, in `O(n)` after
the `O(n log n)` sort.

Let me run the algorithm by hand on a tiny case, because I want to see the flat block appear and confirm
the values stay probabilities. Take sorted scores with labels `g = [0, 1, 0, 1, 1]`, unit weights. Start
with the block `[0]`. Append `1`: `0 ≤ 1`, no violation, blocks `[0], [1]`. Append `0`: `1 > 0`,
violation, pool the last two into value `(1+0)/2 = 0.5`, blocks `[0], [0.5]`; check `0 ≤ 0.5`, fine.
Append `1`: `0.5 ≤ 1`, blocks `[0], [0.5], [1]`. Append `1`: `1 ≤ 1`, no violation, the block absorbs it,
blocks `[0], [0.5], [1,1]`. Fitted values `[0, 0.5, 0.5, 1, 1]`. Two things confirmed. First, every value
is a valid probability and each is the empirical positive rate of its block — the pooled `{1,0}` block is
`0.5`, exactly a rate. Second, the two middle examples, whose raw scores were distinct and ordered, now
carry the *same* calibrated `0.5`: the flat block has collapsed their ordering. That is the one place a
strictly-monotone map preserves rank and PAVA does not, and it is why I should expect `subgroup_auroc` to
take a small hit here that no temperature ever caused — AUROC counts orderings, and a flat block ties
examples that were previously ordered.

Two limiting cases pin down how much freedom PAVA actually grants, and they bracket the variance I am
buying. If the calibration labels already arrive sorted in score order — `g = [0, 0, 1, 1]` — then no
adjacent pair violates monotonicity, nothing pools, and the fit is the raw step `[0, 0, 1, 1]`: every
distinct label level is its own block, the map has as many degrees of freedom as the data has levels, and
isotonic is at its most flexible and most overfit-prone. If instead the labels arrive perfectly reversed
— `g = [1, 1, 0, 0]`, the ranking exactly wrong in this region — every pair violates, everything pools
into a single block, and the fit is the constant grand rate `0.5`: one degree of freedom, maximal
shrinkage to the base rate. Real data sits between these poles, and PAVA slides the effective degrees of
freedom along that axis automatically, block by block, spending resolution where the ranking is
trustworthy and collapsing to the base rate where it is not. That data-adaptivity is the appeal — but it
also means I cannot bound the variance in advance the way I could for the scalar, because the number of
blocks, and therefore the number of independently-estimated rates, is set by the calibration sample
itself rather than fixed at one.

The resulting intervals are a *binning whose boundaries are chosen by the data*: coarse where the
classifier ranks poorly (many violations pooled away), fine where it ranks well (no violations, blocks
stay small). That is binning's arbitrariness repaired, with no bin count to tune — which is exactly why I
prefer it to histogram binning on a split I cannot cross-validate. Turning the fitted blocks into a
function on a *new* score: aggregate tied calibration scores before fitting (one threshold, target the
weighted average of the tied labels), and after fitting interpolate linearly between adjacent threshold
values, which gives a continuous non-decreasing map. A test score outside the calibration range carries
no information, so the safe monotone choice is to clip the *input* to the fitted score domain — a score
below the smallest calibration score is evaluated at that smallest score, above the largest at the
largest — and bound the fitted values to `[0,1]`. That is precisely what
`IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)` does: it sorts by score, averages
duplicate scores with weights, runs PAVA, trims redundant thresholds, predicts by clipped linear
interpolation, and bounds the output. Since this task is binary end to end — the harness always hands the
calibrator a single positive-class probability vector — there is no multiclass one-against-all
reconciliation to build: I fit one isotonic map of the 0/1 label on the positive-class score and apply
it. `groups` is accepted and ignored; this stays group-agnostic, the posture that just won. The full
scaffold module is in the answer.

I should be honest about the cost, because it is the crux of whether this beats temperature scaling
*under the shift*, and I can put a rough number on it. The isotonic map is non-parametric — its effective
degrees of freedom are the number of blocks PAVA leaves, which on a well-ranked score can be a large
fraction of `n`, versus the single degree of freedom of a temperature. More free parameters means more
variance, and variance is exactly what the shift punishes: where the calibration sample is thin, each
small block is an empirical rate estimated from a handful of points, and where the shift pushes test
scores into a region the calibration set barely covered, the out-of-bounds clip drops a whole shifted
block onto one flat value — the clip is the *only* defense at the score-range edges where the shifted
tail is most likely to land. This cuts two ways and that is exactly what makes the prediction
interesting. Where the distortion is genuinely non-uniform, the freedom to bend should drop the
worst-group ECE below what the single global slope managed; where the calibration sample is small or the
shift dominates, the extra freedom can chase calibration-split noise and the clipped extrapolation can
land a whole shifted block on one flat value — a step backward.

Put a number on both halves of that cost. A block of `m` calibration points reports an empirical rate
whose sampling standard deviation is `√(p(1−p)/m) ≤ √(0.25/m)`; a block of ten points therefore carries
a calibration wobble of about `±0.16` in probability, and that wobble is stamped onto every *test* point
whose score falls in the block's range. Against the global temperature's `±3%` transfer of a single
number, an isotonic block on a thin split is more than five times noisier per unit of the map — that is
the price of the extra resolution, paid worst where the sample is thinnest. The clip is the second half
of the cost, and the shift is what activates it. Suppose the calibration scores top out near `0.95` but
the shifted test tail concentrates at `0.99`: every one of those test points is clipped back to the
fitted value at `0.95` and receives one flat calibrated number, chosen by whatever block happened to end
the calibration range. If that terminal block's rate does not match the tail — and under a distribution
shift there is no reason it should — a whole slab of test mass is systematically miscalibrated in one
direction, which is precisely how a non-parametric map that looked clean on the calibration split blows
up on the tail. That is the concrete mechanism behind my worry about Law School, whose split is the
thinnest and whose scalar fit was already good enough that the extra blocks have only noise to chase.

There is a deeper reason a group-agnostic reshaping can beat the group-aware fit on a per-group metric,
and it is worth stating because it is the quiet thesis of this whole ladder. Subgroups can differ in two
ways: in the *shape* of their score→truth map, or merely in *where along the score axis their examples
concentrate*. If a minority cell's scores pile up at the extremes while a majority cell's spread across
the middle, then a single reshaping of the score axis — one that pulls the extremes back harder than the
middle — corrects the minority cell more than the majority cell *without ever being told which cell is
which*, purely because the two cells occupy different regions of the axis it is bending. A per-group
temperature tries to buy that same differentiation by fitting a separate number per cell, and pays the
small-cell variance that just cost it the metric; a global bend buys it for free, with one low-variance
fit, as long as the between-group difference is mostly *positional* rather than *shape*. I do not know
that it is — but the Adult decomposition is suggestive: a worst cell at 0.484 next to a best at 0.066 is
what a positional story predicts, one cell sitting where the map is honest and another where it is blown
out. That is the bet isotonic is making — that the worst subgroup is worst because its scores live where
the global map is most wrong, so bending the global map there fixes the subgroup.

State it against the measured numbers. On Adult, where temperature scaling bottomed out at 0.484
worst-group ECE with that tell-tale internal spread, I expect isotonic to clearly *help* — the
data-chosen binning should bend the map where the global slope could not, and I would not be surprised to
see Adult's worst-group ECE fall well into the 0.3s, the biggest single gain on this ladder so far. That
is the optimistic side of non-parametric flexibility, and it is the whole reason to climb here. On Law
School I am genuinely worried: temperature scaling already got it to 0.3098, its *best* dataset, which
means the global scale correction nearly sufficed there, so the extra isotonic degrees of freedom have
little real signal to fit and a lot of room to overfit the small shifted split — I expect isotonic to do
*worse* on Law School, possibly back up into the 0.4s, and to drag its `subgroup_auroc` down a touch,
because PAVA's flat blocks collapse the ordering within a block just as the toy trace showed. COMPAS I
expect to stay near the 0.48 the scalar posted — a marginal move either way, since its uniform badness
offers little non-uniform shape to exploit and it may lose the most AUROC to pooling, having the most
violations to pool. So the falsifiable claim is mixed by design: isotonic should win on the dataset whose
distortion is non-uniform (Adult) and lose on the dataset whose distortion was already nearly a pure
scale (Law School), which would make its *mean* worst-group ECE only roughly tie temperature scaling
while trading where the error lives. If that is what the numbers say, the diagnosis for the next rung
writes itself: free-form monotonicity over-fits the small shifted splits, so I want to keep the ability
to bend but inside a tightly *parametric* family of only a few parameters, so I get the shape without the
variance. That is the move past isotonic.
