The global-scalar control answered the question I posed and then drew the boundary of its own usefulness.
Plain temperature scaling posted worst-group ECE of 0.4844 / 0.4850 / 0.3098 (mean 0.4264), against the
subgroup method's 0.4938 / 0.4972 / 0.3138 (mean 0.4349). So the single global scalar is *better* than
the per-group machinery on the metric I am graded on — by about a point of worst-group ECE on Adult and
COMPAS and a hair on Law School — and that confirms the suspicion from the last rung: the per-group
temperatures were not buying worst-group calibration under this shift, they were leaking small-group
estimation noise into the worst group, and stripping them out helped. The `subgroup_auroc` came back
identical to the group method, 0.8590 / 0.8819 / 0.7486, exactly as the monotone-map invariance
demanded, so no bug. And the Brier moved only slightly (0.435 / 0.364 / 0.321, mean 0.373, basically the
group method's 0.370). Good — the control is clean. But look at the *level* it bottomed out at: even the
lowest-variance map on this ladder leaves Adult and COMPAS worst-group ECE sitting near 0.48–0.49. A
single scalar that just divides every logit by 0.48 worth of over-confidence cannot do better, because
it has exactly one degree of freedom — it can soften uniformly, but it cannot *bend*. If the true
score→probability distortion on Adult is not a uniform scale — if it is steep in one region of the score
range and flat in another, or asymmetric between the low and high scores — temperature scaling fits the
best single slope and leaves the rest as residual miscalibration, which on the worst subgroup is exactly
the residual I am still reading at 0.484.

So the bottleneck the control exposes is *shape*, not parameter count and not the per-group split. The
lesson from the first two rungs is firm: keep the low-variance, group-agnostic posture that just paid off
under shift — do not go back to per-group fitting — but give the *global* map a richer, still-monotone
shape than a single division can produce. The question becomes: what is the right shape, and how much
shape can I afford on a small, shifted calibration split?

What do I actually believe about the true map, with confidence, independent of any dataset? I believe it
is *non-decreasing*. That belief is not a guess — it is the one thing every run has confirmed: the
classifier ranks well (`subgroup_auroc` ~0.86/0.88/0.75 on every method), so a higher raw score means a
genuinely higher, or at least not lower, calibrated probability. That is the whole prior. Temperature
scaling honored monotonicity but spent it on a single global slope; binning would be shape-free but its
boundaries fall at arbitrary equal-count cuts and the bin count needs cross-validation I cannot do
reliably on a shifted split. So instead of committing to one shape (a scalar division) or to none
(arbitrary bins), restrict the map to the class of *all* non-decreasing functions and ask for the best
fit to the calibration labels within that class. Monotonicity is weaker than a fixed parametric form —
it lets the curve bend however the data wants — but strong enough to be a real regularizer, because it
forbids the wiggling that pure point estimation would do. This is order-constrained regression. Write it
down.

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
the `O(n log n)` sort. The resulting intervals are a *binning whose boundaries are chosen by the data*:
coarse where the classifier ranks poorly (many violations pooled away), fine where it ranks well (no
violations, blocks stay small). That is binning's arbitrariness repaired, with no bin count to tune —
which is exactly why I prefer it to histogram binning on a split I cannot cross-validate.

Turning the fitted blocks into a function on a *new* score: aggregate tied calibration scores before
fitting (one threshold, target the weighted average of the tied labels), and after fitting interpolate
linearly between adjacent threshold values, which gives a continuous non-decreasing map. A test score
outside the calibration range carries no information, so the safe monotone choice is to clip the *input*
to the fitted score domain — a score below the smallest calibration score is evaluated at that smallest
score, above the largest at the largest — and bound the fitted values to `[0,1]`. That is precisely what
`IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)` does: it sorts by score, averages
duplicate scores with weights, runs PAVA, trims redundant thresholds, predicts by clipped linear
interpolation, and bounds the output. Since this task is binary end to end — the harness always hands the
calibrator a single positive-class probability vector — there is no multiclass one-against-all
reconciliation to build: I fit one isotonic map of the 0/1 label on the positive-class score and apply
it. `groups` is accepted and ignored; this stays group-agnostic, the posture that just won. The full
scaffold module is in the answer.

I should be honest about the cost, because it is the crux of whether this beats temperature scaling
*under the shift*. The isotonic map is non-parametric — as many effective degrees of freedom as the data
has blocks, far more than a single scalar — so it needs more calibration data to avoid overfitting, and
the out-of-bounds clip is its only defense at the score range's edges where the shifted test tail is most
likely to land. This cuts two ways and that is exactly what makes the prediction interesting. Where the
distortion is genuinely non-uniform, the freedom to bend should drop the worst-group ECE below what the
single global slope managed; where the calibration sample is small or the shift pushes test scores into a
region the calibration set barely covered, the extra freedom can chase calibration-split noise and the
clipped extrapolation can land a whole shifted block on one flat value — a step backward.

State it against the measured numbers. On Adult, where temperature scaling bottomed out at 0.484
worst-group ECE, I expect isotonic to clearly *help* — the data-chosen binning should bend the map where
the global slope could not, and I would not be surprised to see Adult's worst-group ECE fall well into
the 0.3s, the biggest single gain on this ladder so far. That is the optimistic side of non-parametric
flexibility, and it is the whole reason to climb here. On Law School I am genuinely worried: temperature
scaling already got it to 0.3098, its *best* dataset, which means the global scale correction nearly
sufficed there, so the extra isotonic degrees of freedom have little real signal to fit and a lot of room
to overfit the small shifted split — I expect isotonic to do *worse* on Law School, possibly back up into
the 0.4s, and to drag its `subgroup_auroc` down a touch (because PAVA's flat blocks can collapse the
ordering within a block, unlike the strictly-monotone scalar map, the one place AUROC is *not* perfectly
preserved). COMPAS I expect to stay near the 0.48 the scalar posted — a marginal move either way. So the
falsifiable claim is mixed by design: isotonic should win on the dataset whose distortion is non-uniform
(Adult) and lose on the dataset whose distortion was already nearly a pure scale (Law School), which
would make its *mean* worst-group ECE only roughly tie temperature scaling while trading where the error
lives. If that is what the numbers say, the diagnosis for the next rung writes itself: free-form
monotonicity over-fits the small shifted splits, so I want to keep the ability to bend — including the
*gathering* shapes temperature scaling cannot make — but inside a tightly *parametric* family of only a
few parameters, so I get the shape without the variance. That is the move past isotonic.
