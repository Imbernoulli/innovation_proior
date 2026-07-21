The strategy is the whole point, but it runs against a fixed loop and a fixed budget, and the weakest
honest thing I can put in that loop is the floor everything else has to beat — so I start there, with
the question of what "no strategy at all" should look like inside this scaffold. The loop hands me
`suggest(space, history, budget_left)` and asks for one configuration plus a fidelity, over and over,
until `total_cost` reaches the per-benchmark budget (50 for XGBoost, 40 for SVM and the NN). I have three
search spaces that mix types and scales: 6-D for XGBoost and the NN, 3-D for SVM, with log-scaled learning
rates and regularizers, integer layer widths and tree counts, and a categorical kernel/activation. The
evaluation is a real train-and-score, gradient-free, with no view inside — I get back a single scalar. So
whatever the floor is, it has to (1) produce a valid configuration over this mixed, log-scaled, partly
categorical domain, and (2) make sense as the thing the adaptive methods will be measured against.

The temptation is to reach for grid search as the naive default, and the honest way to dismiss it is to
do the budget arithmetic rather than wave at it, because the arithmetic is what motivates the whole ladder.
A grid enumerates the Cartesian product of per-axis levels, so the trial count is the product of the
per-axis level counts — exponential in dimension. On the 6-D XGBoost space, three levels per axis is
3^6 = 729 evaluations against a budget of 50; even two levels per axis is 2^6 = 64, still over budget, so
I cannot complete a *single* two-level sweep of the XGBoost knobs before the budget runs out. Turn it
around and ask how many levels per axis a budget of 50 actually buys in 6-D: 50^(1/6) ≈ 1.92, i.e. fewer
than two. That number is the deeper objection, not just the count. A grid's points are *aligned*, so they
collapse under projection: with g levels per axis in K dimensions the grid spends g^K trials but probes
each individual knob at only g = (trial count)^(1/K) distinct values, because every grid point shares each
of its coordinates with a whole slab of siblings. On the subspace of knobs that actually matters, a grid
therefore has the resolution of the K-th root of its budget — here, under two distinct settings of the
learning rate, no matter how I spend the other five axes.

The redundancy is the shared-coordinate structure: in a 2-D four-corner grid the first axis takes only
{0, 1}, two distinct settings from four points, because every value of one knob is reused across all levels
of the others. Random draws have no such sharing — T independent uniform draws land almost surely on T
distinct values on *every* axis at once, since two continuous draws coincide on any coordinate with
probability zero. So a budget of 50 buys 50 distinct learning-rate settings from random sampling versus
under two from a grid — a factor of about twenty-six on whichever axis turns out to be the one that
matters, and I do not have to know in advance which axis that is.

That last clause is the real reason random beats grid here, and it rests on an empirical fact about these
response surfaces: they have *low effective dimensionality*. On any given dataset only a few of the knobs
move the loss appreciably, and which few differ across datasets, so I cannot just pick the important ones
and grid them finely. There is a clean, dimension-free way to see why independent sampling is the right
response. Idealize the good region as a target occupying relative volume v/V of the box. Each independent
uniform draw misses it with probability (1 − v/V), so the probability that T draws find it is
1 − (1 − v/V)^T, and the ambient dimension K does not appear anywhere — only the target's relative volume.
Put numbers on it against my actual budgets. With T = 50 (XGBoost): a target of relative volume 0.05 is
hit with probability 1 − 0.95^50 = 1 − 0.077 = 0.923; a 0.02 target with 1 − 0.98^50 = 1 − 0.364 = 0.636;
a 0.01 target with 1 − 0.99^50 = 1 − 0.605 = 0.395. With T = 40 (SVM and NN): the 0.05 target is
1 − 0.95^40 = 0.871, the 0.02 target 1 − 0.98^40 = 0.554. So under a few dozen draws random search reliably
finds any region that occupies even a couple of percent of the box, and — this is the point — it does so
whether that region lives in 3 dimensions or 6, because K never entered the formula. A region that is wide
along the irrelevant knobs and narrow along the few important ones still has findable relative volume in
any number of dimensions, which is exactly why random search "thrives" precisely in the regime grid is
worst at. And against the axis-aligned elongated targets that low effective dimensionality produces, grid
is *especially* bad: a thin axis-aligned rectangle either threads through several collinear grid points
(redundant, collapsing the effective sample size) or slips entirely between the grid lines (catching
nothing), whereas independent draws are never collinear, so each is an independent shot.

The formula's one real assumption is legible in its limits: as v/V → 0 the hit probability goes to 0 for
any finite T, so random sampling cannot find a *measure-zero* target — it relies on the good configurations
forming a fat region, not a knife-edge. That is benign here (a setting that only works at one infinitely
precise value would be useless anyway), but it is the assumption the floor rests on.

Two neighbours a careful person would weigh sharpen what the baseline is. A low-discrepancy sequence (Sobol,
Halton) fills the box more evenly than i.i.d. draws, but its star-discrepancy advantage scales like
(log T)^K / T, whose numerator grows with K, so in this high-dimension / low-budget regime it buys almost
nothing while sacrificing the i.i.d. structure that makes the baseline anytime-stoppable and analyzable.
Latin hypercube sampling is the sharper competitor, because it shares random search's headline virtue: one
draw per axis-bin guarantees T distinct values on every axis, solving the projection problem just as
cleanly. I cannot reject it with the grid argument; I reject it on the baseline's own terms. LHS bakes in a
stratification (a mild coverage prior) and breaks strict i.i.d. (the draws are negatively correlated by
construction). A baseline must isolate exactly one mechanism, and LHS — like a discrepancy prior or an
axis-importance prior — smuggles in a second, which would tangle the measurement. So plain uniform random is
not merely the simplest choice; it isolates "coverage with no adaptation" and *nothing* else, so when a
later method beats it the gain is attributable to the one ingredient that method added rather than to a
better sampler. That cleanliness is the whole value of a floor.

The scaffold has already done the type-and-scale bookkeeping: `space.sample_uniform` samples categoricals
uniformly over choices, floats uniformly over their range, or uniformly in log space when `log_scale` is
set. The log handling is not incidental. A learning rate over [1e-3, 1e0] sampled in the *raw* range puts
≈90% of draws in the top decade [0.1, 1] — about 45 of 50 — and under one draw in fifty in the bottom
decade; sampled in log space each decade gets a third. Since these knobs have a roughly flat response per
decade, the log draw covers the *effect* evenly while the raw draw wastes its budget in the top decade.
Because the scaffold does this correctly, the floor just calls `sample_uniform` and touches no encoding.

The categorical axis on SVM does not merely add a dimension — it *gates* the good region. `sample_uniform`
draws the kernel uniformly, so each of three kernels gets ≈13 of 40 draws, and the good (C, gamma) slab
exists only under the right kernel (RBF). The effective per-draw hit probability is then a product: ≈1/3 for
the kernel times the slab's relative volume within the RBF sub-box. If that slab is a fifth of its box, the
effective probability is ≈(1/3)(1/5) ≈ 0.067, a mean first hit around draw 15 of 40 — findable, but the
kernel factor makes the arrival later and more variable. So on SVM the fat slab guarantees the *final*
quality while the kernel gate lengthens the wait for the first good draw; no adaptive machinery is needed to
hit it, only to hit it *sooner*, which is the ladder's job.

Two design knobs remain, and the floor takes the trivial choice on both. First, *fidelity*: the loop
accepts a fraction in (0,1] and scales the objective's cost by it — at fidelity 1.0 XGBoost trains its full
`n_estimators`, SVM runs 5-fold CV, the NN runs 500 iterations, each at maximum resolution, the honest
setting for measuring *final* quality; a fractional fidelity (floors: n_estimators 10, CV max(2,int(5·b)),
max_iter max(50,int(500·b))) buys a cheaper, noisier, biased look and more evaluations against the same
budget. The floor returns 1.0 every call — multi-fidelity is a lever the later rungs pull, and folding it in
would muddy what the baseline measures. One checkable consequence follows: every call spends exactly one
cost-unit and the loop stops at `total_cost ≥ budget`, so `total_evals` must land exactly on the budget
(50/40/40) with no seed-to-seed variation. Second, *adaptivity*: `suggest` is handed the full `history`,
but the floor ignores it and `budget_left` entirely — every draw is i.i.d. That statelessness is the
defining property of the floor and the one weakness the whole ladder exists to remove: under a budget of a
few dozen trials, discarding every loss already paid for is the most expensive thing a strategy can do, and
the baseline does it on purpose so the cost of *not* adapting shows up in the numbers.

So the edit at step 1 is the minimal one: leave the class essentially at the scaffold default — store the
seed, build one `np.random.RandomState(seed)` so the run is reproducible, and have `suggest` return
`space.sample_uniform(self.rng), 1.0`, ignoring `history` and `budget_left`. One independent uniform draw
per call, full fidelity, no memory. The distilled module is in the answer.

Now reason about what this floor must do on these three benchmarks, because that is the whole reason to run
it, and the three spaces predict three different failure profiles. Two of the spaces are small. SVM is 3-D,
and there the good region is a fat slab in (C, gamma) for the RBF kernel — the loss is forgiving over a
wide band of both once the kernel is right — so by the hit-probability arithmetic (a fat target of a few
percent volume, T = 40) I expect random search to find a near-optimal SVM almost regardless of seed. The
*final* score should therefore be near-identical across seeds; the only discriminator there will be how
*early* it lands a good config, i.e. the convergence AUC, which is at the mercy of draw order and should be
both lower and more variable than an adaptive method's — the easy space is precisely where statelessness
shows up as timing luck rather than as quality loss. XGBoost is 6-D but its objective is forgiving in a
different sense: the cross-validated error on California Housing is dominated by a couple of knobs (learning
rate and depth), so the effective target is fat along the other four axes and the best score should again
land in a narrow band across seeds, with AUC the real differentiator. The NN is the hard case: a 6-D space
with two log-scaled widths, a log learning rate, and a log regularizer, where bad configurations (too-high
learning rate, too-small capacity) can score catastrophically, so an unlucky draw sequence can leave the
best-so-far curve flat for many evaluations before it stumbles onto a workable region — which should make
the NN convergence AUC the noisiest and weakest number random search produces.

The AUC-variance prediction is quantitative, because the metric's mechanics force it. `convergence_auc` is
the area under the non-decreasing min-max-normalized best-so-far curve, so it is dominated by *when* the
curve makes its big jump — the evaluation at which the first good-region config appears — and is, to first
order, one minus the fraction of the budget elapsed before that first hit. For random search that arrival is
geometric: with per-draw hit probability p the first hit has mean 1/p and coefficient of variation √(1−p),
close to 1 for the small p that matters here. For a p = 0.05 target against T = 40 the first hit arrives on
average at draw 20 with a standard deviation of ≈19.5 draws — a spread as large as its center. So the
*final* score is stable (a fat target found once is found at similar quality however the draws fall) while
the *AUC* is volatile (its first-hit arrival has CV near one), and statelessness is exactly what pins random
search to that geometric law: nothing shortens the heavy-tailed wait the AUC integrates.

So the deficiency the numbers should expose is not final quality but *use of history*: `best_val_score`
competitive everywhere and roughly seed-stable, `total_evals` pinned to the budget, but `convergence_auc`
the weakest and by far highest-variance number — worst on the higher-dimensional NN and on any seed whose
draw order withholds a good config until late. Every later method keeps random search's coverage virtues
while adding the one thing it refuses to do: let the losses it has already paid for steer the next draw.
