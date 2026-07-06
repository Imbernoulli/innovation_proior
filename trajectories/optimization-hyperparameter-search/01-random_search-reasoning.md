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

Let me actually check that "K-th root" claim on the smallest case rather than trust it, because it is the
load-bearing fact. Take two levels per axis in 2-D: the grid is the four corners {0,1}×{0,1}. Project onto
the first axis and the values collapse to {0, 1} — two distinct settings from four points, and indeed
4^(1/2) = 2. Extend to three levels in 2-D: nine points, but the first axis takes only {0, 0.5, 1}, three
distinct values, 9^(1/2) = 3. The redundancy is exactly the shared-coordinate structure: every distinct
value of one knob is reused across all the levels of the others. Random draws have no such sharing. T
independent uniform draws land, almost surely, on T distinct values on *every* axis at once, because the
event that two continuous draws coincide on any coordinate has probability zero. So a budget of 50 buys
50 distinct learning-rate settings from random sampling versus under two from a grid — a factor of about
twenty-six on whichever axis turns out to be the one that matters, and I do not have to know in advance
which axis that is.

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

Two limits of that hit-probability formula are worth reading off, because they tell me what the floor can
and cannot do and both are honest checks rather than reassurances. As T → ∞, 1 − (1 − v/V)^T → 1 for any
positive-volume target, so given unlimited draws random search finds any region of nonzero measure — it is
consistent, just slow. But as v/V → 0 the probability goes to 0 for any finite T, so random sampling
cannot find a *measure-zero* target: it relies on the good configurations forming a fat region, not a
knife-edge. That is a real assumption about these benchmarks, and it is a benign one — a hyperparameter
setting that only works at a single infinitely precise value would be useless anyway — but it is the
assumption, and naming it is the difference between a derivation and a slogan.

Before settling on plain i.i.d. sampling I should look at the neighbours a careful person would consider,
because two of them are genuinely close and dismissing them on the right grounds sharpens what the baseline
is. A low-discrepancy sequence (Sobol, Halton) fills the box more evenly than i.i.d. draws and can shave a
few percent off coverage error in low dimensions — but it sacrifices the i.i.d. structure that makes the
baseline anytime-stoppable, freely extendable, and analyzable, and in the high-dimension / low-budget regime
its discrepancy advantage largely evaporates (the classical star-discrepancy bound scales like
(log T)^K / T, whose numerator grows with K), so it would only blur what the baseline is supposed to measure
without buying anything the ladder cares about. Latin hypercube sampling is the sharper competitor, because
it shares random search's headline virtue: stratifying each axis into T bins and placing one draw per bin
guarantees T distinct values on every axis, so it solves the projection problem just as cleanly as
independent sampling does. I cannot reject it with the grid argument, then — I have to reject it on the
baseline's own terms. LHS bakes in a stratification, which is a mild coverage prior, and it breaks strict
i.i.d. (the draws are negatively correlated by construction, so the run is no longer a sequence of
independent shots), and both of those are *second mechanisms* smuggled into a strategy that is supposed to
have exactly one. The third neighbour, "sample the few important axes more finely and coarsen the rest,"
presumes I know which axes are important, and the whole low-effective-dimensionality premise is that I do
not and that it changes per dataset. All three fail for the same structural reason: a baseline must isolate
exactly one mechanism, and each of these adds a second one (a discrepancy prior, a stratification prior, an
axis-importance prior) that would tangle the measurement. So plain uniform random is not merely the simplest
choice; it is the one that keeps the experiment clean, and the cleanliness is the whole value of a floor.

The scaffold has already done the type-and-scale bookkeeping I would otherwise have to write myself.
`space.sample_uniform(rng)` walks `space.params` and, for each `HParam`, samples categoricals uniformly
over `choices`, floats uniformly over `[low, high]` — or uniformly in log space (`exp` of a uniform draw
over `[log low, log high]`) when `log_scale` is set — and integers uniformly over the inclusive range, again
log-scaled when asked. That log-uniform handling is not incidental, and the size of the effect is worth
computing. Take a knob that spans three decades, say a learning rate over [1e-3, 1e0]. Sampled uniformly in
the *raw* range, the fraction of draws landing in the top decade [0.1, 1] is (1 − 0.1)/(1 − 1e-3) ≈ 0.901,
so about 45 of 50 draws pile into the largest tenth of the range, and the fraction reaching the bottom
decade [1e-3, 1e-2] is (1e-2 − 1e-3)/(1 − 1e-3) ≈ 0.009 — fewer than one draw in fifty. Sampled uniformly
in log space, each of the three decades gets a third of the draws, roughly 17 apiece. Since these knobs
have a roughly flat response *per decade* (a factor-of-two change in learning rate matters about the same
whether it is near 1e-1 or near 1e-3), the log draw covers the *effect* evenly while the raw draw wastes
nearly its whole budget in the top decade. Because the scaffold already does this correctly, the floor
strategy does not need to touch encoding at all — it just calls `sample_uniform`, which is the cleanest
possible expression of "draw each knob independently from its own range and type."

The categorical axis deserves its own look, because on SVM it does not merely add a dimension — it *gates*
the good region, and that changes the hit-probability accounting. `sample_uniform` draws the kernel
uniformly over its choices, so over 40 SVM draws each kernel gets roughly 40/3 ≈ 13 draws. The good
(C, gamma) slab exists only conditional on the right kernel (RBF), so the effective per-draw probability of
landing a good SVM config is not the slab's volume but the *product* of two factors: the chance of drawing
the right kernel (≈ 1/3) times the slab's relative volume within the RBF sub-box. If that slab occupies,
say, a fifth of the (C, gamma) box, the effective hit probability is about (1/3)(1/5) ≈ 0.067, which by the
geometric arithmetic above still gives a mean first-hit around draw 15 of 40 — findable, but with the
kernel factor making the arrival noticeably later and more variable than the continuous-only picture would
suggest. This is why I flagged SVM's AUC, not its final score, as the fragile one: the fat slab guarantees
the final quality, but the kernel gate lengthens and roughens the wait for the first good draw. The floor
tolerates all of this without any special handling because `sample_uniform` gives every kernel equal
representation and the continuous slab is fat enough to be found within the RBF third of the draws; no
adaptive machinery is needed to hit it, only to hit it *sooner*, which is the ladder's job.

It is worth being explicit that random search is not merely "better than grid"; it is the right *baseline*
in a way a smarter method is not. A baseline's job is to make the cost of a deficiency legible, and to do
that it must isolate exactly one mechanism. Random search isolates "coverage with no adaptation": it has
the full per-axis resolution of the budget and the dimension-free hit probability, and it has *nothing*
else — no surrogate, no resource scheduling, no memory. So when a later method beats it, the gain is
unambiguously attributable to the one ingredient that method added, rather than tangled up with a better
sampler or a cleverer encoding. That cleanliness is why random search, despite being trivial, is the
correct floor.

There are two design knobs left, and the floor takes the trivial choice on both. The first is *fidelity*:
the loop accepts a fraction in (0,1] and scales the objective's cost by it, and it is worth being concrete
about what the extremes mean here, because that is what the floor is declining to use. At fidelity 1.0
XGBoost trains its full `n_estimators`, SVM runs its full 5-fold CV, and the NN runs its full 500 iterations
— each benchmark evaluated at its maximum resolution, which is the honest setting for measuring *final*
quality. A fractional fidelity would instead scale n_estimators (floored at 10), the CV folds
(max(2, int(5·budget))), and the MLP's max_iter (max(50, int(500·budget))) down, buying a cheaper and
noisier look; at fidelity 1/3, for instance, the SVM would use int(5/3) = 1 → clamped to 2 folds and the NN
would run int(500/3) ≈ 166 iterations, a third of the cost for a partial, biased score. A cheaper
evaluation buys more evaluations against the same budget. The floor
returns fidelity 1.0 on every call — single-fidelity, no cheap-and-noisy partial evaluations — because the
whole point of a floor is to have *no* mechanism beyond uniform sampling; multi-fidelity is a lever the
later rungs will pull, and conflating it into the baseline would muddy what the baseline measures. This
choice has a concrete, checkable consequence I can predict before I run anything: since every call spends
exactly one cost-unit (fidelity 1.0) and the loop stops at `total_cost ≥ budget`, the total number of
evaluations must land exactly on the budget — 50 on XGBoost, 40 on SVM, 40 on the NN — with no seed-to-seed
variation at all. If `total_evals` reads anything but the budget, I have a bug, not a strategy. The second
knob is *adaptivity*: `suggest` is handed the full `history` every call, so a strategy could read past
scores and bias the next draw. The floor deliberately ignores `history` and `budget_left` entirely — every
draw is i.i.d., independent of everything seen so far. That is the defining property of the floor and the
one weakness the entire ladder exists to remove: random search is *stateless*. Under a budget of a few
dozen trials, throwing away every loss already paid for is the single most expensive thing a strategy can
do, and it is exactly what this baseline does on purpose, so that the cost of *not* adapting is what shows
up in the numbers.

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

I can make the AUC-variance prediction quantitative instead of impressionistic, because the metric's
mechanics force it. `convergence_auc` is the trapezoidal area under the min-max-normalized best-so-far curve
plotted against normalized cost, and the best-so-far curve is non-decreasing, so the area is dominated by
*when* it makes its big jump — the evaluation at which the first configuration in the good region appears
and pulls the incumbent up near the top. Everything before that jump contributes little area; everything
after contributes near the maximum. So the AUC is, to first order, one minus the fraction of the budget that
elapsed before the good region was first hit. For random search that arrival time is a geometric random
variable: if the good region has per-draw hit probability p, the number of draws until the first hit has
mean 1/p and standard deviation √(1−p)/p, so its standard deviation is essentially equal to its mean — the
coefficient of variation of a geometric is √(1−p), which is close to 1 for the small p that matters here.
Concretely, for a p = 0.05 target against T = 40 the first hit arrives on average at draw 20 (fraction 0.5
of the budget) with a standard deviation of about 19.5 draws (fraction ≈ 0.49); for p = 0.1 the mean is
draw 10 (fraction 0.25) with standard deviation ≈ 9.5 (fraction ≈ 0.24). A quantity whose spread is as large
as its center is exactly a quantity that will swing wildly seed to seed, and that is the AUC. This is the
mechanism behind the profile I am predicting: the *final* score is stable because a fat target found once is
found at similar quality however the draws fall, but the *AUC* is volatile because the arrival time of that
first hit has a coefficient of variation near one. Statelessness is what pins random search to the geometric
arrival law — nothing pulls the draws toward the good region once evidence points at it, so the strategy has
no way to shorten that heavy-tailed wait, and the tail of the wait is what the AUC integrates.

The falsifiable expectation, then, is sharp and stated in the metrics this task actually reports.
`best_val_score` should be *competitive everywhere* — that is the lesson of low effective dimensionality,
and it is why random search is a famously strong baseline on final quality — and it should be roughly
seed-stable, because a fat target found once is found at similar quality however the draws fall.
`total_evals` should sit exactly on the budget, the tell of single fidelity. And `convergence_auc` is where
the statelessness bill comes due: it should be the weakest and by far the *highest-variance* of the three
numbers, worst on the higher-dimensional NN and on any benchmark where a single seed's draw order happens
to withhold a good config until late — a run where the best-so-far curve sits flat for many evaluations
should produce a visibly low AUC on that seed while its final score is unharmed, and that divergence between
a stable `best_val_score` and a swinging `convergence_auc` is the exact signature I am predicting. Whatever
the precise split, the diagnosis is already aimed at the next rung: the deficiency is not the final quality
of the search but its *use of history* — every later method I try will keep random search's coverage virtues
while adding the one thing it refuses to do, which is to let the losses it has already paid for steer the
next configuration.
