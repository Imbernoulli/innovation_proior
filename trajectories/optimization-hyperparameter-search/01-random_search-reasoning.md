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

The temptation is to reach for grid search as the naive default, but it is wrong here on two counts and
naming why is worth it because it motivates the whole ladder. First, the trial count is the Cartesian
product of per-axis levels, so it is exponential in dimension: even three levels per axis on the 6-D
XGBoost space is 729 evaluations, and my budget is 50. Second — and this is the deeper objection — a grid's
points are *aligned*, so they collapse under projection: g levels per axis in K dimensions uses g^K trials
but probes each individual knob at only g = (trial count)^(1/K) distinct values, because every grid point
shares coordinates with many others. On the subspace of knobs that actually matters, a grid has the
resolution of the K-th root of its budget. So grid is not a serious floor; it is a strawman.

Random sampling is the serious floor, and it has a real argument behind it, not just simplicity. The
empirical fact about hyperparameter response surfaces is that they have *low effective dimensionality* —
on any given dataset only a few of the knobs move the loss appreciably, and which few differ across
datasets, so I cannot just pick the important ones and grid them. Refusing to align the points fixes the
projection problem for free: T independent uniform draws project, almost surely, to T distinct values on
*every* axis, so whichever low-dimensional subspace turns out to matter gets the full budget's worth of
resolution at once, without my having to know in advance which subspace it is. There is a clean way to see
why this is dimension-free. Idealize the good region as a target occupying relative volume v/V of the box;
each independent uniform draw misses it with probability (1 − v/V), so the probability that T draws find it
is 1 − (1 − v/V)^T, and the ambient dimension K does not appear — only the target's relative volume. A
region that is wide along the irrelevant knobs and narrow along the few important ones still has findable
relative volume in any number of dimensions, which is exactly why random search "thrives" precisely in the
regime grid is worst at. And against the axis-aligned elongated targets that low effective dimensionality
produces, grid is *especially* bad: a thin axis-aligned rectangle either threads through several collinear
grid points (redundant, collapsing the effective sample size) or slips entirely between the grid lines
(catching nothing), whereas independent draws are never collinear, so each is an independent shot.

The scaffold has already done the type-and-scale bookkeeping I would otherwise have to write myself.
`space.sample_uniform(rng)` walks `space.params` and, for each `HParam`, samples categoricals uniformly
over `choices`, floats uniformly over `[low, high]` — or uniformly in log space (`exp` of a uniform draw
over `[log low, log high]`) when `log_scale` is set — and integers uniformly over the inclusive range, again
log-scaled when asked. That log-uniform handling is not incidental: learning rates, layer widths, and the
regularizer span orders of magnitude with a roughly flat response per decade, so sampling them uniformly in
log space covers the *effect* evenly rather than wasting almost every draw in the top decade of the raw
range. Because the scaffold already does this correctly, the floor strategy does not need to touch
encoding at all — it just calls `sample_uniform`, which is the cleanest possible expression of "draw each
knob independently from its own range and type."

It is worth being explicit that random search is not merely "better than grid"; it is the right *baseline*
in a way a smarter method is not. A baseline's job is to make the cost of a deficiency legible, and to do
that it must isolate exactly one mechanism. Random search isolates "coverage with no adaptation": it has
the full per-axis resolution of the budget and the dimension-free hit probability, and it has *nothing*
else — no surrogate, no resource scheduling, no memory. So when a later method beats it, the gain is
unambiguously attributable to the one ingredient that method added, rather than tangled up with a better
sampler or a cleverer encoding. That cleanliness is why random search, despite being trivial, is the
correct floor rather than, say, a quasi-random Sobol sequence: Sobol can shave a few percent of coverage
error in low dimensions, but it sacrifices the i.i.d. structure that makes the baseline anytime-stoppable,
freely extendable, and analyzable, and in the high-dimension/low-budget regime it is no real improvement —
so it would only blur what the baseline is supposed to measure without buying anything the ladder cares
about.

There are two design knobs left, and the floor takes the trivial choice on both. The first is *fidelity*:
the loop accepts a fraction in (0,1] and scales the objective's cost by it (fewer trees, fewer CV folds,
fewer MLP iterations), and a cheaper evaluation buys more evaluations against the same budget. The floor
returns fidelity 1.0 on every call — single-fidelity, no cheap-and-noisy partial evaluations — because the
whole point of a floor is to have *no* mechanism beyond uniform sampling; multi-fidelity is a lever the
later rungs will pull, and conflating it into the baseline would muddy what the baseline measures. The
second knob is *adaptivity*: `suggest` is handed the full `history` every call, so a strategy could read
past scores and bias the next draw. The floor deliberately ignores `history` and `budget_left` entirely —
every draw is i.i.d., independent of everything seen so far. That is the defining property of the floor and
the one weakness the entire ladder exists to remove: random search is *stateless*. Under a budget of a few
dozen trials, throwing away every loss already paid for is the single most expensive thing a strategy can
do, and it is exactly what this baseline does on purpose, so that the cost of *not* adapting is what shows
up in the numbers.

So the edit at step 1 is the minimal one: leave the class essentially at the scaffold default — store the
seed, build one `np.random.RandomState(seed)` so the run is reproducible, and have `suggest` return
`space.sample_uniform(self.rng), 1.0`, ignoring `history` and `budget_left`. One independent uniform draw
per call, full fidelity, no memory. The distilled module is in the answer.

Now reason about what this floor must do on these three benchmarks, because that is the whole reason to run
it. Two of the spaces are small — SVM is 3-D, and there the good region is a fat slab in (C, gamma) for the
RBF kernel, easy to hit by chance, so I expect random search to find a near-optimal SVM almost regardless
of seed, and the *only* discriminator there will be how *early* it lands a good config, i.e. the
convergence AUC, which is at the mercy of draw order and should be both lower and more variable than an
adaptive method's. XGBoost is 6-D but its objective is forgiving in the sense that the cross-validated MSE
on California Housing is dominated by a couple of knobs (learning rate and depth), so I expect the best
score to land in a narrow band across seeds, again with AUC the real differentiator. The NN is the hard
case: a 6-D space with two log-scaled widths, a log learning rate, and a log regularizer, where bad
configurations (too-high learning rate, too-small capacity) can score very poorly, so an unlucky draw
sequence can leave the best-so-far curve flat for many evaluations before it stumbles onto a workable
region — which should make the NN convergence AUC the noisiest and weakest number random search produces.

The falsifiable expectation, then, is sharp: random search should reach *competitive final best scores*
everywhere — that is the lesson of low effective dimensionality, and it is why random search is a famously
strong baseline on *final* quality — but it should pay for its statelessness in *convergence speed*,
especially on the higher-dimensional NN space and in the run-to-run variance of the AUC, because nothing in
the strategy concentrates draws toward where the evidence already points. Whatever the precise split, the
diagnosis is already aimed at the next rung: the deficiency is not the final-quality of the search but its
*use of history* — every later method I try will keep random search's coverage virtues while adding the one
thing it refuses to do, which is to let the losses it has already paid for steer the next configuration.
