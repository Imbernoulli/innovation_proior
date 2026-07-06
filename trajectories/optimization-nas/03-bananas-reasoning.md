REA's numbers landed almost exactly where my falsifiable prediction said the risk was, and reading them
carefully is what picks the next rung. The headline is that REA did *not* clearly beat random search:
CIFAR-10 went 93.38 → 93.40 (a wash, as I predicted — the compressed top is near-saturated and there is no
room), CIFAR-100 actually *slipped* 70.65 → 70.43, and ImageNet16-120 rose 44.57 → 45.25, the one setting
where I expected the most room. So the directional prediction held on ImageNet16-120 and CIFAR-10, but the
CIFAR-100 slip and — more telling — the ImageNet16-120 **variance** are the real story, and both deserve to
be read as numbers rather than impressions.

Take the variance first. REA's ImageNet16-120 spread blew up to ±1.44, against random search's ±0.58 — a
factor of 2.5 in standard deviation, with seed 2 cratering to 42.70 while seed 0 hit 46.80, a 4.1-point gap
between the best and worst seed of a single method. That ±1.44 is the diagnosis written in a number, and it
is precisely the premature-convergence failure I worried about when I chose the eviction rule and named the
seeding bet. Let me decompose it to be sure I am reading it right. If I set seed 2 aside, the other four
ImageNet runs average `(46.80 + 46.40 + 44.93 + 45.43)/4 ≈ 45.89` with a spread near ±0.74 — a genuine lift
of more than a point over random search's 44.57, and tighter than the full five-seed number suggests. So on
four of five seeds the twenty evolution steps hill-climbed a strong seed into a genuinely better cell (46.8
and 46.4 are clearly above anything random search drew, whose best ImageNet seed was 45.47). The entire
variance blow-up is one seed. And that one seed is damning in a specific way: 42.70 is not merely low, it is
1.10 points *below* random search's worst ImageNet draw of 43.80 and 1.87 below random search's mean. REA's
directed search, on seed 2, did worse than throwing thirty random darts. That is the seeding bet losing in
the open: ten random seeds landed in a mediocre basin, the tournament locked onto the local best of those
seeds, and twenty single-edge mutations refined that basin without ever crossing out of it.

The mechanism is exactly the distance-across-a-valley problem I flagged. Single-edge mutation reaches only
the 24 neighbors of an evaluated cell, and tournament selection almost always accepts improving or lateral
moves, so it cannot pay the temporary accuracy cost to walk a path of *worse* cells toward a better basin.
If the good ImageNet region sat five or six edits away from seed 2's attractor with a fitness valley
between, the connectivity of the space guarantees a path exists but the greedy local walk cannot traverse
it in twenty steps. REA is high-variance because it is a *local* search: it exploits beautifully when the
seeds put it near the good region and fails badly when they do not, and ten random seeds plus twenty
single-edge steps is not enough coverage to guarantee the former. The mechanism that helped — exploit the
best-seen by mutating it — is the same mechanism that hurt, because mutation only ever moves one edit from
something already evaluated. It cannot *extrapolate*; it can only refine what luck handed it.

CIFAR-100 confirms the same lesson from the other side. There REA did not even inflate the variance — its
spread (±0.68) is about random search's (±0.65) — it simply *centered lower*, 70.43 against 70.65. And its
worst seeds, 69.76 and 69.78, sit below random search's worst of 69.92, while its best seed, 71.46, fails to
reach random search's best of 71.54. So on CIFAR-100 the directed local search found nothing random search
had not already found and on some seeds actively did worse: the twenty mutations replaced twenty random
draws with local refinement of a landscape that apparently does not reward single-edge climbs, and the
`1/11 → 1/31` coverage the seeding bet gave up was not paid back. That is the seeding bet losing quietly
rather than catastrophically. CIFAR-10, meanwhile, is the wash I expected — 93.40 versus 93.38, spreads
essentially equal — because the compressed top leaves no room for any mechanism to separate itself. Across
all three, one diagnosis unifies the reads: local exploitation is a high-variance bet that pays off only
when the seeds land near the good region, and nothing in REA controls whether they do.

So the lever the REA numbers expose is different from the one the random-search numbers exposed. Random
search needed exploitation; REA has exploitation but no way to use the thirty queries to reason about
architectures it has *not* yet evaluated. What I want now is a method that, after each query, builds a
*model* of the accuracy surface from all the `(arch, acc)` pairs seen so far and uses it to choose the next
query — so that the thirty queries compound into a prediction over the whole 15,625-cell space, not just a
one-edit neighborhood of the current best. That is Bayesian optimization in spirit: model what you have
seen, pick the next point to maximize expected progress, evaluate, update. The question is what the
surrogate, the encoding, and the selection rule should be under this budget and inside this scaffold, and
each of those has a wrong obvious answer I need to walk past.

Start with the surrogate, because the textbook choice fails for a concrete reason. The canonical BO
surrogate is a Gaussian process: condition on the data, read off a posterior mean and variance. But a GP
*is* its kernel, and the inputs here are labeled cell graphs — there is no off-the-shelf kernel on DAGs, and
building a bespoke distance over architectures (an optimal-transport metric over labeled graphs, say) is a
hard modeling problem of its own, one that has consumed whole papers. On top of that GP inference is cubic
in the number of observations — irrelevant at ≤29 points computationally, but the kernel problem is the real
blocker. I do not need any of that. All I actually require from the surrogate is: consume the architectures
seen, predict the validation accuracy of an unseen one, and let me rank candidates. A neural network does
exactly that and learns its own notion of similarity from a feature vector, so the "invent a kernel on DAGs"
problem evaporates — I hand the net a featurization and let it fit its own geometry. So the surrogate is a
small feedforward network trained from the `(encoding, val_acc)` pairs at each step.

The encoding is the next decision, and the scaffold has already made the right one available — but I should
understand *why* it is right rather than just take it. The default move would be an adjacency encoding — a
bit per possible edge plus the operation at each node — but that is order-dependent and its features are
violently inter-dependent: an edge bit means nothing without a path connecting it to the output, so the net
would have to learn the DAG's connectivity semantics from a handful of examples before any feature became
meaningful. The scaffold instead provides `path_encoding`, a binary indicator over input→output
operation-paths — one feature per "the tensor can flow through this sequence of operations," length `5 + 5^2
+ 5^3 = 155` for paths of length one, two, and three. Each feature is a self-contained statement about what
the cell *computes* rather than a fragment of its wiring, so the features are far less entangled and a
single architecture maps to a single encoding with no isomorphism ambiguity.

There is a quantitative reason this encoding is exactly what rescues a tiny-data regressor, and it is worth
computing because it is the crux of why a 155-dimensional model can fit from ≤29 points at all. The
NAS-Bench-201 cell is a fixed DAG on four nodes — input node 0, output node 3, edges for every `i → j` with
`i < j`, so six edges. Count the input-to-output paths: the direct `0 → 3`; the two length-two paths `0 → 1
→ 3` and `0 → 2 → 3`; and the single length-three path `0 → 1 → 2 → 3`. That is four structural paths, no
more, fixed by the topology. So any architecture lights up at most *four* of the 155 path features — the op-
sequences realized along those four paths — and fewer whenever a path runs through a `none` operation that
kills the tensor flow. The encoding is therefore extremely sparse, on the order of `4/155 ≈ 2.6%` density,
and the handful of features any set of 29 architectures can activate spans a low-dimensional, near-orthogonal
subspace rather than the full 155. That sparsity is the whole reason the regression is not hopeless: the net
is not really fitting 155 free directions from 29 points, it is fitting the few dozen path-features those 29
cells actually exercise, and the path structure makes those features additive-ish proxies for "does this
cell contain a good computation." With 155 nominal dimensions but this sparsity, the full path encoding is
small enough to use as-is; there is no need to truncate it to the most-frequent short paths the way one would
on a large cell where the encoding length explodes — here it does not, so I take the whole 155-dim vector the
loop hands me.

Now the predictor itself, and here I have to respect what this scaffold can actually run. I cannot assume a
deep-learning framework is in the image, so I build the net in numpy: a two-layer MLP — Linear(155, 64) →
ReLU → Linear(64, 1) — trained with Adam on a plain mean-squared-error objective against the validation
accuracies. That network has `155·64 + 64 + 64 + 1 ≈ 10,000` parameters, fit against at most 29 scalar
targets, which by naive counting is wildly over-determined — ten thousand knobs for twenty-nine numbers — and
the reflex worry is that it will memorize and generalize terribly. Three things stop it, and they are exactly
the design choices, not luck. First, the sparse, low-rank path features mean the *effective* input dimension
the 29 points exercise is a few dozen, not 155, so the fit is far less under-determined than the parameter
count suggests. Second, I train for only 200 Adam steps at `lr = 1e-2` from a `1/√fan-in` small
initialization — that is deliberate early stopping: the net never runs to interpolation, it settles into a
low-complexity fit that captures the dominant path-effects and stops before it can carve the training points
out of noise. Third, the ensemble (below) averages several such fits, cutting the variance of any one. Why
MSE and not a fancier loss that up-weights the high-accuracy architectures? Because at this budget the
predictor's job is modest: rank a pool of candidates well enough to pick a good one, not to nail the error
of near-optimal cells to the decimal. A plain MSE regressor is the honest, low-variance choice — it fits fast,
it has no extra hyperparameter to tune against a budget that cannot afford tuning, and it predicts accuracy
directly so I can simply pick the candidate the model thinks is best.

Every acquisition rule wants more than a point prediction — it wants an uncertainty, so the search can
explore where the model is ignorant — and a single net gives only a point estimate. The cheap, adequately-
calibrated way to get an uncertainty is an ensemble: train M independent copies from different random
initializations and read the disagreement among them as the uncertainty. The variance of the ensemble mean
falls like `1/M` if the members were independent, so M = 5 already cuts the predictor's variance to about a
fifth of a single net's while staying cheap enough to retrain from scratch every step. I do not push M higher
because the returns diminish (M = 10 would only reach a tenth, at double the compute) and, as the cost
accounting below shows, compute is not the scarce resource anyway. Each member gets its own seed
(`self.seed + i + 1`) so the members genuinely disagree rather than collapsing to the same fit.

Now the selection rule, and this is the place where I deliberately keep it simple rather than reaching for the
full BO acquisition machinery. With the ensemble I have, for any candidate, the mean prediction across the
five members. The acquisition I use is the plainest exploitation rule: score every candidate by the ensemble
mean and query the single highest-scoring one. I considered the uncertainty-aware alternatives — upper-
confidence-bound (`mean + β·std`), or Thompson-style sampling from each candidate's predictive spread — and
they are the principled way to balance exploration against exploitation when you have a budget long enough to
pay exploration back. Here they are the wrong tool for a computable reason. With ≤29 points in a 155-dim
encoding, the ensemble std is *poorly calibrated on out-of-distribution candidates*: a cell far from
everything seen makes the five nets extrapolate in five different directions, so its disagreement is huge —
not because it is promising but because it is unknown. A UCB rule adds `β·std` and so would systematically
steer the query toward the most out-of-distribution candidate in the pool, precisely the cell the model has
no basis to rank, and at thirty queries there is no time to recover from spending one on a phantom. Even one
such wasted query is 5% of the twenty predictor-guided steps. So an explicit uncertainty bonus, on top of a
29-point predictor, chases the model's high-variance phantoms into regions it has no business trusting. I let
the acquisition be pure greedy exploitation of the mean, which is also robust to exactly that mis-calibrated
std — and I supply the exploration a different way, through where the candidates come from.

The candidate pool is the last piece and the one that most distinguishes this rung from REA, and it is where
the exploration lives. REA's fatal limitation was that it only ever looked one edit — 24 neighbors — away
from an evaluated architecture, so it could not escape a bad seeding region; that is what produced the 42.70
seed. To fix exactly that, I draw the candidate pool *uniformly at random over the whole space*: each step I
sample a large pool of unseen random architectures and score all of them with the ensemble. This is the
deliberate opposite of REA's mutate-the-best — instead of refining the current neighborhood, the predictor
gets to reach across the entire 15,625-cell space and pull in whichever architecture it predicts is best,
even if it is many edits away from anything seen. Let me size the pool by the same volume arithmetic the
floor used, because it decides whether a good candidate is even *present* to be picked. A pool of 500 unseen
draws contains at least one architecture from the top-1% region with probability `1 - 0.99^{500} ≈ 0.993`,
and from the top-0.5% region with `1 - 0.995^{500} ≈ 0.92`. Contrast REA's reach: its 24-neighbor
neighborhood contains a top-1% cell with probability only `1 - 0.99^{24} ≈ 0.21`. So the random pool is
present-with-a-great-candidate on essentially every step, where REA's mutation had a roughly one-in-five
chance of even having a top-1% cell within reach — the pool's odds of *lacking* a top-1% candidate are about
120 times smaller than the mutation neighborhood's. That factor is the global reach REA lacked, quantified.
The model then only has to *rank* that present-almost-surely good candidate above the rest, which is the job
the surrogate exists to do; the random pool supplies the exploration, the ensemble mean supplies the
exploitation, and neither has to be a schedule.

I can bound the whole thing's cost to confirm I am free to be this lavish. Fitting five MLPs for 200 Adam
steps on a ≤29×155 matrix, then scoring 500 candidates through five nets of ~10k parameters, is a few tens of
millions of floating-point operations per step — microseconds — repeated over the ~20 predictor-guided steps.
Against that, the budgeted resource is the thirty table lookups, each of which "costs" one of my thirty units
regardless of how cheap it is in wall-clock. So the query is the only scarce thing, and the predictor,
ensemble, and pool are all essentially free. That asymmetry is what licenses M = 5, pool = 500, and a full
refit every step: I am spending unlimited cheap compute to squeeze the most ranking signal out of each
precious query.

It is worth checking the design at its limits to be sure it interpolates between the right things. If I
shrank the pool to a single random candidate and the ensemble to one net, the method would degenerate to
random search — it would query one random architecture each step regardless of the model — which is the
correct floor to collapse toward. At the other limit, with a perfect predictor and a pool covering the whole
space, the argmax would pick the true best cell every step, i.e. the oracle. So greedy-mean-over-a-random-
pool is a dial between random search and the oracle, set by how good the surrogate is and how much of the
space the pool exposes — which is exactly the knob I want a predictor-guided search to be. And a graceful-
degradation check on the code: if the budget were ever smaller than the warm-start, or before two points
exist, the loop falls back to random draws, so it never tries to fit a predictor on nothing. Those limits all
land where they should, so the assembly is sound.

Assemble the loop. Warm-start with N0 = 10 random architectures, evaluated and recorded, just so the ensemble
has enough points to fit something meaningful (with fewer than two points the predictor is hopeless, so I
keep sampling randomly until then). Then each remaining step: fit the five-MLP ensemble on the path-encoded
`(arch, val_acc)` pairs seen so far; draw 500 unseen random candidates; encode them; score each by the
ensemble mean; query the single argmax; record it and refit. Track the best-seen and return it. That is
BANANAS as this scaffold runs it: a path-encoded numpy-MLP ensemble with plain-MSE training, greedy ensemble-
mean acquisition, and a uniformly-random candidate pool — the surrogate replacing the GP's kernel, the path
encoding replacing adjacency, the ensemble supplying a cheap uncertainty I choose not to spend, and the random
pool supplying the global reach REA lacked (the distilled class is in the answer).

So the delta from REA is exactly the delta the REA variance demanded. Where REA could only mutate the best-
seen one edit at a time and so collapsed when the seeds were bad, BANANAS fits a predictor over all seen
points and uses it to reach across the whole space each step, querying the architecture the model believes is
best anywhere — not just nearby. Reading the REA numbers, here is what I expect and where I am unsure. The
clearest prediction is on *variance*: BANANAS should not crater the way REA did on ImageNet16-120 seed 2,
because even a poor warm-start leaves the predictor free to pull a strong architecture from anywhere in the
500-candidate global pool rather than being trapped 24 neighbors from the seeds — so I expect the
ImageNet16-120 spread to come *down* from REA's ±1.44, and the mean to hold at or above REA's 45.25. That is
the single most falsifiable claim, because REA's failure there was a variance failure and the pool addresses
exactly its cause. On CIFAR-100, where REA slipped below random search to 70.43 and even its best seed missed
random search's best, I expect BANANAS to recover above 70.6 and likely past random search's 70.65, because
global predictor-guided selection should beat both memoryless sampling and trapped local search on the setting
with a wider quality spread. On CIFAR-10 I again expect a near-wash around 93.4–93.5: the top is saturated, so
the most a global predictor can do is reliably land in the top cluster, worth a tenth or two but not more. The
signature I am looking for is BANANAS matching or slightly beating REA's *mean* on every setting while clearly
*tightening* the ImageNet16-120 variance — predictor-guided global selection buying robustness, not just
average accuracy. If instead BANANAS's mean is no better than REA's and its variance is no tighter, the
diagnosis would be that 29 path-encoded points are too few to train a predictor that generalizes across the
space, and the next move would be a better encoding or a zero-cost proxy to give the surrogate more signal per
query. But I expect the model to earn its place, because the sparsity of the path encoding is what lets a tiny
net generalize from a handful of points, and the whole point of compounding the queries into a surrogate is to
turn thirty lookups into a prediction over all 15,625 cells — precisely the reach REA's 42.70 seed proved it
was missing.
