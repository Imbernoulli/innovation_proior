REA's numbers landed almost exactly where my falsifiable prediction said the risk was, and reading them
carefully is what picks the next rung. The headline is that REA did *not* clearly beat random search:
CIFAR-10 went 93.38 → 93.40 (a wash, as I predicted — the compressed top is near-saturated and there is
no room), CIFAR-100 actually *slipped* 70.65 → 70.43, and ImageNet16-120 rose 44.57 → 45.25, the one
setting where I expected the most room. So the directional prediction held on ImageNet16-120 and CIFAR-10,
but the CIFAR-100 slip and — more telling — the ImageNet16-120 **variance** are the real story. REA's
ImageNet spread blew up to ±1.44, more than double random search's ±0.58, with seed 2 cratering to 42.70
while seed 0 hit 46.80. That ±1.44 is the diagnosis written in a number: this is exactly the
premature-convergence failure I worried about when I chose the eviction rule. On the good seeds, the 20
evolution steps hill-climbed a strong seed into a genuinely better cell (46.8, 46.4 — clearly above
random search's best). On seed 2, the tournament locked onto a mediocre seeding region and single-edge
mutation could not climb out of it in 20 steps, so it finished at 42.70, *below* anything random search
drew. REA is high-variance because it is a *local* search: it exploits beautifully when the seeds put it
near the good region and fails badly when they don't, and 10 random seeds plus 20 single-edge steps is
not enough coverage to guarantee the former. The mechanism that helped — exploit the best-seen by
mutating it — is the same mechanism that hurt, because mutation only ever moves one edit from something
already evaluated. It cannot *extrapolate*; it can only refine what luck handed it.

So the lever the REA numbers expose is different from the one the random-search numbers exposed. Random
search needed exploitation; REA has exploitation but no way to use the 30 queries to reason about
architectures it has *not* yet evaluated. What I want now is a method that, after each query, builds a
*model* of the accuracy surface from all the `(arch, acc)` pairs seen so far and uses it to choose the
next query — so that the 30 queries compound into a prediction over the whole 15,625-cell space, not just
a 1-edit neighborhood of the current best. That is Bayesian optimization in spirit: model what you have
seen, pick the next point to maximize expected progress, evaluate, update. The question is what the
surrogate, the encoding, and the selection rule should be under this budget and inside this scaffold.

Start with the surrogate, because the obvious choice fails. The textbook BO surrogate is a Gaussian
process: condition on the data, read off a posterior mean and variance. But a GP *is* its kernel, and the
inputs here are labeled cell graphs — there is no off-the-shelf kernel on DAGs, and building a bespoke
distance over architectures is a hard modeling problem of its own, plus GP inference is cubic in the
number of observations. I do not need any of that. All I actually require from the surrogate is: consume
the architectures seen, predict the validation accuracy of an unseen one, and let me pick the most
promising candidate. A neural network does exactly that and learns its own notion of similarity from a
feature vector, so the "invent a kernel on DAGs" problem evaporates — I hand the net a featurization and
let it fit. So the surrogate is a small feedforward network trained from the `(encoding, val_acc)` pairs
at each step.

The encoding is the next decision, and the scaffold has already made the right one available. The default
move would be an adjacency encoding — a bit per possible edge plus the op at each node — but that is
order-dependent and its features are violently inter-dependent (an edge bit means nothing without a path
to the output), which is exactly what a small net struggles to fit from a handful of examples. The
scaffold instead provides `path_encoding`, a binary indicator over input→output operation-paths: one
feature per "the tensor can flow through this sequence of operations," length `5 + 5^2 + 5^3 = 155`. Each
feature is a self-contained statement about what the cell computes rather than a fragment of its wiring,
so the features are far less entangled and a single architecture maps to a single encoding — no
isomorphism ambiguity. With 155 dimensions and a 30-cell budget the full path encoding is small enough to
use as-is; there is no need to truncate it to the most-frequent short paths, the way one would on a large
cell where the encoding length explodes — here it does not, so I take the whole 155-dim vector the loop
hands me.

Now the predictor itself, and here I have to respect what this scaffold can actually run. I cannot assume
a deep-learning framework is in the image, so I build the net in numpy: a two-layer MLP — Linear(155, 64)
→ ReLU → Linear(64, 1) — trained with Adam on a plain mean-squared-error objective against the validation
accuracies. Why MSE and not a fancier loss that up-weights the high-accuracy architectures? Because at
this budget, with at most 29 training points by the final step, the predictor's job is modest: rank a
pool of candidates well enough to pick a good one, not to nail the error of near-optimal cells to the
decimal. A plain MSE regressor on the 155-dim path features is the honest, low-variance choice — it fits
fast (200 Adam steps, lr = 1e-2), it has no extra hyperparameter to tune against a budget that cannot
afford tuning, and it predicts accuracy directly so I can simply pick the candidate the model thinks is
best. The width 64 and the 1/√fan-in initialization keep it from overfitting the tiny training set.

Every acquisition rule wants more than a point prediction — it wants an uncertainty, so the search can
explore where the model is ignorant. A single net gives only a point estimate. The cheap, well-calibrated
way to get an uncertainty is an ensemble: train M independent copies from different random
initializations and read the disagreement among them as the uncertainty. I set M = 5 — small enough to
retrain from scratch every step (the predictor's cost is a rounding error against the real query cost,
since a query is just a table lookup but is the budgeted resource), large enough that the ensemble mean
is a stable estimate. Each member gets its own seed (`self.seed + i + 1`) so the members genuinely
disagree.

Now the selection rule, and this is the place where I deliberately keep it simple rather than reaching for
the full BO acquisition machinery. With the ensemble I have, for any candidate, the mean prediction across
the 5 members. The acquisition I use is the plainest exploitation rule: **score every candidate by the
ensemble mean and query the single highest-scoring one.** I considered the uncertainty-aware
alternatives — upper-confidence-bound (`mean + β·std`), or Thompson-style sampling from each candidate's
predictive spread — and they are the principled way to balance exploration against exploitation. But here
the exploration is already being supplied by *where the candidates come from*, and adding an explicit
uncertainty bonus on top of a 29-point predictor risks chasing the model's high-variance phantoms into
regions it has no business trusting. So I let the candidate pool carry the exploration and let the
acquisition be pure greedy exploitation of the mean — which is also robust to a poorly-calibrated
ensemble std at tiny sample sizes.

The candidate pool is the last piece and the one that most distinguishes this rung from REA. REA's fatal
limitation was that it only ever looked one edit away from an evaluated architecture, so it could not
escape a bad seeding region — that is what produced the 42.70 seed. To fix exactly that, I draw the
candidate pool **uniformly at random over the whole space**: each step I sample a large pool (500) of
*unseen* random architectures and score all of them with the ensemble. This is the deliberate opposite of
REA's mutate-the-best: instead of refining the current neighborhood, the predictor gets to reach across
the entire 15,625-cell space and pull in whichever architecture it predicts is best, even if it is many
edits away from anything seen. The model is what generalizes from the seen points to the unseen pool, so
a single good predictor lets me extrapolate where REA could only interpolate locally. The pool is large
(500) so the argmax is taken over a broad sweep of the space rather than a thin sample; it is cheap
because scoring 500 candidates with five tiny MLPs is nothing against the budgeted query.

Assemble the loop. Warm-start with N0 = 10 random architectures, evaluated and recorded, just so the
ensemble has enough points to fit something meaningful (with fewer than two points the predictor is
hopeless, so I keep sampling randomly until then). Then each remaining step: fit the 5-MLP ensemble on
the path-encoded `(arch, val_acc)` pairs seen so far; draw 500 unseen random candidates; encode them;
score each by the ensemble mean; query the single argmax; record it and refit. Track the best-seen and
return it. That is BANANAS as this scaffold runs it: a path-encoded numpy-MLP ensemble with plain-MSE
training, greedy ensemble-mean acquisition, and a uniformly-random candidate pool — the surrogate
replacing the GP's kernel, the path encoding replacing adjacency, the ensemble supplying a cheap
uncertainty, and the random pool supplying the global reach REA lacked (the distilled class is in the
answer).

So the delta from REA is exactly the delta the REA variance demanded. Where REA could only mutate the
best-seen one edge at a time and so collapsed when the seeds were bad, BANANAS fits a predictor over all
seen points and uses it to reach across the whole space each step, querying the architecture the model
believes is best anywhere — not just nearby. Reading the REA numbers, here is what I expect and where I am
unsure. The clearest prediction is on **variance**: BANANAS should not crater the way REA did on
ImageNet16-120 seed 2, because even a poor seeding draw leaves the predictor free to pull in a strong
architecture from anywhere in the space rather than being trapped one edit from the seeds — so I expect
the ImageNet16-120 spread to come *down* from REA's ±1.44, and the mean to hold at or above REA's 45.25.
On CIFAR-100, where REA slipped below random search to 70.43, I expect BANANAS to recover above 70.6 and
likely past random search's 70.65, because global predictor-guided selection should beat both memoryless
sampling and trapped local search on the setting with a wider quality spread. On CIFAR-10 I again expect
a near-wash around 93.4–93.5: the top is saturated, so the most a global predictor can do is reliably land
in the top cluster, which is worth a tenth or two but not more. The signature I am looking for is BANANAS
matching or slightly beating REA's *mean* on every setting while clearly *tightening* the ImageNet16-120
variance — predictor-guided global selection buying robustness, not just average accuracy. If instead
BANANAS's mean is no better than REA's and its variance is no tighter, the diagnosis would be that 29
path-encoded points are too few to train a predictor that generalizes across the space, and the next move
would be a better encoding or a zero-cost proxy to give the surrogate more signal per query — but I expect
the model to earn its place, because the whole point of compounding the queries into a surrogate is to
turn 30 lookups into a prediction over all 15,625 cells, which is precisely the reach REA's 42.70 seed
proved it was missing.
