The MLP landed at 0.9989 on random, 0.8461 on monotone, 0.9876 on sparse — geometric mean 0.9415, the
best so far, but the family breakdown overturns half of what I predicted and sharpens the one fault that
has now survived three different learners. The random family went to 0.9989, essentially solved: a narrow
10-term target over 30 variables is exactly where 256 hidden units with mixed-polarity weights fit nearly
perfectly, just as I expected, and against the 0.5034 base rate that is 0.4955 of the available 0.4966
earned — 99.8% of the recoverable gap closed. Sparse came in at 0.9876, *higher* than I feared — the
decay-driven soft selection did push the 48 irrelevant variables' weights down well enough that the MLP did
not waste much capacity on noise, so my sparse-risk worry was overcautious; it even edged out the forest's
0.9312 there. But monotone *dropped again*, to 0.8461 — below the forest's 0.8536 and below deep_dnf's
0.9088. That is the headline, and reading it through the base rate makes it worse, not better: monotone's
base rate is 0.6277, so 0.8461 is only `0.8461 − 0.6277 = 0.218` of earned accuracy out of an available
0.372 — barely 59% of the recoverable gap, the *lowest* earned fraction of any learner on any family in the
whole ladder. The wide 20-term monotone DNF has now defeated, in order, the differentiable DNF net (0.9088,
its best family but propped up by the base rate), the random forest (0.8536), and now the MLP (0.8461). My
prediction that the MLP's full connectivity would *recover* the monotone family was simply wrong, and the
wrongness is informative, because it refutes a specific mechanism. I had blamed the forest's monotone
failure on a feature-subset coverage bottleneck, and predicted the MLP — which has no such bottleneck — would
fix it. It did not. So the problem with the wide monotone target is *not* a feature-subset bottleneck: the
MLP has none and still failed, and it failed by an even wider margin. What survives both learners is that
fitting 20 overlapping conjunctions to high accuracy from a single *flat* training pass is genuinely hard —
whether the flat pass is bag-of-independent-trees or one-shot gradient descent on a fixed loss, the same
residual structure is left unmodeled.

Let me make the flat-learner diagnosis sharp, because it is what dictates the next move. Both failing
learners share a property: they fit the whole target in one undifferentiated pass over a fixed objective,
with no mechanism to notice "these particular examples are still wrong and deserve more attention." The
forest fits all 200 trees to the *same* labels in parallel; if random feature subsets cause the ensemble to
under-cover, say, three of the twenty terms, no tree is ever told those points are still misclassified — the
misclassified slice just sits there, averaged over, and the forest's vote on it stays wherever the majority
of trees happened to land. The MLP fits one cross-entropy loss in 20 fixed epochs; a rarely-satisfied term
contributes only its ~16-in-256 sliver of each batch's gradient, and once the frequently-firing structure is
fit that sliver is a small fraction of the total gradient, so the optimizer plateaus with the hard terms
half-learned. Different architectures, identical failure shape: the wide monotone target has many terms
competing for a fixed budget of fitting effort, and a flat learner spreads that budget by the *frequency* of
each region rather than by where the *error* currently is, so the hardest, rarest terms are chronically
starved. The geometric mean makes this the decisive family: with `∂G/∂x_i = G/(3 x_i)`, the smallest factor
moves the mean most, and monotone at 0.8461 has by far the highest sensitivity — `0.9415/(3·0.8461) = 0.371`
per point against `~0.314` for the two near-ceiling families — so the entire aggregate is now hostage to this
one number. Nothing flat has cracked it, and the geomean will not clear 0.95 until something does.

So I need a learner that does not fit the target in one flat pass but builds it up *sequentially*, each new
component explicitly correcting the errors the ensemble has made so far. That is boosting, and it is the one
idea on the table that the forest deliberately did not use: a random forest bags *independent* trees and
averages them, so no tree ever sees another tree's mistakes; on a wide target where the average
under-models some terms, there is no mechanism to go back and put more weight on the still-misclassified
examples. Gradient boosting is exactly that mechanism, and it keeps the property that made trees beat the MLP
on the random family — exact axis-aligned conjunctive splits, one root-to-leaf path per DNF term — while
adding the sequential error-correction the MLP and the forest both lack. It is the natural synthesis: the
tree's representation (which held random and sparse at ceiling) plus a fitting procedure that reallocates
effort to the residual (which is what monotone has needed all along).

Let me reconstruct the boosting argument, because the choice of *gradient* boosting over the older
loss-specific schemes is what makes it general and stable here. I want an additive model `F(x) = Σ_m ρ_m
h(x; a_m)`, each `h` a small regression tree, fit greedily one term at a time (forward stagewise: hold
`F_{m-1}` fixed, add one tree). For squared error the stage subproblem is "fit the next tree to the residual
`y − F_{m-1}`" — the classic residual loop. But for classification I want the binomial deviance
`log(1 + e^{-2yF})`, whose stage subproblem has no closed form. The move that rescues it: treat the function
values at the training points as the parameters and take the negative gradient of the loss with respect to
them — a vector of *pseudo-responses*, one per point, defined for any differentiable loss. The unconstrained
negative gradient lives only at the data, so I cannot use it as a model; instead I fit the next tree to it by
least squares — the tree most parallel to the negative gradient over the data — and then choose the step size
by a one-dimensional line search on the *true* deviance. Cheap least-squares to find a generalizable descent
direction, then an honest 1-D step on the real loss. For classification the pseudo-response at each point is
`2y_i / (1 + e^{2 y_i F_{m-1}(x_i)})` — and reading that formula is the whole point: where the current model
is confidently *right* (`y_i F` large positive) the exponential blows up and the pseudo-response is ≈ 0, so a
correctly-classified point exerts no pull; where the model is confidently *wrong* (`y_i F` large negative)
the denominator → 1 and the pseudo-response is ≈ `2 y_i`, its maximum magnitude. So each new tree is fit, by
least squares, to a target that is *large exactly on the still-misclassified points and near-zero everywhere
else*. That is the error-correction the monotone family has been missing: a misclassified region from one of
the 20 under-covered terms gets a large pseudo-response, the next tree carves precisely that region, the
deviance on it comes down, and the round after that moves on to whatever is now most wrong. The fitting
budget follows the error, not the frequency — the exact inversion of the flat-learner failure I diagnosed.

Three pieces of the recipe matter for *this* task and are exactly what the scaffold fill sets. First, **tree
depth tied to the target width.** Each tree is grown to depth `max(4, term_width + 1) = 5` here. A width-4
DNF term is a length-4 conjunction, so a depth-4 path can already pin all four of a term's variables, and
depth 5 gives one extra level of slack — `2^5 = 32` leaves per tree, enough to isolate a term along one path
while the other leaves handle the complementary structure. Deep enough to represent a term exactly, shallow
enough that each tree stays a *weak* learner and the boosting does the composition rather than any single
tree memorizing. This is the task-specific knob, and it reads `config.term_width` to set itself — the learner
adapts its weak-learner capacity to the announced conjunction width without ever inspecting the hidden term
list. Contrast a generic depth-3 default: it could not pin four literals in one path and would under-fit
every term; a generic very-deep tree would stop being weak and would overfit each round, defeating the
shrinkage. Depth 5 is the width-matched sweet spot. Second, **shrinkage and many rounds.** A small learning
rate (`0.05`) scales down each tree's contribution so no single tree overshoots, and `n_estimators=500`
rounds give the ensemble enough small steps to drive the deviance down. The arithmetic of why this helps the
wide family: with 20 terms to install and up to 500 rounds, that is a budget of ~25 rounds per term, and
because the pseudo-response keeps re-pointing at whatever is currently most wrong, those rounds get spent on
the terms that need them rather than re-fitting the easy ones — shrinkage plus many rounds is the well-known
regularization that beats a few large steps, trading compute for generalization. Third, **stochastic
subsampling and early stopping.** `subsample=0.9` fits each tree on a random 90% of the data (stochastic
gradient boosting), which decorrelates the trees a little and regularizes; `n_iter_no_change=25` with a 10%
internal validation split (2000 of the 20000 examples held out, 18000 used to fit each round) and `tol=1e-5`
stops the boosting once the held-out deviance stops improving. That early-stop is what lets one setting serve
all three families: on the *easy* families (random, sparse), where a handful of rounds already reaches the
ceiling, it halts long before 500 and does not overfit; on the hard monotone family it keeps boosting as
long as the residual structure is still being reduced, spending its full round budget where it is needed. The
starting model `F_0` is the best constant (the base-rate logit), and the final prediction thresholds the
additive logit at 0.

I should stress-test the boosting story against one failure mode before I commit, because it is the obvious
objection: could the sequential fitting *overfit* the monotone family the way a single deep tree does, memor-
izing training points instead of recovering terms? Two guards answer it. The trees are weak (depth 5, `32`
leaves against 20000 points, so each leaf averages hundreds of examples — no single-point memorization like
an unpruned forest tree), and the early-stopping validation split watches held-out deviance, so the moment
extra rounds start fitting noise rather than terms the boosting halts. The line search on the true deviance
also keeps each step honest — it will not take a large step that helps train while hurting the loss geometry.
So the regularization is layered (weak learners, shrinkage, subsample, early stop), and the concern that
sequential correction just overfits is met by construction. The residual-chasing is aimed at systematic
error — the under-covered terms — not at individual noisy labels, because a single mislabeled point cannot
sustain a large pseudo-response across many rounds the way a whole unmodeled term's region can.

The contrast with the random forest is worth making precise, because both are tree ensembles and the naive
reading is "more trees, same idea." They are opposites in how they combine. The forest *decorrelates and
averages*: it deliberately makes its trees disagree (bootstrap plus `sqrt(n)` feature subsets) and then
takes the mean, so its whole engine is variance reduction on a set of independently-grown, full-depth,
low-bias trees — and its ceiling is the correlation floor `ρσ²`, which on the wide monotone target it could
not lower without starving the sparse junta. Boosting does the reverse: its trees are *weak* (depth-5,
high-bias) and *dependent by construction* — each is grown to fix the previous ensemble's residual — so its
engine is bias reduction, driving down a systematic error that the forest's averaging leaves in place. That
is why boosting is the right tool for monotone specifically: the forest's monotone failure was never variance
(200 trees had crushed that), it was the residual bias of under-covered terms, and bias is exactly what
sequential residual-fitting removes. The forest had no lever for it; boosting's entire mechanism is that
lever. So this rung is not "a better forest," it is the complementary ensemble that attacks the error
component the forest structurally cannot touch — while keeping the exact-conjunctive-split representation
that made trees beat the MLP's smeared weights on random in the first place.

I should also price the compute, because gbdt gives up the forest's greatest practical virtue. The forest fit
in about 0.67 seconds a family — 200 trees grown fully in *parallel* (`n_jobs=-1`) on a tabular problem. Gbdt
is inherently *sequential*: round `m` cannot start until round `m−1`'s tree is fit and the pseudo-responses
recomputed, so up to 500 rounds run one after another with no parallelism across rounds, and each round fits
a fresh depth-5 tree on 18000 examples plus a line search. That is many more, deeper, sequential fits than
the forest's parallel shallow-in-aggregate work, so I expect gbdt's fit time to land in the tens of seconds —
comparable to deep_dnf's 12-to-38, an order of magnitude above the forest — with the *hard* monotone family
running the longest precisely because early stopping lets it use its full round budget while the easy families
halt early. Under the task's soft wall-clock budget that is acceptable: this is the top of the ladder, the
place to spend compute, and the sequential cost is the direct price of the error-correction that the parallel
forest could not buy at any tree count.

There is one more reason to expect gbdt to hold the easy families where the MLP had to work for them. On
sparse, the MLP needed decay-driven soft selection to suppress 48 noise variables and still leaked a little
capacity onto them; a boosted tree, like any tree, simply never selects a noise variable for a split because
a split on a coordinate uncorrelated with the residual yields no deviance reduction and loses the greedy
least-squares competition to a relevant one. So gbdt gets junta-irrelevance for free, the native tree
advantage the forest already demonstrated at 0.9312 and the MLP only approximated at 0.9876 — I would expect
gbdt at or above the MLP's sparse number without spending any mechanism on variable selection. Random is
even safer: 10 terms over 30 variables is narrow, mixed polarity is free for a split, and with residual
correction on top of exact splits I would be surprised to see anything but a near-perfect number, plausibly
the 1.0 the MLP nearly reached.

In the scaffold this is a clean fill: `build_model` returns the `sklearn` `GradientBoostingClassifier` with
those settings, `make_dataset` is the default uniform sample, `fit_and_predict` calls `.fit` and `.predict`
and returns the 0/1 vector — and notably it reads `config.term_width` to set the tree depth, so the learner
adapts to the announced conjunction width without ever peeking at the hidden terms. The full module is in the
answer.

Now the falsifiable expectations against the MLP's numbers, family by family. On **monotone** (the
bottleneck — MLP 0.8461, the worst standing number and the lowest earned-fraction anywhere), I expect gbdt to
*finally crack it*. This is the whole reason for the rung: 20 terms is a lot of conjunctions, but boosting
fits them one residual at a time, a depth-5 tree per round can isolate each term, and with up to 500 shrunk
rounds and early stopping to let the hard family run long, the deviance should be driven down until nearly
all 20 terms are covered — I am expecting a jump from the mid-0.84s into the high 0.90s. If gbdt does *not*
substantially beat 0.8461 on monotone, then my entire account — that the monotone failure is a flat-fitting
problem cured by sequential error-correction — is wrong, and I would have to conclude the wide target is just
intrinsically hard to learn from 20000 examples regardless of algorithm. On **random** (MLP 0.9989,
near-solved) and **sparse** (MLP 0.9876), I expect gbdt to be *at least as good*: exact conjunctive splits
handle mixed polarity for free (random) and simply never split on irrelevant variables (sparse, the tree's
native advantage over the MLP that already showed up in the forest's 0.9312), and early stopping guards
against overfitting these easy families. I would expect both up near or above the MLP's numbers, with random
plausibly hitting 1.0.

So the prediction I am committing to: gbdt should post the highest geometric mean of the ladder, above the
MLP's 0.9415, and the gain should come *predominantly from the monotone family* — the one family that has
resisted every flat learner — with random and sparse holding at or near their ceilings. By the geomean
sensitivity, monotone is where the leverage is: lifting it from 0.8461 toward the high 0.90s while the other
two hold near 1.0 would push the aggregate up by roughly `0.37 × 0.13 ≈ 0.048`, into the high 0.98s. If the
monotone number jumps like that while random and sparse stay near-perfect, the trajectory's thesis is
confirmed: on uniform-distribution DNF learning, the winning learner is not the one with the DNF inductive
bias hand-coded in (deep_dnf was weakest), nor flat averaging (rf) or flat gradient descent (mlp), but a tree
ensemble that keeps exact conjunctive splits *and* corrects its own residual errors round by round. That is
the rung I expect to end on, and it is the strongest baseline the task provides.
