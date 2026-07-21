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

The flat-learner diagnosis dictates the next move. Both failing
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
starved. By the geomean sensitivity `∂G/∂x_i = G/(3 x_i)`, monotone at 0.8461 is now by far the highest-
leverage factor (0.371 per point against ~0.314 for the near-ceiling families), so the aggregate is
hostage to this one number and will not clear 0.95 until something cracks it.

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

The choice of *gradient* boosting over the older loss-specific schemes is what makes it general and stable
here. I want an additive model `F(x) = Σ_m ρ_m
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

The obvious objection is that sequential fitting could *overfit* monotone the way a single deep tree does.
Two guards answer it. The trees are weak — depth 5, 32 leaves against 20000 points, so each leaf averages
hundreds of examples, no single-point memorization — and the early-stop validation split halts the moment
extra rounds fit noise rather than terms. The residual-chasing is aimed at systematic error (the under-
covered terms), not individual noisy labels: a single mislabeled point cannot sustain a large pseudo-
response across many rounds the way a whole unmodeled term's region can.

Both are tree ensembles, but they are opposites in how they combine, and that is the point. The forest
decorrelates and averages full-depth low-bias trees — its engine is variance reduction, and its ceiling is
the correlation floor `ρσ²` it could not lower on monotone without starving the sparse junta. Boosting's
trees are weak (depth-5, high-bias) and dependent by construction, so its engine is bias reduction, driving
down exactly the residual bias of under-covered terms that the forest's averaging left in place. The
forest had no lever for that; boosting's whole mechanism is that lever — while keeping the exact
conjunctive splits that made trees beat the MLP on random.

Gbdt gives up the forest's practical virtue: where the forest grew 200 trees in *parallel* (0.67s a
family), boosting is inherently *sequential* — round `m` waits on round `m−1`'s tree and pseudo-responses,
up to 500 rounds each fitting a depth-5 tree on 18000 examples plus a line search. So I expect fit time in
the tens of seconds, an order of magnitude above the forest, with the *hard* monotone family running
longest because early stopping lets it use its full round budget while the easy families halt early. Under the task's soft wall-clock budget that is acceptable: this is the top of the ladder, the
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

The fill is clean: `GradientBoostingClassifier` with those settings on the default uniform sample, `.fit`
then `.predict` — and it reads `config.term_width` to set the tree depth, adapting to the announced
conjunction width without ever peeking at the hidden terms. The full module is in the answer.

Expectations against the MLP. On **monotone** (the bottleneck, 0.8461) gbdt should *finally crack it*: 20
terms fit one residual at a time, a depth-5 tree isolates each, and with up to 500 shrunk rounds and early
stopping to run the hard family long, the deviance should fall until nearly all 20 terms are covered — a
jump into the high 0.90s. If gbdt does *not* substantially beat 0.8461 there, my whole account — monotone
as a flat-fitting problem cured by sequential correction — is wrong, and the wide target is intrinsically
hard regardless of algorithm. On **random** (0.9989) and **sparse** (0.9876) I expect at least as good:
exact splits handle mixed polarity free and never split on irrelevant variables, and early stopping guards
the easy families. If that monotone jump lands while the others hold, the thesis is confirmed: for
uniform-distribution DNF the winning learner is not the hand-coded DNF bias, nor flat averaging or flat
gradient descent, but a tree ensemble that keeps exact conjunctive splits *and* corrects its own residual
errors round by round.
