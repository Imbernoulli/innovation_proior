The MLP landed at 0.9989 on random, 0.8461 on monotone, 0.9876 on sparse — geometric mean 0.9415, the
best so far, but the family breakdown overturns half of what I predicted and sharpens the one fault that
has now survived three different learners. The random family went to 0.9989, essentially solved: a narrow
10-term target over 30 variables is exactly where 256 hidden units with mixed-polarity weights fit
nearly perfectly, just as I expected. Sparse came in at 0.9876, *higher* than I feared — weight decay
did push the 48 irrelevant variables' weights toward zero well enough that the MLP did not waste much
capacity on noise, so my sparse-risk worry was overcautious. But monotone *dropped again*, to 0.8461 —
below the forest's 0.8536 and below deep_dnf's 0.9088. That is the headline. The wide 20-term monotone
DNF has now defeated, in order, the differentiable DNF net (well, it was deep_dnf's *best* family at
0.9088, but still not solved), the random forest (0.8536), and now the MLP (0.8461). My prediction that
the MLP's full connectivity would *recover* the monotone family was simply wrong, and the wrongness is
informative: the problem with the wide monotone target is not a feature-subset bottleneck (the MLP has
none and still failed) — it is that fitting 20 overlapping conjunctions to high accuracy from a single
flat training pass is genuinely hard, and every *flat* learner I have tried — bag-of-trees, one-shot
gradient descent on a fixed loss — leaves the same residual structure unmodeled. The monotone family is
the bottleneck that decides the geometric mean, and nothing flat has cracked it.

So I need a learner that does not fit the target in one flat pass but builds it up *sequentially*, each
new component explicitly correcting the errors the ensemble has made so far. That is boosting, and it is
the one idea on the table that the forest deliberately did not use: a random forest bags *independent*
trees and averages them, so no tree ever sees another tree's mistakes; on a wide target where the
average under-models some terms, there is no mechanism to go back and put more weight on the
still-misclassified examples. Gradient boosting is exactly that mechanism, and it keeps the property that
made trees beat the MLP on the random family — exact axis-aligned conjunctive splits, one root-to-leaf
path per DNF term — while adding the sequential error-correction the MLP and the forest both lack.

Let me reconstruct the boosting argument, because the choice of *gradient* boosting over the older
loss-specific schemes is what makes it general and stable here. I want an additive model
`F(x) = Σ_m ρ_m h(x; a_m)`, each `h` a small regression tree, fit greedily one term at a time
(forward stagewise: hold `F_{m-1}` fixed, add one tree). For squared error the stage subproblem is
"fit the next tree to the residual `y − F_{m-1}`" — the classic residual loop. But for classification I
want the binomial deviance `log(1 + e^{-2yF})`, whose stage subproblem has no closed form. The move that
rescues it: treat the function values at the training points as the parameters and take the negative
gradient of the loss with respect to them — a vector of *pseudo-responses*, one per point, defined for
any differentiable loss. The unconstrained negative gradient lives only at the data, so I cannot use it
as a model; instead I fit the next tree to it by least squares — the tree most parallel to the negative
gradient over the data — and then choose the step size by a one-dimensional line search on the *true*
deviance. Cheap least-squares to find a generalizable descent direction, then an honest 1-D step on the
real loss. For classification the pseudo-response at each point is `2y_i / (1 + e^{2 y_i F_{m-1}(x_i)})`
— large where the current model is confidently wrong, near zero where it is already right — so each new
tree is literally fit to *where the ensemble is still making mistakes*. That is the error-correction the
monotone family has been missing: a misclassified region from one of the 20 terms gets a large
pseudo-response, the next tree carves it out, and the deviance comes down term by term.

Contrast this with the two flat learners that failed on monotone, because the contrast is the whole
argument. The random forest fits all its trees to the *same* labels in parallel; if the random feature
subsets cause the ensemble to under-model, say, three of the twenty terms, no tree is ever told "these
points are still wrong" — the misclassified slice just sits there, averaged over, and the forest's
prediction on it stays at whatever the majority of trees happened to vote. The MLP fits one loss in a
fixed number of passes; a rarely-satisfied term contributes a thin slice of the gradient that the
frequently-firing structure drowns out, so the optimizer plateaus with those terms half-learned. Boosting
breaks both failure modes by construction: the pseudo-response *re-weights* attention onto exactly the
still-wrong points every round, so the slice that the forest averaged away and the MLP's gradient
neglected becomes the *largest* signal driving the next tree. The wide monotone target is hard precisely
because it has many terms competing for a fixed budget of fitting effort; boosting reallocates that
budget, round by round, to wherever the residual error currently is.

Three pieces of the recipe matter for *this* task and are exactly what the scaffold fill sets. First,
**tree depth tied to the target width.** Each tree is grown to depth `max(4, term_width + 1)` = 5 here.
A width-4 DNF term is a length-4 conjunction, so a depth-4-to-5 tree can isolate a single term per
root-to-leaf path with one extra level of slack — deep enough to represent a term exactly, shallow
enough that each tree stays a *weak* learner and the boosting does the composition. This is the
task-specific knob: the depth is chosen to match the conjunction width, not left at a generic default.
Second, **shrinkage and many rounds.** A small learning rate (`0.05`) scales down each tree's
contribution so no single tree overshoots, and `n_estimators=500` rounds give the ensemble enough small
steps to drive the deviance down — shrinkage plus many rounds is the well-known regularization that beats
a few large steps, trading compute for generalization. Third, **stochastic subsampling and early
stopping.** `subsample=0.9` fits each tree on a random 90% of the data (stochastic gradient boosting),
which decorrelates the trees a little and regularizes; `n_iter_no_change=25` with a 10% internal
validation split and `tol=1e-5` stops the boosting once the held-out deviance stops improving, so on the
*easy* families (random, sparse) it does not waste 500 rounds overfitting, while on the hard monotone
family it keeps boosting as long as the residual structure is still being reduced. The starting model
`F_0` is the best constant (the base-rate logit), and the final prediction thresholds the additive logit
at 0.

In the scaffold this is a clean fill: `build_model` returns the `sklearn` `GradientBoostingClassifier`
with those settings, `make_dataset` is the default uniform sample, `fit_and_predict` calls `.fit` and
`.predict` and returns the 0/1 vector — and notably it reads `config.term_width` to set the tree depth,
so the learner adapts its weak-learner capacity to the announced conjunction width without ever
inspecting the hidden term list. The full module is in the answer.

Now the falsifiable expectations against the MLP's numbers, family by family. On **monotone** (the
bottleneck — MLP 0.8461, the worst standing number), I expect gbdt to *finally crack it*. This is the
whole reason for the rung: 20 terms is a lot of conjunctions, but boosting fits them one residual at a
time, and a depth-5 tree per round can isolate each term; with up to 500 shrunk rounds and early
stopping to let the hard family run long, the deviance should be driven down until nearly all 20 terms
are covered. If gbdt does *not* substantially beat 0.8461 on monotone, then my entire account — that the
monotone failure is a flat-fitting problem cured by sequential error-correction — is wrong, and I would
have to conclude the wide target is just intrinsically hard to learn from 20000 examples regardless of
algorithm. On **random** (MLP 0.9989, near-solved) and **sparse** (MLP 0.9876), I expect gbdt to be
*at least as good*: exact conjunctive splits handle mixed polarity for free (random) and simply never
split on irrelevant variables (sparse, the tree's native advantage over the MLP), and early stopping
guards against overfitting these easy families. I would expect both up near or above the MLP's numbers,
with random plausibly hitting 1.0.

So the prediction I am committing to: gbdt should post the highest geometric mean of the ladder, above
the MLP's 0.9415, and the gain should come *predominantly from the monotone family* — the one family
that has resisted every flat learner — with random and sparse holding at or near their ceilings. If the
monotone number jumps from the mid-0.84s into the high 0.90s while random and sparse stay near-perfect,
the trajectory's thesis is confirmed: on uniform-distribution DNF learning, the winning learner is not
the one with the DNF inductive bias hand-coded in (deep_dnf was weakest), nor flat averaging (rf) or
flat gradient descent (mlp), but a tree ensemble that keeps exact conjunctive splits *and* corrects its
own residual errors round by round. That is the rung I expect to end on, and it is the strongest baseline
the task provides.
