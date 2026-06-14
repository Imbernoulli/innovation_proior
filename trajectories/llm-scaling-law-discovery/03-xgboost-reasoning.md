The symbolic rung did exactly what I predicted on the two easy families and exactly what I feared on the
hard one, and the numbers say it precisely. Vocab swung from kernel ridge's $-0.567$ to $0.929$, and
dataconstrained from $-13.4$ to $0.929$ — both rescued, because the power-law-plus-floor backbone and the
repeat-efficiency effective-token term finally gave the law something that extrapolates to the
larger/denser test points instead of collapsing off the hull. So the diagnosis that "the missing
ingredient was the asymptotic form, not flexibility" was right for those families. But lrbsz came back at
$R^2 = -3.052$ — still negative, still worse than predicting the mean. And the secondary metrics tell the
real story: the lrbsz `MAE` dropped from kernel ridge's $0.619$ all the way to $0.0629$, and `RMSE` from
$0.777$ to $0.0768$, an order-of-magnitude improvement in absolute error. So the hand-shaped basin is
*close* in absolute terms — it is predicting loss values that are tiny fractions off — but it still loses
to the trivial mean in $R^2$, which means it is getting the *ranking* of the held-out points wrong even
while the magnitudes are nearly right. That is the signature of exactly the failure I flagged: a basin
with a single fitted center $(\log l^\star, \log b^\star)$ cannot follow an optimum that drifts with
scale. Out-of-sample the bowl is centered in the wrong place, so the quadratic penalty it adds has the
wrong sign of curvature relative to the true surface, and the predicted ordering of held-out
configurations is off even though the absolute numbers are small (the lrbsz target has a tiny spread —
that is why `NMAE` is $1.65$, the error is small absolutely but large relative to how little the target
varies). So the symbolic law has plateaued: I cannot get lrbsz positive by tweaking a hand-shaped form,
because the problem is that one fitted center cannot track a moving optimum.

This reopens the question I thought I had closed at the first rung. I went black-box-then-symbolic
because the black box (kernel ridge) collapsed off the hull. But the *reason* kernel ridge collapsed was
specific — RBF locality, kernel values decaying to zero far from training points, no notion of a floor.
It was not that flexible learners are wrong here; it was that *that* flexible learner had no way to carry
structure into the extrapolation region. So the right question for this rung is: is there a flexible,
model-free learner whose inductive bias does *not* collapse off the hull, and that can represent a
scale-dependent basin and cross-axis interactions without my hand-shaping a single center? If there is,
it might beat the symbolic law on lrbsz — where the symbolic law's rigidity is the bottleneck — while
holding the easy families. The candidate I keep coming back to on structured tabular data is a boosted
ensemble of regression trees. Let me derive why it fits, from what trees actually do.

A regression tree partitions the input space into axis-aligned cells and predicts a constant per cell. By
itself that is a crude, high-variance learner. But boosting builds an additive ensemble
$\hat y_i = \sum_k f_k(x_i)$ where each $f_k$ is a tree fit to correct the residual of the ensemble so
far, and the magic is in *how* each tree is grown. The modern formulation regularizes the whole thing:
minimize $\sum_i l(\hat y_i, y_i) + \sum_k \Omega(f_k)$ with $\Omega(f) = \gamma T + \tfrac12\lambda
\lVert w\rVert^2$ — a per-leaf cost $\gamma T$ and an L2 penalty on the leaf scores. Fit additively: at
round $t$, second-order Taylor-expand the loss around the current prediction so the per-round objective
depends on the loss only through the gradient $g_i$ and curvature $h_i$, and the optimal leaf weight is
$w_j^\star = -G_j/(H_j + \lambda)$ with $G_j, H_j$ the gradient and curvature sums in the leaf — a
Newton step damped by $\lambda$ so an under-populated leaf cannot blow up. The split gain
$\tfrac12[G_L^2/(H_L+\lambda) + G_R^2/(H_R+\lambda) - G^2/(H+\lambda)] - \gamma$ scores every candidate
split, and $\gamma$ doubles as the prune threshold (a split that does not clear $\gamma$ is not made).
That is the engine. Why does it fit *this* problem where kernel ridge did not?

First, the inductive bias off the hull is fundamentally different from an RBF kernel's. A tree does not
decay to zero away from training points — it predicts the constant of whichever leaf the query falls
into, which for an out-of-region query is the constant of the *nearest boundary cell*. So instead of
collapsing toward zero, a boosted-tree prediction *flattens to the last seen value* in that direction. On
a loss surface that saturates toward a floor, "flatten to the boundary value" is a far better
extrapolation than "decay to zero" — it is a crude version of the floor the symbolic law imposes
explicitly. So the structural reason kernel ridge failed (collapse off the hull) is exactly the reason a
tree ensemble might not.

Second — and this is what could finally crack lrbsz — trees represent basins and cross-axis interactions
*natively*, with no hand-shaped center. The basin in $(\log l, \log b)$ is just a region of low loss
surrounded by higher loss; an axis-aligned partition carves that region out as a set of cells, and the
ensemble assigns each cell its own constant, so it reconstructs the bowl as a staircase without ever
fitting a quadratic or its center. More importantly, because every split is conditional on the splits
above it, a tree captures interactions for free: a split on $\log l$ *inside* a branch already split on
$N$ is precisely "the best learning rate depends on the model scale" — the scale-dependent optimum drift
that broke the symbolic law's single fitted center. The tree does not need a closed-form
$l^\star(N, D)$; it learns a different effective optimum in each scale region by branching. That is the
direct answer to the lrbsz failure mode.

So I expect this to beat the symbolic law on lrbsz. But I have to be honest about where the tree's bias
*costs* me, because the same staircase that helps off the hull hurts on the families the symbolic law
already nailed. A power-law decay is *smooth and monotone*; a tree approximates it with a finite
staircase, so on a family where the true surface is a clean additive power law — vocab, dataconstrained —
the tree will be slightly *worse* than the exact symbolic form, because it pays a discretization error
the symbolic form does not. I expect vocab to stay strong (the surface is smooth and the test region is
near the hull, so the staircase is fine — maybe even a touch better than the symbolic $0.929$ if the
symbolic form's cross term was slightly mis-specified) but dataconstrained to *drop* below the symbolic
$0.929$, because the dataconstrained test region is denser/larger and the staircase cannot extrapolate
the saturating power-law tail as cleanly as the explicit effective-token term — past the training hull
the tree just holds the boundary constant where the symbolic law keeps bending toward the floor. So this
rung is a *trade*: win lrbsz (where flexibility beats a rigid hand-shaped basin), hold vocab, give back
some dataconstrained (where the explicit asymptotic form beats a staircase).

Now make it concrete in the task's edit surface, because the loop only lets me fill the model class. I
reuse the same mixed feature map as the kernel-ridge rung — standardized raw numerics plus standardized
$\log(1+\cdot)$ numerics plus a one-hot of `group` — because the log features give the trees the
power-law geometry to split on and the one-hot lets the single ensemble separate families by branching on
the group indicator. I fit gradient-boosted trees: shallow trees (`max_depth=3`, weak learners),
many rounds (`n_estimators=120`) with a small learning rate (`0.05`, Friedman shrinkage so no single
tree dominates), row subsampling (`subsample=0.9`, stochastic boosting) and column subsampling
(`colsample_bytree=0.8`, decorrelating the trees), and the L2 leaf penalty (`reg_lambda=1.0`, the
$+\lambda$ in $w^\star$). The `hist` tree method uses the weighted-quantile-binned split finder. One
detail is load-bearing and is where this rung's implementation diverges from a textbook regressor: I fit
in *log-target* space **only when the target is strictly positive** — for lrbsz's lm_loss and
dataconstrained's loss, fitting $\log y$ gives the trees a multiplicative error scale that matches a
power-law surface — but the vocab target is a unigram-normalised loss that *can be negative*, so for
vocab I fall back to fitting $y$ directly in the linear domain. That conditional is the whole
signed-target handling, and it mirrors the linear-vs-log residual choice the symbolic rung made per
family. I also keep a `GradientBoostingRegressor` fallback for when the boosted-tree package is
unavailable, with matching hyperparameters, so the rung runs either way. The full scaffold module is in
the answer.

So the delta from the symbolic rung is concrete: where the hand-shaped basin failed because one fitted
center cannot follow a drifting optimum, I now let an axis-aligned tree ensemble learn a *different*
effective optimum per scale region by conditional splitting, with an off-hull bias that flattens to the
boundary instead of collapsing to zero. The falsifiable claims against the symbolic numbers: lrbsz should
improve in $R^2$ from $-3.05$ toward $-1$ (still likely negative — the held-out lrbsz region is genuinely
hard and the tree's staircase cannot fully extrapolate the surface, but the conditional splits should cut
the ranking error the fixed basin made) with `MAE` dropping further below the symbolic $0.063$; vocab
should *rise* above $0.929$ toward the high $0.97$s (smooth surface near the hull, where the ensemble's
flexibility helps and its discretization error is small); and dataconstrained should *fall* below the
symbolic $0.929$, into the mid-$0.8$s, because the staircase cannot extrapolate the saturating tail as
cleanly as the explicit effective-token law. If that is the pattern — lrbsz and vocab won, dataconstrained
given back — then the lesson for the next rung is sharp: neither the rigid hand-shaped symbolic form nor
the flexible-but-asymptotically-blind tree is dominant; the tree wins where flexibility matters (lrbsz,
vocab) and loses where the explicit asymptotic form matters (dataconstrained), and the strongest
solution is the one that carries the *correct literature-grounded asymptotic form per family* — including
a scale-dependent optimum for lrbsz that the symbolic rung's fixed basin lacked.
