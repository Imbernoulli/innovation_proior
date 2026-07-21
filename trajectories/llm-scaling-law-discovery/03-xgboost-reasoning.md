The symbolic rung did what I predicted on the two easy families and what I feared on the hard one. Vocab
swung from kernel ridge's $-0.567$ to $0.929$, and dataconstrained from $-13.4$ to $0.929$ — both rescued,
because the power-law-plus-floor backbone and the repeat-efficiency effective-token term finally gave the
law something that extrapolates to the larger/denser test points instead of collapsing off the hull. So the
diagnosis that "the missing ingredient was the asymptotic form, not flexibility" was right for those
families. The normalizer is unchanged with the split fixed: $\sigma = \mathrm{RMSE}/\sqrt{1 - R^2}$ still
gives vocab $0.846$, dataconstrained $0.554$, lrbsz $0.0382$.

Lrbsz came back at $R^2 = -3.052$ — still worse than the mean. But the secondary metrics tell the real
story: `MAE` dropped from kernel ridge's $0.619$ to $0.0629$ and `RMSE` from $0.777$ to $0.0768$, an
order-of-magnitude improvement in absolute error. The $\sigma$ arithmetic explains why that did not buy a
positive $R^2$: with $\sigma = 0.0382$ the bar for $R^2 = 0$ is `RMSE` $< 0.0382$, and the basin landed at
$0.0768$ — a factor of two over, so $R^2 = 1 - (0.0768/0.0382)^2 = -3.04$, the number I see. So the basin is
*close* in absolute terms — predicting loss values tiny fractions off — but it is getting the *ranking* of
the held-out points wrong even while the magnitudes are nearly right. That is the failure I flagged: a basin
with a single fitted center cannot follow an optimum that drifts with scale, so out-of-sample the bowl is
centered wrong and its quadratic penalty has the wrong local slope. The symbolic law has plateaued on
lrbsz; I cannot get it positive by tweaking a hand-shaped form, and to beat $-3.05$ at all I need `RMSE`
below the $0.0768$ the fixed basin achieved.

That plateau forces a fork, and I should walk both branches. Branch one keeps the symbolic law and makes the
basin center an explicit function of scale — replace the constant $\log l^\star$ with $\gamma_0 +
\gamma_1\log N + \gamma_2\log D$, similarly for $b^\star$. That is almost certainly the correct physics, but
the lrbsz form already carries nine coupled parameters and this adds four more, all inside the log-argument
of a quadratic where a $\gamma_0$ shift trades against the curvature $k$ and the slopes trade against each
other; identifiability was already weakest on lrbsz, and worse, I would be *guessing* the drift's functional
form. Branch two reopens a question I thought I had closed at the first rung. The black box collapsed off
the hull for a *specific* reason — RBF locality, kernel values decaying to zero, no floor — not because
flexible learners are wrong here. So the right question is: is there a flexible, model-free learner whose
inductive bias does *not* collapse off the hull, and that can represent a scale-dependent basin without my
hand-shaping or hand-guessing a center or its drift? If so it might beat the symbolic law on lrbsz while
holding the easy families. The candidate on structured tabular data is a boosted ensemble of regression
trees.

A regression tree partitions the input space into axis-aligned cells and predicts a constant per cell — by
itself crude and high-variance. Boosting builds an additive ensemble $\hat y_i = \sum_k f_k(x_i)$ where each
$f_k$ corrects the residual so far, and the modern formulation regularizes: minimize $\sum_i l(\hat y_i,
y_i) + \sum_k \Omega(f_k)$ with $\Omega(f) = \gamma T + \tfrac12\lambda\lVert w\rVert^2$. Fit additively —
second-order Taylor-expand the loss around the current prediction so the round depends on it only through
the gradient $g_i$ and curvature $h_i$ — and the optimal leaf weight is $w_j^\star = -G_j/(H_j + \lambda)$,
a Newton step damped by $\lambda$ so an under-populated leaf cannot blow up; the split gain
$\tfrac12[G_L^2/(H_L+\lambda) + G_R^2/(H_R+\lambda) - G^2/(H+\lambda)] - \gamma$ scores each split and
$\gamma$ doubles as the prune threshold. Why does that fit *this* problem where kernel ridge did not?

First, the off-hull bias is fundamentally different from an RBF kernel's. A tree does not decay to zero away
from training points — it predicts the constant of whichever leaf the query falls into, which for an
out-of-region query is the constant of the *nearest boundary cell*. So instead of collapsing toward zero, a
boosted-tree prediction *flattens to the last seen value* in that direction. On a positive-loss family that
saturates toward a floor, holding the boundary loss is a far better guess than zero — a crude version of the
floor the symbolic law imposes explicitly. So the structural reason kernel ridge failed is precisely the
reason a tree ensemble might not.

Second — and this is what could crack lrbsz — trees represent basins and cross-axis interactions natively,
with no hand-shaped center and no guessed drift. The basin in $(\log l, \log b)$ is just a region of low
loss surrounded by higher loss; an axis-aligned partition carves it into cells and assigns each its own
constant, reconstructing the bowl as a staircase without fitting a quadratic. And because every split is
conditional on the splits above it, a tree captures interactions for free: a split on $\log l$ *inside* a
branch already split on $N$ is precisely "the best learning rate depends on the model scale" — the
scale-dependent optimum drift that broke the symbolic law's single fitted center. The tree learns a
different effective optimum in each scale region by branching, no closed-form $l^\star(N, D)$ required. That
is the direct answer to the lrbsz failure mode, and it is why I take branch two.

The tree depth I want to pin from the problem, and the reasoning is about interaction order. The lrbsz
surface is a $(l, b)$ basin whose location is conditioned on scale, so even the coarsest version needs a
single tree to split once to pick the scale region ($N$ or $D$), then on $l$, then on $b$: a three-way
conjunction, exactly a depth-3 path. So `max_depth=3` is the *minimum* depth that can represent the
scale-conditioned basin at all. Shallower (stumps) would let the ensemble add only marginal one-axis
effects — additive again, the very blindness that sank the additive law; deeper (6+) would let each tree
carve six-way conjunctions of a few hundred points, memorizing the interior and returning to the
collapse-off-the-hull regime in staircase clothing. Depth 3 is the sweet spot the surface picks.

I have to be honest about where the tree's bias *costs* me. A power-law decay is smooth and monotone; a tree
approximates it with a finite staircase, so on a clean additive power law — vocab, dataconstrained — the
tree pays a discretization error the exact symbolic form does not. And its off-hull flattening, a virtue on
the saturating basin, becomes a liability on a family whose test points sit past a still-descending tail:
where the symbolic effective-token law keeps bending toward the floor, the tree holds the boundary constant.
So I expect vocab to stay strong or improve (smooth surface, test region near the hull, extra flexibility
shaves in-region variance) but dataconstrained to *drop* below the symbolic $0.929$ (its denser test region
is exactly where flatten-to-boundary undershoots a descending tail). This rung is a *trade*: win lrbsz,
hold or raise vocab, give back some dataconstrained.

Now the edit surface. I reuse the same mixed feature map as the kernel-ridge rung — standardized raw
numerics plus standardized $\log(1+\cdot)$ numerics plus a one-hot of `group` — because the log features
give the trees the power-law geometry to split on and the one-hot lets the single ensemble separate families
by branching on the group indicator. The log point is not cosmetic for a tree: the greedy split-finder only
thresholds a single feature at a value, and on a *raw* $N$ spanning four orders of magnitude almost all
models pile into the bottom bin with useful thresholds only among the few largest; on $\log N$ the models
spread evenly, so a good split threshold is available at every scale and "the optimum shifts between $10^8$
and $10^9$ parameters" becomes a clean cut. A split at a log-threshold is a split at a scale ratio, the
currency power-law structure trades in. I fit gradient-boosted trees: `max_depth=3`, many rounds
(`n_estimators=120`) with a small learning rate (`0.05`, Friedman shrinkage so no single tree dominates),
row subsampling (`subsample=0.9`) and column subsampling (`colsample_bytree=0.8`, decorrelating the trees),
and the L2 leaf penalty (`reg_lambda=1.0`). The capacity checks out against the data: a depth-3 tree has at
most eight leaves, so 120 of them span up to 960 leaf regions — alarming against a few hundred runs per
group, except that the learning rate scales every leaf weight by $0.05$, so the effective additive budget is
roughly $120 \times 0.05 = 6$ units of correction, further damped by the subsampling and leaf penalty. That
is a *regularized* flexible learner, not a memorizer. The row and column subsampling draw from a seeded
generator, so with the fit pinned to seed 42 the ensemble is reproducible, which the evaluation requires.

One detail is load-bearing: I fit in *log-target* space only when the target is strictly positive. For
lrbsz's lm_loss and dataconstrained's loss, fitting $\log y$ gives the trees a multiplicative error scale —
the right currency on a tight positive cluster, for the same first-order reason as the symbolic rung's
residual choice — but the vocab target can be negative, so for vocab I fit $y$ directly in the linear
domain, which is also the domain the metric scores. I keep a `GradientBoostingRegressor` fallback for when
the boosted-tree package is unavailable, with matching hyperparameters, so the rung runs either way. The two
are not bit-identical — the boosted-tree package takes the damped Newton leaf step using the loss curvature,
the fallback a gradient-only stagewise step — but the inductive bias I rely on is the same: shallow
axis-aligned trees, conditional splits that learn scale-dependent optima, and boundary-flattening rather
than collapse-to-zero. The full scaffold module is in the answer.

Even in the best case this rung does not *solve* lrbsz. The prediction is piecewise-constant, so where the
held-out optimum sits past the last training boundary the ideal tree can only hold the boundary cell's
constant — conditional splitting buys a different constant per scale region *up to* the boundary, a real
gain over one fixed basin, but past it the staircase is flat by construction. The dataconstrained give-back
has the mirror-image mechanism: on that family the test points are denser (larger $D/U$) than the training
runs, so they sit past the hull along the repetition axis, where the symbolic effective-token law keeps
bending $B\,D_{\text{eff}}^{-\beta}$ toward the floor while the tree, routed to its highest-repetition leaf,
holds that leaf's constant flat. The gap is small per point but systematic and one-signed (always too high)
— the kind of bias that inflates `RMSE` without being huge in `MAE`, and against $\sigma = 0.554$ it is
enough to pull $R^2$ off the symbolic form's strong number.

So the a-priori read follows the bias. On lrbsz the conditional splits should cut the ranking error the
fixed basin got wrong, so absolute error improves over the symbolic `MAE` $0.063$ and $R^2$ climbs from
$-3.05$ — but the staircase can only flatten at the boundary where the test optimum has drifted past it, so
it narrows the miss without erasing it and I expect lrbsz *still negative*. Vocab should hold or slightly
beat the symbolic $0.929$; dataconstrained should drop below it. If that is the pattern — lrbsz and vocab
won, dataconstrained given back — the lesson for the next rung is sharp: neither the rigid hand-shaped
symbolic form nor the flexible-but-asymptotically-blind tree dominates, and the strongest solution carries
the *correct literature-grounded asymptotic form per family* — including a scale-dependent optimum for
lrbsz that both the fixed basin and the tree's boundary-flattening fail to carry.
