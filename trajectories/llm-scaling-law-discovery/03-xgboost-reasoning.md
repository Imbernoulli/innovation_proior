The symbolic rung did exactly what I predicted on the two easy families and exactly what I feared on the
hard one, and the numbers say it precisely. Vocab swung from kernel ridge's $-0.567$ to $0.929$, and
dataconstrained from $-13.4$ to $0.929$ — both rescued, because the power-law-plus-floor backbone and the
repeat-efficiency effective-token term finally gave the law something that extrapolates to the
larger/denser test points instead of collapsing off the hull. So the diagnosis that "the missing
ingredient was the asymptotic form, not flexibility" was right for those families. Before I read lrbsz I
should confirm the one tool I trust across rungs still holds: the metric normalizer. Reconstruct the
held-out standard deviation from the new table via $\sigma = \mathrm{RMSE}/\sqrt{1 - R^2}$. Vocab gives
$0.2261/\sqrt{1 - 0.9285} = 0.846$; dataconstrained $0.1476/\sqrt{1 - 0.9292} = 0.554$; lrbsz
$0.0768/\sqrt{1 + 3.0519} = 0.0382$. These are the *same* $\sigma$ values I recovered from the kernel-ridge
table — vocab $0.846$, dataconstrained $0.554$, lrbsz $0.0382$ — as they must be, since the split and the
targets are fixed; and $\mathrm{MAE}/\sigma = 0.0629/0.0382 = 1.65$ reproduces the reported lrbsz `NMAE` of
$1.647$. So the normalizer is locked, and I can keep reading every family's difficulty through its $\sigma$.

Now lrbsz. It came back at $R^2 = -3.052$ — still negative, still worse than predicting the mean. And the
secondary metrics tell the real story: the lrbsz `MAE` dropped from kernel ridge's $0.619$ all the way to
$0.0629$, and `RMSE` from $0.777$ to $0.0768$, an order-of-magnitude improvement in absolute error. The
$\sigma$ arithmetic explains exactly why that improvement did not buy a positive $R^2$: with the needle-thin
$\sigma = 0.0382$, the bar for $R^2 = 0$ is $\mathrm{RMSE} < 0.0382$, and the basin fit landed at
$\mathrm{RMSE} = 0.0768$ — a factor of two over the bar, so $R^2 = 1 - (0.0768/0.0382)^2 = 1 - 4.04 = -3.04$,
which is the $-3.05$ I see. So the hand-shaped basin is *close* in absolute terms — it is predicting loss
values that are tiny fractions off — but it still loses to the trivial mean in $R^2$, because it is getting
the *ranking* of the held-out points wrong even while the magnitudes are nearly right. That is the signature
of exactly the failure I flagged: a basin with a single fitted center $(\log l^\star, \log b^\star)$ cannot
follow an optimum that drifts with scale. Out-of-sample the bowl is centered in the wrong place, so the
quadratic penalty it adds has the wrong local slope relative to the true surface, and the predicted ordering
of held-out configurations is off even though the absolute numbers are small. So the symbolic law has
plateaued on lrbsz: I cannot get it positive by tweaking a hand-shaped form, because the problem is that one
fitted center cannot track a moving optimum, and to beat $-3.05$ at all I need to push `RMSE` below the
$0.0768$ the fixed basin achieved.

That plateau forces a real fork, and I should walk both branches before choosing. Branch one keeps the
symbolic law and makes the basin center an explicit function of scale — replace the constant $\log l^\star$
with something like $\log l^\star = \gamma_0 + \gamma_1\log N + \gamma_2\log D$, and similarly for
$b^\star$. That is almost certainly the *correct* physics, but count what it costs the fit: the lrbsz form
already carries nine coupled parameters, and this adds four more, all of them inside the log-argument of a
quadratic where a shift in $\gamma_0$ trades against the curvature $k$ and the slopes $\gamma_1, \gamma_2$
trade against each other. The identifiability was already the weakest on lrbsz; making the center a
regressed surface deepens the shallow valleys the multi-start has to escape, and — worse — I would be
*guessing* the drift's functional form, when the whole point of a hand-derived law is that I do not have the
right one for the optimum drift yet. Branch two reopens a question I thought I had closed at the first rung.
I went black-box-then-symbolic because the black box (kernel ridge) collapsed off the hull. But the
*reason* it collapsed was specific — RBF locality, kernel values decaying to zero far from training points,
no floor. It was not that flexible learners are wrong here; it was that *that* learner had no way to carry
structure into the extrapolation region. So the right question is: is there a flexible, model-free learner
whose inductive bias does *not* collapse off the hull, and that can represent a scale-dependent basin
without my hand-shaping — or hand-guessing — a single center or its drift? If there is, it might beat the
symbolic law on lrbsz, where the symbolic law's rigidity is the bottleneck, while holding the easy families.
The candidate I keep coming back to on structured tabular data is a boosted ensemble of regression trees.
Let me derive why it fits, from what trees actually do.

A regression tree partitions the input space into axis-aligned cells and predicts a constant per cell. By
itself that is a crude, high-variance learner. But boosting builds an additive ensemble $\hat y_i = \sum_k
f_k(x_i)$ where each $f_k$ is a tree fit to correct the residual of the ensemble so far, and the leverage is
in *how* each tree is grown. The modern formulation regularizes the whole thing: minimize $\sum_i l(\hat
y_i, y_i) + \sum_k \Omega(f_k)$ with $\Omega(f) = \gamma T + \tfrac12\lambda\lVert w\rVert^2$ — a per-leaf
cost $\gamma T$ and an L2 penalty on the leaf scores. Fit additively: at round $t$, second-order
Taylor-expand the loss around the current prediction so the per-round objective depends on the loss only
through the gradient $g_i$ and curvature $h_i$, and the optimal leaf weight is $w_j^\star = -G_j/(H_j +
\lambda)$ with $G_j, H_j$ the gradient and curvature sums in the leaf — a Newton step damped by $\lambda$ so
an under-populated leaf cannot blow up. The split gain $\tfrac12[G_L^2/(H_L+\lambda) + G_R^2/(H_R+\lambda) -
G^2/(H+\lambda)] - \gamma$ scores every candidate split, and $\gamma$ doubles as the prune threshold (a
split that does not clear $\gamma$ is not made). That is the engine. Why does it fit *this* problem where
kernel ridge did not?

First, the inductive bias off the hull is fundamentally different from an RBF kernel's. A tree does not
decay to zero away from training points — it predicts the constant of whichever leaf the query falls into,
which for an out-of-region query is the constant of the *nearest boundary cell*. So instead of collapsing
toward zero, a boosted-tree prediction *flattens to the last seen value* in that direction. This matters in
the exact terms of the mean-predictor floor I worked out at the start: the constant-mean scaffold sits below
zero by the squared train/test mean gap, and "collapse to zero" is even worse than that on a positive-loss
family because it does not even track the mean; "flatten to the boundary value," by contrast, holds a
prediction near the *edge* of the training loss surface, which on a surface that saturates toward a floor is
a far better guess than zero — a crude version of the floor the symbolic law imposes explicitly. So the
structural reason kernel ridge failed is precisely the reason a tree ensemble might not.

Second — and this is what could finally crack lrbsz — trees represent basins and cross-axis interactions
*natively*, with no hand-shaped center and no guessed drift. The basin in $(\log l, \log b)$ is just a
region of low loss surrounded by higher loss; an axis-aligned partition carves that region out as a set of
cells, and the ensemble assigns each cell its own constant, so it reconstructs the bowl as a staircase
without ever fitting a quadratic or its center. More to the point, because every split is conditional on the
splits above it, a tree captures interactions for free: a split on $\log l$ *inside* a branch already split
on $N$ is precisely "the best learning rate depends on the model scale" — the scale-dependent optimum drift
that broke the symbolic law's single fitted center. The tree does not need a closed-form $l^\star(N, D)$ or
my guess at its slopes; it learns a different effective optimum in each scale region by branching. That is
the direct answer to the lrbsz failure mode, and it is why I take branch two: the tree gets the drift the
symbolic branch would have to hand-guess.

It is worth tracing the flatten-versus-collapse difference on a single concrete query, because the whole bet
rides on it. Take a held-out lrbsz configuration whose $N$ is larger than any model in the training grid.
Kernel ridge sees a point whose standardized log-$N$ coordinate sits at, say, $+3$; every training kernel
value $\exp(-\gamma\lVert x - x_t\rVert^2)$ is small, the weighted sum shrinks toward zero, and the
prediction is a near-zero blend that has nothing to do with the true lm-loss of a few units. The tree routes
that same query down whichever branch handles "largest $N$ region," lands in the leaf built from the
biggest models it *did* see, and predicts that leaf's constant — a loss near the low edge of the training
surface, i.e. near the floor. Both are wrong about the exact held-out value, but the tree is wrong by the
distance from the boundary loss to the true loss (small, because the surface is nearly flat in $N$ at the
top of the grid), whereas kernel ridge is wrong by the whole magnitude of the loss. On the needle-thin
lrbsz $\sigma = 0.0382$ that difference is the difference between an $R^2$ near $-1$ and one near $-400$. So
the flatten bias is not a minor tie-breaker here; it is the mechanism that makes a flexible learner viable
on a family that destroyed the last one.

The depth of the trees is the one hyperparameter I want to pin from the problem rather than a default, and
the reasoning is about interaction order. The lrbsz surface I need is a $(l, b)$ basin whose location is
conditioned on scale — so to express even the coarsest version, a single tree has to split once to pick the
scale region ($N$ or $D$), then split on $l$, then on $b$: a three-way conjunction, which is exactly a
depth-3 path. So `max_depth=3` is the *minimum* depth that can represent the scale-conditioned basin at all.
Going shallower (stumps, depth 1) would let the ensemble add only marginal one-axis effects, additive again,
reintroducing the very blindness that sank the additive law; going deeper (depth 6+) would let each tree
carve six-way conjunctions of a few hundred points, which is memorization of the interior and a return to
the collapse-off-the-hull regime in staircase clothing. Depth 3 is the sweet spot the surface itself picks:
deep enough for the three-way scale$\times l\times b$ interaction, shallow enough that each learner stays a
weak, generalizing correction.

I have to be honest about where the tree's bias *costs* me, because the same staircase that helps off the
hull hurts on the families the symbolic law already nailed. A power-law decay is *smooth and monotone*; a
tree approximates it with a finite staircase, so on a family where the true surface is a clean additive
power law — vocab, dataconstrained — the tree pays a discretization error the exact symbolic form does not.
And its off-hull flattening, which was a virtue on the saturating basin, becomes a liability on a family
whose test points sit past a still-descending tail: where the symbolic effective-token law keeps bending
toward the floor, the tree just holds the boundary constant. So I expect vocab to stay strong or improve
(the surface is smooth and the test region near the hull, so the staircase is fine and the ensemble's
extra flexibility can shave in-region variance) but dataconstrained to *drop* below the symbolic $0.929$
(its denser test region is exactly where flatten-to-boundary undershoots a descending tail). This rung is a
*trade*: win lrbsz, hold or raise vocab, give back some dataconstrained.

Now make it concrete in the task's edit surface, because the loop only lets me fill the model class. I reuse
the same mixed feature map as the kernel-ridge rung — standardized raw numerics plus standardized
$\log(1+\cdot)$ numerics plus a one-hot of `group` — because the log features give the trees the power-law
geometry to split on and the one-hot lets the single ensemble separate families by branching on the group
indicator. The log point is not cosmetic for a tree: the greedy split-finder only ever thresholds a single
feature at a value, and on a *raw* $N$ that spans four orders of magnitude, almost all of the models pile
into the bottom bin and the only useful thresholds sit out among the few largest — the finder has poor
candidate cuts across the small-model range. On $\log N$ the same models are spread evenly across the axis,
so there is a good split threshold available at every scale, and "the optimum shifts by such-and-such
between $10^8$ and $10^9$ parameters" becomes a clean cut the tree can actually find. A split at a
log-threshold is a split at a scale ratio, which is the currency power-law structure trades in. I fit gradient-boosted trees: shallow trees (`max_depth=3`, so each learner sees at most a
depth-three conjunction of splits — enough to express a three-way $N\times l\times b$ interaction but too
shallow to memorize), many rounds (`n_estimators=120`) with a small learning rate (`0.05`, Friedman
shrinkage so no single tree dominates), row subsampling (`subsample=0.9`, stochastic boosting) and column
subsampling (`colsample_bytree=0.8`, decorrelating the trees), and the L2 leaf penalty (`reg_lambda=1.0`,
the $+\lambda$ in $w^\star$). It is worth checking the capacity against the data: a depth-3 tree has at most
eight leaves, so $120$ of them span up to $960$ leaf regions, which against a few hundred runs per group
would be alarming — except that the learning rate scales every leaf weight by $0.05$, so the effective
additive budget is roughly $120 \times 0.05 = 6$ units of correction, and the row/column subsampling plus
the leaf penalty further damp it. That is why this configuration is a *regularized* flexible learner rather
than a memorizer, which is the whole point after watching kernel ridge near-interpolate the interior and
collapse outside it. The row and column subsampling draw from a seeded generator, so with the fit pinned to
seed 42 the ensemble is reproducible run to run — which the evaluation requires, and which is easy to lose
if the stochastic boosting is left to a fresh random state each call.

I should also be clear-eyed that even in the best case this rung does not *solve* lrbsz, and knowing why
keeps the next step honest. The tree's prediction is piecewise-constant, so on a loss surface that keeps
descending past the last training boundary — which is where the held-out lrbsz optimum sits — the ideal
tree can only hold the boundary cell's constant; it cannot continue the descent, because it has no cell out
there to descend into. Conditional splitting buys me a *different constant per scale region up to the
boundary*, which is a real gain over one fixed basin, but past the boundary the staircase is flat by
construction. So the tree narrows the lrbsz miss without erasing it, and the residual it leaves is
specifically the part of the surface that lives beyond the training hull — the part only an explicit
asymptotic form, with the optimum written as a function of scale, can extrapolate rather than flatten.

One detail is load-bearing and is where this rung's implementation diverges from a textbook regressor: I fit
in *log-target* space **only when the target is strictly positive**. For lrbsz's lm_loss and
dataconstrained's loss, fitting $\log y$ gives the trees a multiplicative error scale — which, for the same
first-order reason as the symbolic rung's residual choice, is the right currency on a tight positive cluster
and matches the power-law surface — but the vocab target is a unigram-normalised loss that *can be negative*,
so for vocab I fall back to fitting $y$ directly in the linear domain, which is also exactly the domain the
metric scores. That conditional is the whole signed-target handling, and it mirrors the linear-vs-log
residual choice the symbolic rung made per family. I also keep a `GradientBoostingRegressor` fallback for
when the boosted-tree package is unavailable, with matching hyperparameters, so the rung runs either way. The
two are not bit-identical — the boosted-tree package takes the damped Newton leaf step $w^\star = -G/(H +
\lambda)$ using the loss curvature, while the fallback takes a gradient-only stagewise step — so the numbers
can differ slightly, but the inductive bias I am relying on is the same in both: shallow axis-aligned trees,
conditional splits that learn scale-dependent optima, and a boundary-flattening rather than
collapse-to-zero extrapolation. The point of the fallback is that the diagnosis this rung is meant to
deliver does not hinge on one package being installed. The full scaffold module is in the answer.

The dataconstrained give-back deserves a concrete mechanism rather than a shrug, because it is the price I
am knowingly paying. On that family the test points are denser — more repeated tokens, larger $D/U$ — than
the training runs, so they sit past the hull along the repetition axis. The symbolic effective-token law
handles them by construction: as $D/U$ grows, $D_{\text{eff}}$ keeps rising toward its ceiling and
$B\,D_{\text{eff}}^{-\beta}$ keeps bending the predicted loss downward toward the floor, so the law
*extrapolates the saturation*. The tree, routed to its highest-repetition leaf, predicts that leaf's
constant — the loss at the densest training point — and holds it flat for every denser test point, so its
error on those points is roughly the gap between the boundary loss and the true, slightly-lower saturated
loss. That gap is small per point but systematic and one-signed (always too high), which is exactly the kind
of bias that inflates `RMSE` without being huge in `MAE`; against dataconstrained's $\sigma = 0.554$ a
systematic `RMSE` of $\sim 0.21$ pulls $R^2$ down to the mid-$0.8$s. So the give-back is not the tree being
"worse" everywhere — it is the tree being unable to extrapolate one specific asymptotic that the symbolic
form wrote down explicitly, localized to the densest test points.

So the delta from the symbolic rung is concrete: where the hand-shaped basin failed because one fitted center
cannot follow a drifting optimum, I now let an axis-aligned tree ensemble learn a *different* effective
optimum per scale region by conditional splitting, with an off-hull bias that flattens to the boundary
instead of collapsing to zero. Let me turn the locked $\sigma$ values into falsifiable numbers. On lrbsz, to
beat the symbolic $-3.05$ the tree must push `RMSE` below $0.0768$; if the conditional splits cut the
ranking error enough to bring `MAE` down to roughly $0.053$ — a further $\sim 15\%$ over the basin's
$0.063$, about what I would expect from replacing a wrong-centered bowl with region-wise constants — then
`RMSE` lands near $0.059$ and $R^2 = 1 - (0.059/0.0382)^2 = -1.4$: a large improvement from $-3.05$ but
*still negative*, because the staircase narrows the miss without fully reconstructing a surface whose test
optimum has drifted past the training boundary it can only flatten to. On vocab, if the ensemble shaves
`RMSE` to $\sim 0.115$ then $R^2 = 1 - (0.115/0.846)^2 = 0.98$, a touch above the symbolic $0.929$. On
dataconstrained, if the staircase gives back some tail and `RMSE` rises to $\sim 0.21$ then $R^2 = 1 -
(0.21/0.554)^2 = 0.86$, below the symbolic $0.929$ — the cost of discretization on a descending tail. If
that is the pattern — lrbsz and vocab won, dataconstrained given back — then the lesson for the next rung is
sharp: neither the rigid hand-shaped symbolic form nor the flexible-but-asymptotically-blind tree is
dominant; the tree wins where flexibility matters (lrbsz, vocab) and loses where the explicit asymptotic form
matters (dataconstrained), and the strongest solution is the one that carries the *correct
literature-grounded asymptotic form per family* — including a scale-dependent optimum for lrbsz that both the
symbolic rung's fixed basin and the tree's boundary-flattening fail to carry.
