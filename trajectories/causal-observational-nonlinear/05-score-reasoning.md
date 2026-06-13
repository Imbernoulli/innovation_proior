CAM's numbers are the bar, and they tell me precisely where the remaining slack is. It is the first rung
to get the nonlinear order right and then prune, and it shows: SF20-GP F1 0.881 at SHD 7, ER20-Gauss F1
0.732 at precision 0.97, ER12-LowSample F1 0.564. The decouple-order-from-edges thesis is vindicated —
the recall collapse that sank NOTEARS-MLP (0.13 on SF20-GP) is gone (CAM's recall there is 0.82), and
the precision NOTEARS-MLP bought is kept (CAM's ER20-Gauss precision is 0.97–1.0). So the architecture is
right. But read where CAM is *weakest* and it is exactly the stage I named as its residual weakness: the
**order search**. ER12-LowSample is CAM's soft spot — F1 0.564, and the limiting metric is precision
(0.44–0.51), not recall (0.67–0.78): on 150 samples the greedy boosted residual-variance ordering places
some nodes in the wrong relative order, which then lets spurious upstream edges through that the
partial-residual pruning cannot fully remove. And on ER20-Gauss the recall (0.59) is the laggard — again
an order-quality issue: a node placed slightly too early loses true parents that come "after" it. The
common thread is that CAM's order is recovered *greedily and indirectly*, by repeatedly fitting a
gradient-boosted regression of every remaining variable on the whole current prefix and appending the
lowest-residual-variance one. That is `O(d²)` boosted fits, each on a growing predictor set, and at small
`n` or with subtle mechanisms the residual-variance comparisons are noisy, so the greedy append takes
wrong turns it never backtracks. The lever, then, is to recover the order *directly and globally* from a
quantity that pins down leaves exactly — not by comparing fitted residual variances, but by reading the
order off the data distribution's own geometry. That is what score matching gives me.

Let me derive it from the additive-noise structure, because the whole method falls out of one identity.
Write the model `X_j = f_j(pa(j)) + N_j` with `N_j` Gaussian, independent. The joint log-density is
`log p(x) = sum_j log p_{N_j}(x_j - f_j(pa(j)))`, and since the noise is Gaussian,
`log p_{N_j}(z) = -z²/(2σ_j²) + const`. Now take the *score* — the gradient of the log-density,
`s(x) = ∇ log p(x)` — and look at its `j`-th component. Two kinds of terms contribute to `∂ log p / ∂ x_j`:
the term where `x_j` is the *argument of its own noise* (from `log p_{N_j}(x_j - f_j(pa(j)))`), and the
terms where `x_j` appears *inside some child's mechanism* `f_c(pa(c))` with `j ∈ pa(c)`. So in general
`s_j(x)` depends on `x_j`, on `x_j`'s parents (through `f_j`), and on `x_j`'s children (through their
mechanisms). Here is the crux. Take the *second* derivative of the log-density along `x_j` — the
`j`-th diagonal entry of the Hessian of `log p`, call it `H_{jj}(x) = ∂² log p / ∂ x_j²`. For the
Gaussian-noise term, `∂²/∂x_j² [ -(x_j - f_j)²/(2σ_j²) ] = -1/σ_j²`, a *constant*, because `f_j` does
not depend on `x_j` (no self-loops). The *only* `x`-dependence in `H_{jj}` comes from `x_j`'s
appearance inside its *children's* mechanisms. Therefore, if `j` is a **leaf** — no children — then
`H_{jj}(x) = -1/σ_j²` is *constant in `x`*: its variance over the data is zero. And if `j` is *not* a
leaf, `H_{jj}` varies with `x` (through the nonlinear child mechanisms), so its variance is strictly
positive. This is the exact, distributional characterization of a leaf I was missing: **a variable is a
leaf iff the variance of the `j`-th diagonal Hessian of `log p` is zero.** No regression, no
residual-variance comparison — a property of the score's Jacobian, read directly off the distribution.

That single fact gives a clean, *global*, backtracking-free order recovery: estimate the diagonal of the
Hessian of `log p` at the sample points, pick the variable with the **minimum empirical variance** of its
diagonal Hessian entry as the current leaf, append it (it goes *last* in the topological order), then
*remove that variable from the data* and recompute on the remaining variables — because deleting a leaf
leaves an ANM over the rest, and the next leaf of the reduced system is the second-to-last variable, and
so on. After `d-1` removals the last remaining variable is the root. Reverse the leaf sequence and I have
a topological order, roots first. This is strictly better-posed than CAM's greedy residual-variance
append: leaf identification is a *direct* read of a distributional quantity (the order is recovered
exactly in the population limit, with no greedy commitment that compounds errors), and each step removes
one variable so the problem shrinks cleanly — exactly the small-`n` robustness CAM's growing-predictor
regressions lacked.

So everything reduces to estimating, from samples, the diagonal of the Hessian of `log p` — the
quantity `∂² log p / ∂ x_j²` at each data point. I cannot differentiate a density I do not have, but I
do not need the density; I need its score and the diagonal of the score's Jacobian, and those can be
estimated *non-parametrically* via Stein's identities with a kernel. First-order Stein gives the score:
for a smooth kernel `K`, the regularized estimate is `G = (K + η_G I)^{-1} ∇K`, where `K_{ij} =
exp(-||x_i - x_j||²/(2s²))/s` is the Gaussian Gram matrix with bandwidth `s` set to the median pairwise
distance, and `∇K` is its sample gradient — `G` is then the `(n × d)` matrix of estimated scores at the
sample points. Second-order Stein gives the diagonal of the score's Jacobian: `H = -G² + (K + η_H
I)^{-1} ∇²K`, where `∇²K` is the second sample derivative of the kernel and the `-G²` term is the
correction that makes this the *diagonal Hessian of log p* rather than just the derivative of the score.
Both are ridge-regularized matrix solves (`η_G = η_H = 0.001`), `O(n³)` per evaluation — entirely
practical at the `n = 150–2000` and `d = 12–20` of these scenarios. Before taking the variance I
normalize each column of `H` by its mean (so variables on different scales are compared fairly), then
the leaf is `argmin` of the column variances. This is the part of the method that has *no analogue* in
any prior rung: DirectLiNGAM used a linear residual entropy, GraN-DAG used network-path products,
NOTEARS-MLP used a linear skeleton, CAM used fitted residual variances — none of them touch the score's
Hessian, which is the object that pins leaves down exactly.

Once the order is in hand, the edge selection is the same kind of regularized cleanup CAM already does
well, and I should not reinvent it — the order was the bottleneck, not the pruning. For each node in
order position `pos`, fit a nonlinear regression of it on its predecessors and keep a parent only if it
genuinely contributes; the canonical method uses CAM-style significance pruning, and on this harness the
natural realization is the same gradient-boosted regression with a feature-importance threshold that CAM
already uses (the libraries available are numpy/scipy/scikit-learn plus causal-learn). So I reuse the
proven pruning machinery and replace only the order-recovery stage with the score-matching leaf
detection. That is the minimal, principled delta from CAM: same decoupled architecture, same nonlinear
order-respecting candidate generation, same pruning — but the order itself comes from the exact
leaf-variance characterization instead of a greedy residual-variance search.

Let me be careful about what carries over verbatim from the canonical reference and what is re-expressed
for this harness, since the finale code must be faithful. The leaf-detection core — the Gaussian kernel
with median-distance bandwidth, `∇K = -Σ_i (x_k - x_i) K_{ik}/s²`, the score `G = (K+η_G I)^{-1}∇K`, the
second-derivative `∇²K` with the `-1/s² + (x_k-x_i)²/s⁴` form, the diagonal Hessian `H = -G² +
(K+η_H I)^{-1}∇²K`, the per-column mean normalization, and the `argmin`-variance leaf with iterative
column removal and final reversal — is transcribed directly from the reference implementation (the
reference is in PyTorch; I re-express the identical algebra in numpy, which the harness supports). The
two defaults `η_G = η_H = 0.001` and the median-distance bandwidth are the reference defaults. The only
adaptation is the pruning stage, where I use the harness's gradient-boosted regression with the same
`0.05` importance cutoff CAM uses, rather than a GAM significance test — the same pragmatic substitution
the task's other baselines make. The full scaffold module is in the answer.

Now the bar this has to clear and what I would validate, because there is no leaderboard row for it. The
claim is narrow and falsifiable: SCORE should match or beat CAM *everywhere*, and beat it *most* exactly
where CAM's order search was weakest. On **ER12-LowSample** — CAM's softest scenario (F1 0.564,
precision-limited at ~0.44–0.51) — the score-based leaf detection should place nodes correctly even at
`n = 150` because it reads a distributional quantity rather than comparing noisy fitted residual
variances on a growing predictor set, so I expect the order errors that let spurious upstream edges
through to drop, lifting precision and pushing F1 above 0.564. On **ER20-Gauss** — where CAM's recall
(0.59) lagged — a more accurate order should stop placing nodes too early and losing their true parents,
so I expect recall to rise toward CAM's precision (already ~0.97) and F1 to clear 0.732. On **SF20-GP**,
CAM is already excellent (F1 0.881, SHD 7); here the realistic claim is *parity*, since the order on a
2000-sample scale-free GP graph is already nearly perfect for CAM, and the score-matching order should
match it without a clear margin — the win is on the harder, lower-sample scenarios, not this one. The
single number that would falsify the thesis that "the order search was the remaining bottleneck": if
SCORE's averaged F1 does not clear CAM's ~0.73 and its ER12-LowSample F1 does not beat CAM's 0.564, then
the greedy residual-variance order was not the limiting factor and the score-matching detour bought
nothing. What I would validate first, then, is precisely the ER12-LowSample order accuracy — the cheapest
direct test of whether reading leaves off the score's Hessian beats appending them by residual variance.
