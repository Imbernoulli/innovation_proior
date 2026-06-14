Let me start from the thing that actually keeps biting me. On almost every structured-data problem I touch — claims, click logs, ranking, physics events — the model that wins is a boosted ensemble of regression trees. I am not short on accuracy. I am short on a *system*: the data has grown to hundreds of millions, sometimes billions, of rows, it is mostly sparse, and the boosting packages I have assume everything sits in RAM, re-sort the data at every node of every tree, and special-case sparsity with whatever hack the user wired up. So the goal isn't a cleverer loss. It's to take tree boosting and make it (a) regularized enough that I trust it not to memorize, and (b) scalable enough to push a terabyte of sparse data through a desktop or a handful of machines. The accuracy and the scale have to come from the same design, not be bolted together.

So let me re-derive tree boosting carefully, because I suspect the standard recipe is leaving something on the table that, if I pick it up, will *also* be the thing that makes it scale. What is the model? An ensemble of `K` trees, prediction `ŷ_i = Σ_{k=1}^K f_k(x_i)`, where each `f_k` is a regression tree. I'll write a tree as `f(x) = w_{q(x)}`: there's a structure `q` that maps an example to a leaf index in `{1,…,T}`, and a vector `w ∈ ℝ^T` of leaf scores — continuous scores, not class labels, since this is regression at the leaf even when the task is classification. `T` is the number of leaves. That's CART as a function. The ensemble is a sum of these.

Now, how do existing methods fit such a sum? Friedman's gradient boosting is the framework everyone uses and it's genuinely elegant: think of the value `F(x)` at each point as a parameter and do steepest descent *in function space*. The negative gradient of the loss with respect to the current prediction, `-g_m(x_i) = -[∂L(y_i, F(x_i))/∂F(x_i)]` evaluated at `F_{m-1}`, is the ideal direction to move at the data points — the "pseudo-residuals". That direction only lives at the training points, so you generalize it: fit a tree by least squares to those pseudo-residuals, which gives you a tree whose output is "most parallel" to `-g_m` over the data. Then, because least-squares to the gradient doesn't tell you how *far* to step, do a line search `ρ_m = argmin_ρ Σ_i L(y_i, F_{m-1}(x_i) + ρ h(x_i))`, and with trees you sharpen this into a separate optimal constant per leaf, `γ_jm = argmin_γ Σ_{x_i ∈ R_jm} L(y_i, F_{m-1}(x_i) + γ)`. For squared error that leaf constant is the residual mean; for absolute error it's the median. Shrink by `ν` and append. That's TreeBoost, that's `scikit-learn` and R's `gbm`.

But stare at the two-step structure for a second, because it's a little incoherent and the incoherence is the opening. I grow the tree's *shape* by fitting least squares to the gradient — that's the impurity criterion deciding the splits. Then I set the leaf *values* by a completely different optimization, the per-leaf line search on the real loss. The quantity I use to *choose a split* is not the quantity I'm actually trying to minimize. For squared error they coincide, which is why nobody worries about it, but in general the split-selection criterion and the objective have drifted apart. And there's a second thing missing: there's no model-complexity term anywhere in the criterion that picks splits. The only regularization is shrinkage and stopping early, both *outside* the tree-growing decision. So when I ask "is this split worth it?", the answer is computed with no notion of the cost of adding a leaf. I'd like both fixed at once: one objective that says how good a tree structure is, *including* its complexity, and that same objective should determine the leaf values.

There's a clue in an older reading of boosting. Friedman, Hastie and Tibshirani showed boosting is adaptive *Newton* stepping on an additive model. Expand the loss to second order in the increment `f`; the per-point Newton update is negative first derivative divided by second derivative. For the logistic loss that's LogitBoost: each round fits a base learner by *weighted* least squares to a working response `z_i = (y_i^* - p_i)/(p_i(1-p_i))` with weights `w_i = p_i(1-p_i)`. Look at what those are. The working response `z_i = (y^*-p)/(p(1-p))` is exactly negative-gradient-over-curvature, and the weight `p(1-p)` is exactly the second derivative of the logistic loss. So the Newton step is "a weighted least-squares fit where the weights are the curvature." Boosting already carries a per-instance second-order weight; mainstream gradient boosting just throws away the curvature, uses the bare gradient to grow the tree, and patches it back with a per-leaf line search. Friedman's own LK-TreeBoost even computes the per-leaf value as `Σ ỹ_i / Σ |ỹ_i|(2-|ỹ_i|)` — a single Newton-Raphson step, numerator the negative-gradient sum, denominator the curvature sum. The curvature is right there. So instead of treating the second order as a per-leaf afterthought, what if I build the *entire* split criterion and leaf value out of the gradient *and* the curvature from the start, and fold the complexity penalty into the same objective?

Let me try to write that single objective down. I want to minimize, over the whole ensemble,
`L(φ) = Σ_i l(ŷ_i, y_i) + Σ_k Ω(f_k)`,
where `l` is any differentiable convex loss and `Ω` penalizes the complexity of each tree. What should `Ω` be? Two things make a tree complex: how many leaves it has, and how large its leaf scores are. So `Ω(f) = γT + ½ λ‖w‖²`. The `γT` term charges `γ` per leaf — a direct knob on tree size, and, I'll see in a moment, a pruning threshold. The `½ λ‖w‖²` is an L2 penalty on the leaf scores that smooths them toward zero, the same impulse as ridge regression: don't let any one leaf make a huge, overconfident jump. I put a `½` in front so the algebra stays clean when I differentiate. If `λ = 0` and `γ = 0` this collapses back to ordinary gradient tree boosting, which is the sanity check I want — I'm generalizing, not replacing. (Regularized Greedy Forest also puts an explicit penalty on the forest, but it then re-optimizes all leaf weights jointly, fully-correctively, which is exactly the thing that's painful to parallelize; I want a penalty I can evaluate *locally* while growing one tree.)

The ensemble has functions as parameters, so I can't just hand this to a Euclidean optimizer. Fit it additively, like everyone: at round `t`, hold `F_{t-1}` fixed and add the `f_t` that minimizes
`L^(t) = Σ_{i=1}^n l(y_i, ŷ_i^{(t-1)} + f_t(x_i)) + Ω(f_t)`.
Now I take the second-order step I argued for. Taylor-expand `l(y_i, ŷ_i^{(t-1)} + f_t(x_i))` in its second argument around `ŷ_i^{(t-1)}`:
`l(y_i, ŷ^{(t-1)} + f_t) ≈ l(y_i, ŷ^{(t-1)}) + g_i f_t(x_i) + ½ h_i f_t^2(x_i)`,
with `g_i = ∂_{ŷ^{(t-1)}} l(y_i, ŷ^{(t-1)})` the first derivative and `h_i = ∂²_{ŷ^{(t-1)}} l(y_i, ŷ^{(t-1)})` the second. The term `l(y_i, ŷ^{(t-1)})` is a constant at this round — it doesn't depend on `f_t` — so I drop it, leaving the objective I actually optimize:
`L̃^(t) = Σ_{i=1}^n [g_i f_t(x_i) + ½ h_i f_t^2(x_i)] + Ω(f_t)`.
Now the loss `l` has disappeared from the tree builder except through the pair `(g_i, h_i)`. Plug in any differentiable convex loss, compute its `g` and `h`, and the rest of the machinery is identical. Squared error gives `g_i = ŷ - y`, `h_i = 1`. Logistic gives `g_i = p - y`, `h_i = p(1-p)`. A ranking objective gives its own pair. So a single split-finding engine serves regression, classification, ranking, and arbitrary user objectives — the loss is decoupled from the learner. That's not just convenient; it's what lets one *system* be the consensus tool instead of a different package per task.

Now solve it. A tree partitions the instances by leaf, so let `I_j = {i : q(x_i) = j}` be the instance set of leaf `j`, and note `f_t(x_i) = w_j` for all `i ∈ I_j`. Substitute and expand `Ω`:
`L̃^(t) = Σ_{i=1}^n [g_i w_{q(x_i)} + ½ h_i w_{q(x_i)}^2] + γT + ½ λ Σ_{j=1}^T w_j^2`
`     = Σ_{j=1}^T [ (Σ_{i∈I_j} g_i) w_j + ½ (Σ_{i∈I_j} h_i + λ) w_j^2 ] + γT`.
Let me name the leaf-aggregated statistics `G_j = Σ_{i∈I_j} g_i` and `H_j = Σ_{i∈I_j} h_i`. Then per leaf the objective is just a one-dimensional convex quadratic in `w_j`:
`G_j w_j + ½ (H_j + λ) w_j^2`.
This is `a w + ½ b w²` with `a = G_j`, `b = H_j + λ > 0` (assuming `h_i ≥ 0`, which holds for a convex loss, plus a strictly positive `λ`). Its minimizer is `w* = -a/b`, so the optimal leaf weight is
`w_j* = - G_j / (H_j + λ)`.
That's exactly the Newton step from before — negative gradient sum over regularized curvature sum — with one new ingredient: the `+λ` in the denominator. And now I see precisely what the regularizer *buys*. Without it, a leaf with tiny curvature `H_j` (few instances, or a region where the loss is nearly flat) gets a leaf weight `-G_j/H_j` that can explode. The `+λ` damps that: it caps how large a leaf score can get when the local curvature is small, and it shrinks every leaf score toward zero. The L2 penalty isn't decorative; it's the Tikhonov term that keeps the Newton step from blowing up on under-populated or flat leaves. Good.

Plug `w_j*` back to get the value of the objective at the optimum. For the quadratic `a w + ½ b w²` at `w* = -a/b`, the value is `a(-a/b) + ½ b(a²/b²) = -a²/b + ½ a²/b = -½ a²/b`. Summing over leaves,
`L̃^(t)(q) = -½ Σ_{j=1}^T G_j² / (H_j + λ) + γT`.
This is a scalar that scores a tree *structure* `q`: it tells me how good the best possible leaf weights make this particular partition, with the complexity charged in via `γT`. It plays the role impurity plays in ordinary CART — except I derived it from an arbitrary loss and it includes the model-complexity penalty. Lower is better. This is the single objective I was missing: it determines the leaf values (`w*`) *and* it scores the split, from the same expression.

Of course I can't enumerate all tree structures `q`. So grow greedily, as always: start from a single leaf and add splits one at a time. But now I score a candidate split with the *derived* score, not an external impurity. Take a node holding instances `I`, with `G = Σ_{i∈I} g_i`, `H = Σ_{i∈I} h_i`. Its contribution to the score (the `-½ G²/(H+λ)` part) is fixed before the split. After splitting into a left child with `I_L` and a right child with `I_R = I \ I_L`, the contribution becomes `-½ G_L²/(H_L+λ) - ½ G_R²/(H_R+λ)`, and I've added one leaf, costing `+γ`. The *reduction* in loss — the gain, larger is better — is (before) − (after):
`Gain = ½ [ G_L²/(H_L+λ) + G_R²/(H_R+λ) − (G_L+G_R)²/(H_L+H_R+λ) ] − γ`,
where I used `G = G_L + G_R`, `H = H_L + H_R`. Read off the structure: the fraction terms are the curvature-weighted "purity improvement" of the split, and the `−γ` is the price of the extra leaf. If the best split's gain is not positive, the loss reduction has not paid for the extra leaf, so I am better off *not* making the split. That's pre-pruning, and it falls straight out of the same `γ` that sits in `Ω`; I didn't have to invent a separate stopping rule. One constant, `γ`, simultaneously penalizes leaves in the objective and acts as the minimum-gain split threshold.

So the inner loop of split finding: for a node, for each feature, I need to find the split point that maximizes that bracket. The exact way is the obvious one. Sort the node's instances by the feature value. Sweep left to right; maintain `G_L` and `H_L` as running sums, get `G_R = G − G_L`, `H_R = H − H_L` for free, and at each candidate threshold evaluate `G_L²/(H_L+λ) + G_R²/(H_R+λ) − G²/(H+λ)` and track the max. One linear scan per feature once the data is sorted, over all features, picks the best split. That's the exact greedy algorithm, and it's what `scikit-learn` and `gbm` and the single-machine version do too. It's powerful because it considers every possible split point.

But "sort the node's instances per feature" is exactly the operation I said is killing me at scale. It can't be done if the data doesn't fit in memory, and in a distributed setting you can't sort the whole thing per node either. So I need an *approximate* split finder that doesn't require a global sort at every node. The natural relaxation: instead of considering every split point, propose a small set of candidate split points per feature — say the percentiles of that feature's distribution — bucket the instances between consecutive candidates, aggregate `(G, H)` per bucket, and search for the best split only among the bucket boundaries. With roughly `1/ε` candidates I get a controllable approximation. The proposal can be *global* (propose once per tree, reuse at every level — fewer proposal steps, but needs more candidates because they aren't refined as the tree deepens) or *local* (re-propose after each split — more work, but candidates get refined, better for deep trees). Fine. But now I have to be careful about *which* percentiles.

Here's the subtlety, and it's a beautiful one that comes straight back out of the second-order objective. If I propose candidate splits by ordinary quantiles — split points that put an equal *count* of instances in each bucket — am I doing the right thing? Let me look at what the objective actually weights. Take the per-round objective `L̃^(t) = Σ_i [g_i f_t(x_i) + ½ h_i f_t^2(x_i)] + Ω(f_t)` and complete the square in `f_t(x_i)`:
`g_i f_t + ½ h_i f_t^2 = ½ h_i (f_t^2 + 2 (g_i/h_i) f_t) = ½ h_i (f_t + g_i/h_i)^2 − ½ g_i²/h_i`.
The last term is constant in `f_t`, so up to a constant,
`L̃^(t) = Σ_i ½ h_i ( f_t(x_i) − (−g_i/h_i) )² + Ω(f_t) + const`.
That is *weighted squared error*: each instance has a target label `−g_i/h_i` and a weight `h_i`. So the importance of instance `i` to this round is its curvature `h_i` — exactly the LogitBoost weight, reappearing, but now derived for an arbitrary loss. Which means the candidate split points should be spaced evenly in the **`h`-weighted** rank of the feature, not the plain count rank. If some region of feature space has high-curvature instances, that region matters more and deserves finer candidate spacing there. So I define a rank function that weights each point by `h_i`:
`r_k(z) = ( Σ_{(x,h)∈D_k, x < z} h ) / ( Σ_{(x,h)∈D_k} h )`,
the fraction of *weight* (not count) below `z` for feature `k`, and I want candidates `s_{k1}, …, s_{kl}` with `|r_k(s_{k,j}) − r_k(s_{k,j+1})| < ε`, anchored at the feature min and max. That gives ≈ `1/ε` weighted-quantile candidates. When all `h_i` are equal — squared error, `h_i = 1` — this is just ordinary quantiles, a clean special case.

And now I hit a real wall. Computing exact weighted quantiles over a billion rows, distributed, is infeasible by sorting. For *unweighted* data, the database community solved this: an `ε`-approximate quantile summary that supports a **merge** (combine two summaries; error becomes `max(ε_1, ε_2)`) and a **prune** (shrink to `b+1` elements; error grows from `ε` to `ε + 1/b`). Those two operations are what make it streamable and distributable — you summarize chunks, merge them up a tree, prune to stay small. But every such guarantee I can find is for *equal* weights. With instance weights `h_i`, there's no off-the-shelf sketch with a provable bound; the existing approximate boosters either sort a random subset (which can fail) or use guarantee-free heuristics. I need to build a weighted quantile sketch with merge and prune that keep the *same* `ε`-guarantee. If I can't, the approximate split finder has no rigorous footing.

Let me construct it. The data is a multiset `D = {(x_1, w_1), …, (x_n, w_n)}`, weights `w_i ≥ 0`, a total order on the `x`. Two rank functions: `r⁻_D(y) = Σ_{x<y} w` (total weight strictly below `y`) and `r⁺_D(y) = Σ_{x≤y} w` (total weight at or below `y`); their difference `ω_D(y) = r⁺_D(y) − r⁻_D(y) = Σ_{x=y} w` is the weight sitting exactly at `y` (a multiset can have repeats). Total weight `ω(D) = Σ w`. A *quantile summary* is a tuple `Q(D) = (S, r̃⁺, r̃⁻, ω̃)` where `S = {x_1 < … < x_k}` is a selected subset of the points (with the global min and max always in `S`), and `r̃⁺, r̃⁻, ω̃` are stored estimates at those selected points satisfying the one-sided bounds
`r̃⁻(x_i) ≤ r⁻_D(x_i)`, `r̃⁺(x_i) ≥ r⁺_D(x_i)`, `ω̃(x_i) ≤ ω_D(x_i)`,
with equality at the min and max, plus interleaving constraints `r̃⁻(x_i) + ω̃(x_i) ≤ r̃⁻(x_{i+1})` and `r̃⁺(x_i) ≤ r̃⁺(x_{i+1}) − ω̃(x_{i+1})`. The summary stores `4k` numbers. I call `Q` an `ε`-approximate summary if for *every* `y`,
`r̃⁺(y) − r̃⁻(y) − ω̃(y) ≤ ε ω(D)`,
because then I can pin `r⁻(y) ∈ [r̃⁻(y), r̃⁺(y) − ω̃(y)]` and `r⁺(y) ∈ [r̃⁻(y) + ω̃(y), r̃⁺(y)]`, i.e. estimate either rank to within `ε ω(D)`. Exactly the weighted analogue of the unweighted guarantee.

First I need the functions defined for *all* `y`, not just the stored points, so I can reason about queries between stored points. Extend: for `y < x_1`, everything is 0; for `y > x_k`, `r̃⁻(y) = r̃⁺(y) = r̃⁺(x_k)` and `ω̃(y) = 0`; for `y ∈ (x_i, x_{i+1})`, set `r̃⁻(y) = r̃⁻(x_i) + ω̃(x_i)`, `r̃⁺(y) = r̃⁺(x_{i+1}) − ω̃(x_{i+1})`, `ω̃(y) = 0` (no stored mass between points). I have to check this extension still respects the true ranks. Take `y ∈ (x_i, x_{i+1})`. Then
`r̃⁻(y) = r̃⁻(x_i) + ω̃(x_i) ≤ r⁻_D(x_i) + ω_D(x_i) = r⁺_D(x_i) ≤ r⁻_D(y)`,
the last step because nothing in `D` lies in the open interval `(x_i, y)`... wait, there *could* be points of `D` in `(x_i, x_{i+1})` that just weren't selected into `S`. Let me be careful: `r⁺_D(x_i)` counts all weight `≤ x_i`, and `r⁻_D(y)` counts all weight `< y`; since `x_i < y`, every point `≤ x_i` is also `< y`, so `r⁺_D(x_i) ≤ r⁻_D(y)` holds regardless of unselected points in between. Good, the inequality is safe. Symmetrically `r̃⁺(y) = r̃⁺(x_{i+1}) − ω̃(x_{i+1}) ≥ r⁺_D(x_{i+1}) − ω_D(x_{i+1}) = r⁻_D(x_{i+1}) ≥ r⁺_D(y)`, using `y < x_{i+1}` so everything `≤ y` is `< x_{i+1}`. So the extended functions obey the same one-sided bounds at every `y`, and the interleaving constraints carry over by transitivity. That extension property is the workhorse for everything below — the constraints on the stored points propagate to constraints everywhere.

It's also worth nailing that the `ε`-condition only has to be checked at and between consecutive stored points. For `y ∈ (x_i, x_{i+1})`, by the extension `r̃⁺(y) − r̃⁻(y) − ω̃(y) = [r̃⁺(x_{i+1}) − ω̃(x_{i+1})] − [r̃⁻(x_i) + ω̃(x_i)] − 0`. So if I guarantee both `r̃⁺(x_i) − r̃⁻(x_i) − ω̃(x_i) ≤ ε ω(D)` (the at-a-point condition) and `r̃⁺(x_{i+1}) − r̃⁻(x_i) − ω̃(x_{i+1}) − ω̃(x_i) ≤ ε ω(D)` (the between-points condition), the whole continuum is `ε`-approximate. Two discrete conditions per gap; finite to check.

Initial summary: from a small raw multiset, take `S` = all distinct values and set `r̃⁺, r̃⁻, ω̃` to the exact `r⁺_D, r⁻_D, ω_D`. That's `0`-approximate — it answers everything exactly. Now the two operations.

For merging, take two summaries `Q(D_1)` with error `ε_1` and `Q(D_2)` with error `ε_2`, over `D = D_1 ∪ D_2`. Take `S = S_1 ∪ S_2` and add the *extended* functions pointwise: `r̃⁻(x_i) = r̃⁻_{D_1}(x_i) + r̃⁻_{D_2}(x_i)`, same for `r̃⁺` and `ω̃` (using each summary's extension to evaluate at points it doesn't store). Because the true ranks are additive over the union — `r⁻_D(y) = r⁻_{D_1}(y) + r⁻_{D_2}(y)`, likewise `r⁺` and `ω` — and each summand respects its own one-sided bounds, the sum respects them too, so the merged thing is a valid summary. Its error: for any `y`,
`r̃⁺(y) − r̃⁻(y) − ω̃(y) = [r̃⁺_{D_1}(y) − r̃⁻_{D_1}(y) − ω̃_{D_1}(y)] + [r̃⁺_{D_2}(y) − r̃⁻_{D_2}(y) − ω̃_{D_2}(y)] ≤ ε_1 ω(D_1) + ε_2 ω(D_2) ≤ max(ε_1, ε_2) ω(D_1 ∪ D_2)`,
since `ω(D) = ω(D_1) + ω(D_2)`. So merging gives a `max(ε_1, ε_2)`-approximate summary — error does *not* accumulate under merge. That's the property that makes hierarchical/distributed aggregation work: I can merge a whole tree of chunk-summaries and the error is just the worst single chunk's.

Merging keeps growing `S`; I have to shrink it back without wrecking the bound. Introduce a query: given a target rank `d ∈ [0, ω(D)]`, return a stored point whose true rank is close to `d`. The natural choice is to compare `d` against the *midpoints* `½(r̃⁻(x_i) + r̃⁺(x_i))`: return `x_1` if `d` is below the first midpoint, `x_k` if above the last, otherwise locate the gap and pick `x_i` or `x_{i+1}` by which side of `½[r̃⁻(x_i) + ω̃(x_i) + r̃⁺(x_{i+1}) − ω̃(x_{i+1})]` the doubled rank `2d` falls. I claim the returned `x* = g(Q,d)` satisfies
`d ≥ r̃⁺(x*) − ω̃(x*) − (ε/2) ω(D)` and `d ≤ r̃⁻(x*) + ω̃(x*) + (ε/2) ω(D)`.
This is a four-case check. Take the general interior case where the query returns `x* = x_i` (the others — `x_1`, `x_k`, and `x_{i+1}` — go the same way, using that the endpoints have exact rank info). The branch condition gives `2d < r̃⁻(x_i) + ω̃(x_i) + r̃⁺(x_{i+1}) − ω̃(x_{i+1})`. Rewrite the right side by adding and subtracting `r̃⁻(x_i) + ω̃(x_i)`:
`2d < 2[r̃⁻(x_i) + ω̃(x_i)] + [ r̃⁺(x_{i+1}) − ω̃(x_{i+1}) − r̃⁻(x_i) − ω̃(x_i) ]`,
and the bracket is exactly the between-points quantity `≤ ε ω(D)`, so `2d ≤ 2[r̃⁻(x_i) + ω̃(x_i)] + ε ω(D)`, giving `d ≤ r̃⁻(x_i) + ω̃(x_i) + (ε/2) ω(D)`. For the other direction, the gap was located so `2d ≥ r̃⁻(x_i) + r̃⁺(x_i)`; rewrite the right side around `r̃⁺(x_i) − ω̃(x_i)`:
`2d ≥ 2[r̃⁺(x_i) − ω̃(x_i)] − [ r̃⁺(x_i) − ω̃(x_i) − r̃⁻(x_i) ] + ω̃(x_i)`,
and the bracketed term is exactly the at-a-point quantity `r̃⁺(x_i) − r̃⁻(x_i) − ω̃(x_i)`, so it is at most `εω(D)`. Thus `2d ≥ 2[r̃⁺(x_i) − ω̃(x_i)] − εω(D)`, i.e. `d ≥ r̃⁺(x_i) − ω̃(x_i) − (ε/2)ω(D)`. Both halves hold. The query returns a point whose rank brackets `d` to within `(ε/2)ω(D)`.

Now prune: to cut down to a budget `b`, query the summary at the `b+1` evenly spaced ranks `d = (i-1)/b · ω(D)` for `i = 1, …, b+1`, and keep `x'_i = g(Q, (i-1)/b · ω(D))`, copying the stored `r̃` values restricted to the kept points (drop duplicates). All kept points come from the original `Q`, so the structural constraints survive — it's a valid summary. Its error: apply the query lemma at consecutive ranks. For `x'_i` and `x'_{i+1}`,
`(i-1)/b · ω(D) − (ε/2) ω(D) ≤ r̃⁻(x'_i) + ω̃(x'_i)` and `i/b · ω(D) + (ε/2) ω(D) ≥ r̃⁺(x'_{i+1}) − ω̃(x'_{i+1})`.
Subtract:
`r̃⁺(x'_{i+1}) − ω̃(x'_{i+1}) − r̃⁻(x'_i) − ω̃(x'_i) ≤ [ i/b + ε/2 − (i-1)/b + ε/2 ] ω(D) = (1/b + ε) ω(D)`.
That's the between-points condition for the pruned summary with error `ε + 1/b`. So prune takes an `ε`-approximate summary to an `(ε + 1/b)`-approximate one with at most `b+1` points. Same guarantee shape as the classic unweighted sketch, now for arbitrary weights. With merge keeping error flat and prune adding only `1/b` per pruning, a streaming/distributed pipeline of merges-and-prunes keeps total error bounded by a controllable `ε`. The weighted quantile sketch is the rigorous primitive the approximate split finder was missing, and the weights it carries are precisely the curvatures `h_i` from the second-order objective — the two parts of the design are the same idea seen twice.

Back up to the data, because there's a second scaling pain I've been ignoring: sparsity. Real inputs are mostly empty — missing values, frequent zeros, and one-hot encodings that blow one categorical into a thousand mostly-zero columns. If my split finder visits every cell, I'm paying for an ocean of zeros, and I have no principled rule for *where a missing value goes* at a split. The usual hacks (impute a mean, treat zero as a real value, special-case categoricals) are all guesses. Let me make the algorithm decide. Add to each split node a default direction: any instance whose split feature is missing (or absent in the sparse representation) is sent the default way. And don't *guess* the default — *learn* it, from the same gain criterion. For a given feature, only the non-missing entries `I_k` have a value to sort by; the missing ones form a block whose `(G, H)` is `(G − Σ_{I_k} g, H − Σ_{I_k} h)`. So I scan the non-missing entries twice. First, sweep them in ascending order accumulating `G_L, H_L`, with the missing block assigned to the *right* (i.e. `G_R, H_R` include the missing mass): at each threshold score `G_L²/(H_L+λ) + G_R²/(H_R+λ) − G²/(H+λ)`. Then sweep them in descending order accumulating `G_R, H_R`, with the missing block assigned to the *left*. Take the best gain over both sweeps; the winning sweep's direction is the learned default. Crucially, both sweeps touch only `I_k`, the non-missing entries — so the cost of split finding is *linear in the number of non-zeros*, not in the dense matrix size. Sparsity stops being a problem to mitigate and becomes a speedup. And the same default-direction idea drops straight into the bucketed approximate setting: just accumulate only non-missing entries into buckets.

Two more regularizers I'll fold in, because they're cheap and they're known to help. Shrinkage scales each newly added tree by a factor `η ∈ (0,1)` before appending, `F_t = F_{t-1} + η f_t`. This is Friedman's learning rate — it reduces the influence of any single tree and leaves room for later trees to keep improving, generally generalizing better than just stopping early; `η` and the number of trees trade off. Column subsampling means that, at each tree, I consider only a random subset of features. This is the RandomForest trick — it decorrelates the trees and, per a lot of practitioner experience, prevents overfitting even more than subsampling rows; it also makes the parallel split finding cheaper because each tree touches fewer columns. (Row subsampling, stochastic gradient boosting, is the other knob and helps too.) These sit *outside* the per-tree objective but on top of the regularized core; they don't change any of the derivations above.

Now the scaling story, which I've been promising the regularized core would also unlock — and it does, because the score only needs *aggregated* `(G, H)` over instance sets, and aggregation parallelizes and bucketizes cleanly. The expensive operation is getting data into feature-sorted order for the scan. So I won't re-sort per node. I'll store the data once, before training, in compressed-column (CSC) blocks where each column is pre-sorted by its feature value, and reuse that layout across every iteration and every node. A single linear scan over a sorted column enumerates all split candidates for that feature at every leaf simultaneously (I find splits for all leaves in one pass). This is where the `log n` per scan goes away — sort once, amortize forever. For the exact greedy single-machine case the whole dataset is one block; for approximate/out-of-core/distributed, use multiple blocks, one per row-subset, distributed across machines or spilled to disk; the weighted-quantile candidate finding becomes a linear scan over the sorted columns, which is exactly what makes the local-proposal variant affordable. Column subsampling is trivial here — just pick a subset of columns from a block — and per-column statistic collection parallelizes directly, so this layout *is* the parallel split-finding algorithm.

The block layout creates one new problem I should head off. Scanning a column in feature-sorted order means the gradient statistics `(g_i, h_i)` are fetched by *row index in non-contiguous order* — an indirect, cache-unfriendly access. On large data those scattered fetches miss the CPU cache and stall the accumulation. Fix it by decoupling the fetch from the accumulate: each thread prefetches a batch of the needed `(g_i, h_i)` into a small internal buffer, then accumulates from the buffer in a tight loop. That turns the immediate read-after-fetch dependency into a longer, pipelineable one. For the approximate algorithm the analogous lever is block *size*: too-small blocks underutilize threads, too-large blocks overflow the cache with gradient statistics; a block of about `2^16` instances balances parallelism against cache residency. And for truly out-of-core data: store blocks on disk, prefetch with an independent thread so disk IO overlaps compute, compress blocks by column and decompress on the fly (trading a little CPU for a lot less disk traffic), and shard across multiple disks so several prefetchers feed the trainer in parallel. None of this touches the math; it's all in service of feeding the same `(G, H)`-aggregation engine fast enough.

Let me also re-derive the simplest special case end to end to make sure the whole construction is self-consistent. Squared error, `l = ½(y − ŷ)²`: then `g_i = ŷ_i − y_i` (the negative residual) and `h_i = 1`. The optimal leaf weight is `w_j* = −(Σ_{i∈I_j}(ŷ_i − y_i)) / (n_j + λ)` — for `λ = 0` that's the mean residual in the leaf, exactly LS-TreeBoost. The instance weights `h_i = 1` are all equal, so the weighted quantile sketch reduces to ordinary quantiles. The structure score becomes `−½ Σ_j (Σ_{i∈I_j} residual)² / (n_j + λ) + γT`. Everything collapses to the classical first-order picture when the curvature is constant, which is the consistency check I wanted: this construction contains gradient boosting and adds (a) genuine second-order leaf values and split scores for non-quadratic losses, (b) an explicit complexity penalty that doubles as a pruning threshold, and (c) the weighted-quantile + sparsity-aware + block machinery that lets it run at scale.

So let me write the algorithm I'd actually ship, filling the open slot from the harness — the base learner is a regularized second-order regression tree, grown by the derived gain, with the leaf weight `−G/(H+λ)`, the missing-value default direction, and the loss reaching the tree only through `(g, h)`. I'll write a compact but faithful reference of the core, then the production usage.

```python
import numpy as np


def _leaf_score(G, H, lam):
    # value contribution of a leaf at its optimal weight w* = -G/(H+lam):
    #   min_w [ G w + 0.5 (H+lam) w^2 ] = -0.5 * G^2 / (H+lam)
    return G * G / (H + lam)


class _Node:
    __slots__ = ("feat", "thresh", "default_left", "left", "right", "weight")

    def __init__(self):
        self.feat = -1            # split feature (-1 => leaf)
        self.thresh = 0.0
        self.default_left = True  # learned direction for missing values
        self.left = self.right = None
        self.weight = 0.0         # leaf score w* (only if leaf)


class RegularizedTree:
    """One base learner: a second-order regression tree grown on (g, h).
    Split gain = 0.5[ G_L^2/(H_L+lam) + G_R^2/(H_R+lam) - G^2/(H+lam) ] - gamma;
    a split is kept only if its gain > 0 (the gamma term is the prune threshold).
    Missing values (NaN) flow to a default direction learned from the same gain."""

    def __init__(self, max_depth=6, lam=1.0, gamma=0.0, min_child_h=1.0):
        self.max_depth = max_depth
        self.lam = lam              # L2 on leaf weights (the +lambda in w*)
        self.gamma = gamma          # per-leaf cost == minimum split gain
        self.min_child_h = min_child_h

    def fit(self, X, g, h):
        self.root = self._build(X, g, h, np.arange(len(g)), depth=0)
        return self

    def _build(self, X, g, h, idx, depth):
        node = _Node()
        G, H = g[idx].sum(), h[idx].sum()
        node.weight = -G / (H + self.lam)            # optimal leaf weight
        if depth >= self.max_depth or len(idx) <= 1:
            return node
        best = self._best_split(X, g, h, idx, G, H)
        if best is None:
            return node                              # no gain > 0 => prune (leaf)
        feat, thresh, default_left, left_idx, right_idx = best
        node.feat, node.thresh, node.default_left = feat, thresh, default_left
        node.weight = 0.0
        node.left = self._build(X, g, h, left_idx, depth + 1)
        node.right = self._build(X, g, h, right_idx, depth + 1)
        return node

    def _best_split(self, X, g, h, idx, G, H):
        lam, parent = self.lam, _leaf_score(G, H, self.lam)
        best_gain, best = 0.0, None                  # gain must clear 0 (i.e. clear gamma)
        for feat in range(X.shape[1]):               # (column subsampling would restrict this)
            col = X[idx, feat]
            present = ~np.isnan(col)                  # sparsity-aware: only non-missing entries
            pid = idx[present]                        # row ids with this feature observed
            mid = idx[~present]                       # row ids whose split feature is missing
            if len(pid) < 2:
                continue
            order = np.argsort(col[present], kind="stable")
            pid = pid[order]                          # present rows, sorted by feature value
            vals = col[present][order]
            gp, hp = g[pid], h[pid]
            # Alg. 3 (sparsity-aware): two passes deciding the missing block's side.
            # Pass A -- missing -> right: left child = a prefix of the present rows.
            csg, csh = np.cumsum(gp), np.cumsum(hp)
            for s in range(1, len(pid)):             # split between present rows s-1 and s
                if vals[s] == vals[s - 1]:
                    continue                         # no split between equal feature values
                gl, hl = csg[s - 1], csh[s - 1]      # left = present prefix
                gr, hr = G - gl, H - hl              # right = present suffix + missing block
                if hl < self.min_child_h or hr < self.min_child_h:
                    continue
                gain = 0.5 * (_leaf_score(gl, hl, lam)
                              + _leaf_score(gr, hr, lam) - parent) - self.gamma
                if gain > best_gain:
                    best_gain = gain
                    thr = 0.5 * (vals[s] + vals[s - 1])
                    right_idx = np.concatenate([pid[s:], mid])
                    best = (feat, thr, False, pid[:s], right_idx)  # default_left = False
            # Pass B -- missing -> left: right child = a suffix of the present rows.
            Gp_tot, Hp_tot = csg[-1], csh[-1]        # total present (g, h) on this feature
            for s in range(1, len(pid)):             # right = present rows [s:]
                if vals[s] == vals[s - 1]:
                    continue
                gr, hr = Gp_tot - csg[s - 1], Hp_tot - csh[s - 1]  # present suffix only
                gl, hl = G - gr, H - hr              # left = present prefix + missing block
                if hl < self.min_child_h or hr < self.min_child_h:
                    continue
                gain = 0.5 * (_leaf_score(gl, hl, lam)
                              + _leaf_score(gr, hr, lam) - parent) - self.gamma
                if gain > best_gain:
                    best_gain = gain
                    thr = 0.5 * (vals[s] + vals[s - 1])
                    left_idx = np.concatenate([pid[:s], mid])
                    best = (feat, thr, True, left_idx, pid[s:])  # default_left = True
        return best

    def _predict_one(self, x):
        node = self.root
        while node.feat != -1:
            v = x[node.feat]
            go_left = node.default_left if np.isnan(v) else (v < node.thresh)
            node = node.left if go_left else node.right
        return node.weight

    def predict(self, X):
        return np.array([self._predict_one(X[i]) for i in range(X.shape[0])])


class BoostedTrees:
    """Forward stagewise second-order boosting: each round computes (g, h) of the
    loss at the current prediction, grows a RegularizedTree on them, shrinks by
    the learning rate, and appends. The loss enters only through (g, h)."""

    def __init__(self, n_rounds=100, learning_rate=0.1, max_depth=6,
                 lam=1.0, gamma=0.0, loss="squarederror"):
        self.n_rounds = n_rounds
        self.lr = learning_rate
        self.max_depth = max_depth
        self.lam = lam
        self.gamma = gamma
        self.loss = loss
        self.trees = []

    def _grad_hess(self, y, yp):
        if self.loss == "squarederror":              # l = 0.5 (y - yp)^2
            return (yp - y), np.ones_like(y)         # g = yp - y, h = 1
        if self.loss == "logistic":                  # binary logloss, y in {0,1}
            p = 1.0 / (1.0 + np.exp(-yp))
            return (p - y), np.maximum(p * (1 - p), 1e-6)  # g = p - y, h = p(1-p)
        raise ValueError(self.loss)

    def fit(self, X, y):
        y = np.asarray(y, dtype=float)
        if self.loss == "squarederror":
            self.base = float(np.mean(y))
        else:
            p0 = np.clip(np.mean(y), 1e-6, 1 - 1e-6)
            self.base = float(np.log(p0 / (1 - p0)))
        yp = np.full(len(y), self.base)
        for _ in range(self.n_rounds):
            g, h = self._grad_hess(y, yp)            # second-order signal of the loss
            tree = RegularizedTree(self.max_depth, self.lam, self.gamma).fit(X, g, h)
            self.trees.append(tree)
            yp += self.lr * tree.predict(X)          # shrinkage: scale each tree by lr
        return self

    def predict(self, X):
        out = np.full(X.shape[0], self.base)
        for tree in self.trees:
            out += self.lr * tree.predict(X)
        return out
```

And the production form is the same algorithm with the scalable engine (pre-sorted column blocks, the histogram-binned weighted-quantile split finder, cache-aware prefetch, out-of-core blocks, parallel/distributed allreduce) underneath, exposed as a regressor. On a black-box regression task I'd just hand it engineered features and let the boosted second-order trees fit:

```python
from xgboost import XGBRegressor

model = XGBRegressor(
    objective="reg:squarederror",  # l = 0.5 (y - yhat)^2  =>  g = yhat - y, h = 1
    n_estimators=120,              # number of boosting rounds (trades off with learning_rate)
    max_depth=3,                   # shallow trees = weak learners
    learning_rate=0.05,            # shrinkage eta: scale each new tree
    subsample=0.9,                 # stochastic gradient boosting (row subsample)
    colsample_bytree=0.8,          # RandomForest-style column subsample
    reg_lambda=1.0,                # the +lambda in w* = -G/(H+lambda)
    tree_method="hist",            # histogram-binned weighted-quantile split finder
)
model.fit(X_train, y_train)        # builds pre-sorted blocks once, then boosts
y_pred = model.predict(X_test)
```

The causal chain, start to finish: tree boosting wins on accuracy but the standard recipe grows the tree from the bare gradient and then sets leaf values by a separate per-leaf line search, with no complexity penalty in the split criterion — two criteria where there should be one. Reading FHT's adaptive-Newton view, the curvature `h_i` is already implicit (LogitBoost's weight is the Hessian), so I fold gradient *and* curvature into a single regularized objective `Σ_i l + Σ_k (γT + ½λ‖w‖²)`; second-order Taylor reduces each round to `Σ_i [g_i f + ½ h_i f²] + Ω`, which depends on the loss only through `(g, h)` and so serves any objective. Minimizing the per-leaf quadratic gives `w_j* = −G_j/(H_j+λ)` (the Newton step, with `λ` damping flat leaves) and the structure score `−½ Σ_j G_j²/(H_j+λ) + γT`; the split gain `½[G_L²/(H_L+λ)+G_R²/(H_R+λ)−G²/(H+λ)] − γ` falls out, with `γ` doubling as the prune threshold. Exact greedy split finding sorts and scans; to scale, an approximate finder proposes candidate splits, and completing the square shows the right candidates are *`h`-weighted* quantiles — which forces a weighted quantile sketch, built with merge (`max(ε₁,ε₂)`) and prune (`ε+1/b`) operations whose proofs preserve the `ε`-guarantee. Sparsity is handled by a learned default direction that visits only non-missing entries, making split finding linear in the non-zeros. Shrinkage and column subsampling add cheap regularization on top. And the whole thing scales because the score needs only aggregated `(G,H)`: pre-sorted column blocks kill the per-iteration sort, cache-aware prefetch hides the scattered gradient fetches, and compressed sharded blocks take it out of core — one end-to-end system that is both the accurate learner and the scalable one.
