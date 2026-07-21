AdaRNN's numbers settle the question the last two rungs were circling, and the answer is not the one
the research framing wanted. On `csi300_shifted` AdaRNN does well — mean IC 0.0525, info ratio 1.66,
even edging TRA's 1.59 — and its best seed there hits info ratio 1.82. But on `csi300_recent`, the
regime that already broke TRA, it stays broken: mean info ratio 0.47, essentially tied with TRA's
0.48, and now with the seed variance I predicted — seed 42 comes in at info ratio 0.004 and
annualized return 0.0003 on that window, a near-dead run, while seed 456 recovers to 1.12. So
explicit two-half temporal alignment, exactly the "learn the invariant" move I reached for, did *not*
rescue the far regime; it produced the same far-regime collapse plus instability. And the aggregate
tells the deeper story: across all three regimes, gmean of the per-regime metric means, AdaRNN
(≈0.561) and TRA (≈0.560) are essentially tied, both *adaptive* sequence models, both heavy, both
weak on the window that matters most for drift. Two independent adaptation mechanisms — routing
within a distribution, aligning across halves — converged to the same ceiling. That is the signal I
have to act on: the bottleneck on this task is not the adaptation *mechanism*; it is the
fit-able signal a sequence model can extract from a thin, noisy CSI300 panel, and both models pay an
overhead (a 20-factor `FilterCol` slice for TRA, a single-timestep GRU for AdaRNN) that costs more
than the drift-robustness it buys. AdaRNN's seed-42 `csi300_recent` near-zero is the extreme of that:
a model that aligned two training halves and then had nothing informative to say about 2019–2020.

The seed variance I flagged for AdaRNN is worth quantifying, because it is the tell that the far
regime is being solved by luck rather than mechanism. Its three `csi300_recent` info ratios are
0.004, 0.283, and 1.118; their mean is 0.468 and their spread is enormous — a standard deviation of
about 0.47, a coefficient of variation near 1.0, so the run-to-run noise on that window is as large as
the signal itself. TRA, deterministic, sat flat at 0.480 there. So the two adaptive models arrive at
the same *mean* on the decisive regime by opposite routes: TRA by a stable-but-weak 0.48, AdaRNN by
averaging a dead seed and a good one. Neither is a model I would trust to forecast a distant regime;
one is reliably mediocre, the other is a coin flip. The near windows sharpen the point. On `csi300`
AdaRNN's info ratio (1.089) actually edges TRA's (1.008), about +8%, yet its IC (0.0384) sits *below*
TRA's (0.0420) by a similar 8% — the alignment slightly improved the top-of-book ordering the backtest
trades while slightly degrading the average correlation. On `csi300_shifted` AdaRNN gains modestly on
both (IC +2.5%, info ratio +4.8%). Read together: alignment bought a little on the two near windows and
nothing dependable on the far one — a small, regime-local benefit, not the drift robustness the whole
framing was after.

So I am going to do the thing the drift framing resists: drop sequence-model adaptation entirely and
fit the strongest plain, *full-feature*, non-adaptive tabular learner I can, and let it be the
control the adaptive models have to beat. The hypothesis is concrete and falsifiable — that on this
panel a well-tuned gradient-boosted tree over all 158 Alpha158 factors generalizes across the three
regimes *better* than either adaptive sequence model, because it spends none of its capacity on
adaptation machinery and all of it on extracting cross-sectional signal, and because trees on
RobustZScore-free raw factors are naturally robust to the marginal drift that hurts a neural encoder.
Let me derive why a GBDT is the right control and exactly how the scaffold wants it filled.

The alternatives deserve a hearing, since "give up on sequence adaptation" is a strong claim. One move
is to keep a sequence model but make it heavier or smarter — a real multi-step window with attention
instead of AdaRNN's degenerate single timestep, or a domain-adversarial alignment (gradient reversal)
instead of a first-moment cosine match. I reject this on the arithmetic of the ceiling: two
structurally different adaptation mechanisms already converged to the same aggregate and the same
far-regime collapse, and what they share is not their adaptation design, it is the data — an IC around
0.05, an R² near 0.0025, a panel where about 99.75% of each label is noise. A larger attention model
has *more* parameters to fit that noise and needs *more* clean signal per sample to justify them,
exactly what this panel lacks; it would raise variance, not the ceiling. A second move is explicit
domain alignment across regimes (MMD or DANN) on a flat feature view — but that is AdaRNN's idea
without the recurrence, and it inherits the same defect: aligning training-era distributions says
nothing about a 2019–2020 window that lies outside them. Both spend more capacity on the axis that is
already saturated. The move with actual headroom is orthogonal: stop trying to *learn* drift-robustness with a
high-variance encoder and instead pick a learner whose *inductive bias* is drift-robust for free and
whose regularization is built for noise. That is a gradient-boosted tree, and let me derive why the
scaffold's exact fill is the right instance of it.

A GBDT is an additive sequence of regression trees. At round `t`, with current ensemble `F_{t-1}`,
each sample gets a gradient `g_i = ∂L(y_i, F_{t-1}(x_i))/∂F`; for the squared loss `objective=mse`
this is `ŷ_i − y_i`, so `−g_i` is the residual and each round fits a tree to the residuals, folded in
shrunk by the learning rate. Inside one tree the split is chosen to maximize the variance gain — in
sum-of-gradient form, `V_j(d) = (1/n)[(Σ_{x_ij≤d} g_i)²/n_l + (Σ_{x_ij>d} g_i)²/n_r]` — searched per
feature over candidate thresholds `d`; LightGBM does this on bucketed histograms (each factor binned
into ≤255 bins, one pass per node accumulating per-bin gradient sums, the sibling histogram recovered
by parent-minus-smaller-child subtraction) and grows leaf-wise (always split the leaf with the largest
gain), which is why `num_leaves` and `max_depth` are both capped to bound the leaf-wise overfit. Why is
this the right thing for return prediction across regimes? Three reasons that bear directly on the
failure I just saw. First, trees split on *thresholds* of individual factors, so they are invariant to
any monotone rescaling of a factor — the split `x_ij ≤ d` only cares about rank order, not scale or
mean. That is the crux for drift: temporal covariate shift moves `P(x)` — the level, spread, and
correlations of the factors drift between 2008–2014 and 2019–2020 — but a relation expressed as a
rank threshold on a single factor is far more portable across that shift than a neural encoder whose
first layer takes a fixed linear combination of all 158 factors, a combination whose meaning changes
the moment the factor marginals move. The very drift that made TRA's router read a stale error-history
signal and AdaRNN align two halves that don't span the test regime does not move a tree's split points
the same way; the tree's inductive bias is *already* the kind of robustness the adaptive models tried
to learn. Second, a GBDT sees the *full* 158-factor view, not a curated slice, so it loses none of the
factor information the adaptive models discarded — no `FilterCol` to 20 columns, no collapse to a
single timestep. Third, boosting with heavy L1/L2 leaf regularization is a strong defense against the
noise that dominates this panel: with leaf weight `w* = −G/(H+λ2)` and L1 entering as a soft-threshold
on `G`, large penalties shrink every leaf value toward zero, so the trees commit only to splits whose
gradient mass clearly survives the regularizer — exactly the right posture on low-SNR data where the
unregularized optimum would memorize idiosyncratic returns. It is the consensus high-accuracy learner
on exactly this kind of tabular data, and on this task it is the control the two adaptive models have
spent four rungs failing to clearly beat.

The regularization is the whole reason to expect this tree to survive drift where the encoders did
not. For squared loss a leaf of `n` samples takes value `w* = −G/(n + λ2)` with `λ2 = 580.98`, so a
100-sample leaf keeps only about `100/681 ≈ 15%` of its unregularized magnitude and even a 500-sample
leaf keeps under half. On top of that `λ1 = 205.7` soft-thresholds the gradient sum, zeroing any leaf
whose accumulated `|G|` fails to clear it. Together they let the booster commit a nonzero prediction
only where the gradient mass is both large and consistent — on low-SNR returns, exactly the discipline
that separates a real cross-sectional edge from a lucky residual. It is the tree analogue of the
regularizer role I gave AdaRNN's matching loss, except aimed directly at the noise instead of at
cross-half alignment, and costing no adaptation machinery.

Now the scaffold specifics, because the fill is a *faithful* qlib `LGBModel`, not a generic GBDT, and
a couple of details in it look like adaptation but are not. The hyperparameters are the official
Alpha158 CSI300 benchmark: `learning_rate=0.2`, `num_leaves=210`, `max_depth=8` (a leaf-wise tree
engine with both a leaf budget and a depth cap to bound the leaf-wise overfit), and the load-bearing
ones for this noisy data, `lambda_l1=205.6999` and `lambda_l2=580.9768` — enormous leaf-weight
penalties that shrink every leaf value hard, which is precisely the regularization that keeps the
trees from chasing the unlearnable residuals. The `num_leaves=210`/`max_depth=8` pairing is worth
reading rather than skimming: a fully balanced depth-8 tree holds `2^8 = 256` leaves, so a 210-leaf
budget under a depth-8 cap lets the leaf-wise engine grow almost-full trees that are nonetheless
kept shallow — wide but not deep. On its own that is a lot of capacity, enough to carve the panel into
210 cells per round; it is only safe *because* the enormous `λ1`/`λ2` then shrink or zero most of those
cells' values, so the two knobs work as a pair — the leaf budget supplies expressive splits and the
penalties refuse to trust most of them. That is the right division of labor on noisy data: let the tree
find structure freely, then make it pay dearly to act on any of it. `colsample_bytree=0.8879` and
`subsample=0.8789` are
the booster's own per-tree feature and row sub-sampling parameters forwarded to `lgb.train`; they are
*not* a request for the gradient-based one-side sampling the method is famous for, and they are *not*
temporal-domain adaptation — they are vanilla stochastic-GBDT knobs that add a little member-to-member
decorrelation. The fill trains with `num_boost_round=1000` and `early_stopping_rounds=50` against the
validation segment (2015–2016), so the actual tree count is chosen by validation, and it is
deterministic, which is why I expect identical numbers across the three seeds.

The data view is the other place the scaffold makes a deliberate choice, and it is the right one for
trees. The fill edits the workflow processor block to set `infer_processors: []` — it removes the
`RobustZScoreNorm` and `Fillna` that the neural baselines depend on, and keeps only the label
processors (`DropnaLabel`, `CSRankNorm`). That is correct GBDT practice: trees split on raw factor
thresholds and gain nothing from per-feature z-scoring (which would, if anything, smear the natural
split points), and LightGBM handles NaNs natively by learning a default direction, so there is no need
for external imputation. The one processor the fill keeps is worth dwelling on, because it lines up with what the TRA table
already taught me. `CSRankNorm` on the label cross-sectionally rank-normalizes the returns each day, so
the tree is fit to predict a stock's *rank* within the day's cross-section rather than its raw return
magnitude. That is the same axis I found most drift-stable in TRA's numbers — its rank IC decayed to
73% across regimes while plain IC fell to 46% — so training the tree against a rank target, on
scale-invariant threshold splits, stacks two rank-based robustnesses on top of each other: a rank label
learned by a rank-invariant learner. It is a coherent bet that the *ordering* of returns is the portable
part of the signal, which is exactly the part the top-50/drop-5 backtest trades. So the tree baseline
sees the *raw* Alpha158 factors over the default `DatasetH`
(no sequence sampler), with `fit` preparing train+valid `lgb.Dataset`s, training one booster with
early stopping, and `predict` returning `pd.Series(model.predict(x_test), index=x_test.index)` over
the test segment. There is no domain split, no alignment, no router — by design. The full module is
the fill of `custom_model.py` plus the one processor-block YAML edit; it is in the answer.

There is a second, quieter prediction hiding in the regularization, and it is about drawdown rather
than signal. Both encoders ran max drawdowns around −0.09 to −0.10 across the near windows; a
portfolio's drawdown is driven by the tail of wrong, confident bets, and a confident wrong bet is
exactly what an under-regularized model makes when it trusts a memorized idiosyncratic residual. The
tree's `λ1`/`λ2` do the opposite of trusting: I showed a typical leaf is shrunk to ~15% of its natural
magnitude and sub-threshold leaves are zeroed outright, so the tree rarely takes a large position on
thin evidence. That should show up as a *shallower* drawdown than the encoders posted, and if it does
it is corroborating evidence that the shrinkage — not luck — is what makes the tree steady. A tree that
matched the encoders on IC but also ran a −0.10 drawdown would undercut the whole "regularization buys
robustness" story, so the drawdown is a second, independent place the mechanism can be checked.

So the whole bet reduces to the far regime. The aggregate to clear is the adaptive pair's
≈0.560–0.561, and the geometric mean punishes any collapsed window, so LGBM cannot afford a
`csi300_recent` failure the way the encoders had — a single weak window would drag it back to the tie
even if it dominated the other two. Both encoders posted recent info ratios near 0.48 with recent
annualized returns around 0.033–0.035; if the tree's scale-invariant threshold rules are as portable
as the algebra says, it should clear that collapse zone decisively rather than merely matching it, and
that single cell is where the trajectory's thesis lives or dies. On `csi300_shifted` I expect it
competitive but perhaps a touch behind AdaRNN's alignment-helped showing, since the mild-shift near
window rewards a sequence encoder. And on the long `csi300` window I expect the best signal of the
three (higher IC and rank IC) with a markedly *shallower max drawdown* than the encoders' ≈−0.10,
because the heavy leaf shrinkage keeps the portfolio from large bets on thin evidence. If those hold —
competitive near, decisively better far, best signal and shallowest drawdown long — the aggregate
gmean puts LGBM above both adaptive models, and the trajectory's lesson is sharp and a little
subversive: on a thin, noisy CSI300 panel the most effective response to concept drift among these
baselines is *not* an explicit drift-adaptation mechanism but a strong, heavily-regularized,
full-feature non-adaptive learner whose inductive bias — threshold splits, leaf shrinkage — is already
robust to the marginal shift. The refuting outcome is symmetric: if LGBM's `csi300_recent` also lands
near the encoders' collapse, the far regime is simply unforecastable and the ceiling is the data, not
the model. And if LGBM does win as the strong control, the next move is not to abandon it but to make
*it* drift-robust — ensemble it deliberately, reweighting the samples it learns from and the features
it uses — which is exactly where a stronger method would build from here.
