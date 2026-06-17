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

So I am going to do the thing the drift framing resists: drop sequence-model adaptation entirely and
fit the strongest plain, *full-feature*, non-adaptive tabular learner I can, and let it be the
control the adaptive models have to beat. The hypothesis is concrete and falsifiable — that on this
panel a well-tuned gradient-boosted tree over all 158 Alpha158 factors generalizes across the three
regimes *better* than either adaptive sequence model, because it spends none of its capacity on
adaptation machinery and all of it on extracting cross-sectional signal, and because trees on
RobustZScore-free raw factors are naturally robust to the marginal drift that hurts a neural encoder.
Let me derive why a GBDT is the right control and exactly how the scaffold wants it filled.

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

Now the scaffold specifics, because the fill is a *faithful* qlib `LGBModel`, not a generic GBDT, and
a couple of details in it look like adaptation but are not. The hyperparameters are the official
Alpha158 CSI300 benchmark: `learning_rate=0.2`, `num_leaves=210`, `max_depth=8` (a leaf-wise tree
engine with both a leaf budget and a depth cap to bound the leaf-wise overfit), and the load-bearing
ones for this noisy data, `lambda_l1=205.6999` and `lambda_l2=580.9768` — enormous leaf-weight
penalties that shrink every leaf value hard, which is precisely the regularization that keeps the
trees from chasing the unlearnable residuals. `colsample_bytree=0.8879` and `subsample=0.8789` are
the booster's own per-tree feature and row sub-sampling parameters forwarded to `lgb.train`; they are
*not* a request for the gradient-based one-side sampling the method is famous for, and they are *not*
temporal-domain adaptation — they are vanilla stochastic-GBDT knobs that add a little member-to-member
decorrelation. The fill trains with `num_boost_round=1000` and `early_stopping_rounds=50` against the
validation segment (2015–2016), so the actual tree count is chosen by validation, and it is
deterministic, which is why I expect (and the harness shows) identical numbers across the three seeds.

The data view is the other place the scaffold makes a deliberate choice, and it is the right one for
trees. The fill edits the workflow processor block to set `infer_processors: []` — it removes the
`RobustZScoreNorm` and `Fillna` that the neural baselines depend on, and keeps only the label
processors (`DropnaLabel`, `CSRankNorm`). That is correct GBDT practice: trees split on raw factor
thresholds and gain nothing from per-feature z-scoring (which would, if anything, smear the natural
split points), and LightGBM handles NaNs natively by learning a default direction, so there is no
need to fill them. So the tree baseline sees the *raw* Alpha158 factors over the default `DatasetH`
(no sequence sampler), with `fit` preparing train+valid `lgb.Dataset`s, training one booster with
early stopping, and `predict` returning `pd.Series(model.predict(x_test), index=x_test.index)` over
the test segment. There is no domain split, no alignment, no router — by design. The full module is
the fill of `custom_model.py` plus the one processor-block YAML edit; it is in the answer.

Let me reason carefully about what this control should do, because if it merely ties the adaptive
models the trajectory has no strong rung, and if it wins it reframes the whole task. On
`csi300_shifted` I expect LGBM to be competitive but perhaps a touch behind AdaRNN's strong showing
there (AdaRNN's alignment genuinely helps where the shift is mild and the near window rewards a
sequence encoder). The decisive regime is `csi300_recent`. Both adaptive models died there
(info ratio ≈0.47–0.48); my falsifiable prediction is that the tree, seeing all factors and immune to
marginal rescaling, will *not* die there — its `csi300_recent` annualized return and info ratio should
come in clearly above the adaptive models' (I expect info ratio comfortably above 1.0 and annualized
return well above their ≈0.03–0.04), because the relation the trees learn from raw thresholds is more
portable to a distant regime than a neural encoder's drifted representation. And on the long-horizon
`csi300` window I expect LGBM to be the best of the three on the signal metrics (higher IC and
rank IC) with a markedly *shallower max drawdown* — the strong regularization should make its
portfolio steadier than the adaptive models', whose drawdowns ran to −0.10. If those three predictions
hold — competitive on the near window, decisively better on the far window, best signal and shallowest
drawdown on the long window — then the aggregate gmean puts LGBM above both adaptive models, and the
trajectory's lesson is sharp and a little subversive: on a thin, noisy CSI300 panel, the most
effective response to concept drift among these baselines is *not* an explicit drift-adaptation
mechanism but a strong, heavily-regularized, full-feature non-adaptive learner whose inductive bias
(threshold splits, leaf shrinkage) is already robust to the marginal shift. The failure mode that
would refute this is the symmetric one: if LGBM's `csi300_recent` also lands near info ratio 0.5, then
the far regime is simply unforecastable for everyone and the ceiling is the data, not the model. The
numbers will decide it — and if LGBM does win as the strong control, the natural next move is not to
abandon it but to make *it* drift-robust by ensembling it deliberately, reweighting the samples it
learns from and the features it uses, which is exactly where a stronger method would build from here.
