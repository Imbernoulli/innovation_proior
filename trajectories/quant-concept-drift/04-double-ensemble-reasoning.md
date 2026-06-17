LightGBM did exactly what I hoped and a bit more, and its numbers tell me precisely where the next
gain has to come from. On the long `csi300` window it is the best of the three baselines on signal —
IC 0.0457, rank IC 0.0569 — and crucially its max drawdown is the shallowest of all, −0.0579 against
the adaptive models' −0.09 to −0.10, with information ratio 1.35. On the far `csi300_recent` window,
the regime that killed both adaptive models at info ratio ≈0.48, the tree does *not* die: info ratio
1.10, annualized return 0.0807 — more than double either adaptive model's, and steady across seeds
because it is deterministic. On `csi300_shifted` it is competitive (info ratio 1.30, slightly behind
AdaRNN's alignment-helped 1.66 on the near window, but with a much shallower drawdown). So the
aggregate gmean puts LGBM clearly on top (≈0.571 vs ≈0.560–0.561 for the adaptive pair), and the
trajectory's lesson held: on this thin, noisy panel the most effective response to concept drift among
the baselines is a strong, heavily-regularized, full-feature non-adaptive learner whose threshold-split
inductive bias is already robust to marginal drift. But that very result tells me the tree is not
*finished*. Two soft spots are visible in its own numbers. First, the near window: AdaRNN's explicit
alignment beat the tree on `csi300_shifted` info ratio, so there is drift-adaptation value the single
tree is leaving on the table where the shift is mild. Second, and more fundamentally, a single GBDT
has no defense against the two pathologies of this data beyond its leaf penalties — it overfits the
abundant noise (its later trees chase residuals that are largely unlearnable) and it uses *all*
features *all* the time, so it cannot shed regime-stale factors or down-weight unlearnable samples. The
natural next move is not to abandon the tree but to make *it* drift-robust by ensembling it
deliberately — and that is a real, published technique that fits this exact edit surface.

So the finale is to wrap the LightGBM base learner in an ensemble that is deliberate about two things a
plain bagged or boosted GBDT leaves uniform: which *samples* each new member focuses on, and which
*features* it may use. The framework trains members sequentially and, after each member, runs two
modules driven by the current ensemble's per-sample loss. Let me derive both against the failure I
just measured.

Take the sample axis first, because that is where the noise pathology lives, and LGBM's behavior
already hints at it: a single GBDT's late trees chase residuals, and on return data those residuals
are disproportionately the unlearnable noise (a stock that jumped on an unforecastable headline has
huge loss and nothing to learn from). The boosting reflex — weight up the high-current-loss samples —
is therefore the *wrong sign* here: it leans into the noise and overfits harder, which is exactly why a
single deep-boosted tree needs the enormous `lambda_l1`/`lambda_l2` to survive. I need a per-sample
importance that distinguishes hard-*but-learnable* from hard-*because-noise*, and the scalar final loss
cannot do it because both have high loss. The richer signal is free: a GBDT produces, for each sample,
the *trajectory* of its loss as trees are added. A sample whose loss collapses in the first few trees
was learned from structure the low-variance early trees could capture — transferable. A sample whose
loss stays high and only falls late, when the model is memorizing, is noise. So the start-to-end shape
of the loss curve separates the two. Concretely the Sample Reweighting (SR) module rank-normalizes each
sample's loss curve down the sample axis at each iteration, averages the first 10% of iterations into
`l_start` and the last 10% into `l_end`, and forms the curve statistic `h2 = rank(l_end / l_start)` —
small ratio means the loss collapsed (well-behaved), large ratio means it stalled. It blends this with
the current-ensemble error `h1` (rank-normalized) as `h = α1·h1 + α2·h2`, so it still leans toward
residual error like boosting but the `h2` term injects the learnability discrimination that keeps it
off pure noise. To stabilize, it bins samples by `h` into `bins_sr=10` bins and assigns a single weight
per bin, `1/(decay^k · h_avg + 0.1)` — binning regularizes the weighting, `decay^k` (with `decay=0.5`)
anneals the chase as members accumulate so later members reweight gently, and `+0.1` floors the
denominator. This is the right correction to the single tree: focus each member on under-served but
learnable samples instead of letting heavy leaf penalties bluntly suppress everything.

Now the feature axis, which is where the drift pathology lives — the one the adaptive sequence models
tried and failed to handle, and the one the single LGBM ignores by using every factor always. I want
each successive member trained on a *different, deliberately chosen* feature subset, so the ensemble
does not collapse onto one clique of factors that may be regime-specific. The wrong way is a single
global importance pruning — keep the top factors once — because that bakes in the present regime and
discards factors that will matter when the regime turns, the exact mistake that hurt `csi300_recent`.
So the Feature Selection (FS) module measures each factor's *reliance* by permutation — shuffle its
column, run the current ensemble, and take the standardized loss increase `g = mean(Δloss)/std(Δloss)`
— then bins factors by `g` into `bins_fs=5` bins and samples *from every bin* at declining ratios
(`sample_ratios=[0.8,0.7,0.6,0.5,0.4]`): keep most of the load-bearing bin but always retain a tail of
currently-weak factors, never zero. That deliberate inclusion of dormant factors is what gives the
ensemble exposure to relations that are quiet now but revive in a shifted regime — drift-robustness
built on the feature axis rather than by aligning representations. It is the move the sequence models
gestured at and could not land on a thin panel, expressed in a way the strong tree control can actually
use.

The two modules alternate off the current ensemble's loss: train member `k`, evaluate the
ensemble-so-far on the training data to get the loss values and member `k`'s loss curve, run SR to set
member `k+1`'s sample weights and FS to set its feature subset, train member `k+1`, repeat. The last
member needs neither (nothing follows it). At inference the whole SR/FS apparatus is gone — `predict`
just runs each member on its own stored feature subset and averages, weighted by `sub_weights`
(uniform). So, like the adaptive models, all the adaptation is training-time; unlike them, the
inference is a plain tree-ensemble average with no encoder to drift.

The scaffold fit is the cleanest of the four rungs. The finale fills `custom_model.py` with qlib's
`DEnsembleModel` (`Model`, `FeatureInt`) and *leaves the workflow at the default* `DatasetH` + Alpha158
— the same full-158-raw-factor view the winning LGBM used, no sequence sampler, no special processors.
`fit(dataset)` prepares train+valid via `dataset.prepare(["train","valid"], DK_L)`, then runs the
sequential member loop with SR (`sample_reweight`) and FS (`feature_selection`); each member is a
LightGBM booster trained by `train_submodel` on the weighted samples and selected features.
`retrieve_loss_curve` reconstructs the per-tree per-sample loss with `model.predict(..., start_iteration,
num_iteration=1)`. `predict` averages the members over their `sub_features`. The base LightGBM params
are deliberately *identical to the winning LGBM baseline* — `learning_rate=0.2`, `num_leaves=210`,
`max_depth=8`, `lambda_l1=205.6999`, `lambda_l2=580.9768`, `colsample_bytree=0.8879`,
`subsample=0.8789` — so the finale isolates the *framework*: it is an ensemble of the exact tree that
already won, made drift-robust by SR and FS. The DoubleEnsemble defaults are `num_models=3`,
`enable_sr=True`, `enable_fs=True`, `alpha1=alpha2=1`, `bins_sr=10`, `bins_fs=5`, `decay=0.5`,
`sample_ratios=[0.8,0.7,0.6,0.5,0.4]`, `sub_weights=[1,1,1]`, `epochs=28` — crucially *short* members
(28 rounds, not the ~1000 a single GBDT used), because the ensemble, not boosting depth, does the
variance reduction and short members resist memorizing the noise. The full module is in the answer,
verified line-by-line against qlib's `qlib/contrib/model/double_ensemble.py`.

The bar this has to clear is LGBM's real numbers, and my expectations are falsifiable against them. The
aggregate gmean must beat LGBM's ≈0.571, or the framework added nothing. The mechanism predicts *where*
the gain comes from: on the long `csi300` window I expect the ensemble to lift IC/rank IC modestly
above the single tree (0.0457 / 0.0569) and, more importantly, to hold or improve the already-shallow
−0.0579 max drawdown — the SR down-weighting of noise and FS feature diversity should make the
portfolio at least as steady, not less. On `csi300_recent`, the drift-decisive regime, I expect the
clearest separation from the *adaptive* models (whose 0.48 info ratio it should crush as LGBM already
did at 1.10) and a hold-or-improve over LGBM's annualized return 0.0807 — the FS tail of dormant
factors is precisely the exposure a far regime rewards. On `csi300_shifted`, the near window where
AdaRNN's alignment beat the single tree, the ensemble is my chance to close that gap: I expect it to
push the tree's 1.30 info ratio up toward or past AdaRNN's 1.66 by reweighting toward the learnable
samples that the mild shift makes informative. The honest risk, which the numbers would expose, is
over-thinning: with only three short members each on a sampled feature subset, the ensemble could
*lose* a little IC on the long window relative to one deep tree if the per-member feature subsets drop
too much signal — if `csi300` IC comes in *below* 0.0457, that is the tell that the FS ratios are too
aggressive for a 158-factor set on this sample size. But the framework is built from the exact tree
that won, only made deliberate about samples and features, so the expectation is a uniform-or-better
lift across all three regimes with the drawdown held — an ensemble of the strongest control, turned
drift-robust on the two axes the single tree left uniform. That is the endpoint I would put past the
strongest baseline, and the `csi300` IC and the `csi300_recent` annualized return are the two numbers I
would check first to confirm or refute it.
