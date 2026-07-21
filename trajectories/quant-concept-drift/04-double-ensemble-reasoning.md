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

The finale's job is asymmetric, and the numbers say which way. The far-regime gain is already banked
and large — recent info ratio 1.10 against the adaptive pair's ~0.48, recent return 0.0807 against
their ~0.034 — so the finale is not fixing a broken far regime but making sure it gives none of that
back. The one place the single tree lost is `csi300_shifted`: AdaRNN's alignment posted 1.66 there
against the tree's 1.30, a 0.36 gap, so the tree runs at 78% of the alignment model's near-window info
ratio. That is the only cell where an adaptive idea beat the control, and it is precisely the
mild-shift window where reweighting toward newly-informative samples should pay. So: hold the
far-window and drawdown wins, recover the ~0.36 the tree left on the table where the shift is mild.

So the finale is to wrap the LightGBM base learner in an ensemble that is deliberate about two things a
plain bagged or boosted GBDT leaves uniform: which *samples* each new member focuses on, and which
*features* it may use. The framework trains members sequentially and, after each member, runs two
modules driven by the current ensemble's per-sample loss. Let me derive both against the failure I
just measured.

The obvious cheaper wrappers deserve a hearing first, so I am sure the two modules earn their
complexity. The simplest is uniform bagging: fit several LightGBMs on bootstrap resamples and average.
A bootstrap draws each member's rows uniformly with replacement, so each sees the expected
`1 − 1/e ≈ 63.2%` unique samples and *all* 158 features; averaging decorrelated members reduces
variance around the base tree's prediction. But the base tree is deterministic and already steady — its
seed-to-seed variance is literally zero in the feedback — so the variance bagging removes is not the
problem I have. And uniform bagging touches neither axis that matters: every member still uses all
features, so nothing sheds a regime-stale factor, and every member weights all samples the same, so
nothing down-weights the unlearnable noise. It would tighten a variance that is already tight and leave
the two drift pathologies untouched. The second cheap move is to simply boost harder — more rounds on
the single tree — but I argued above that on return residuals extra rounds chase noise, the wrong sign;
the tree already needs enormous leaf penalties to survive the ~1000 rounds it takes, and piling on more
would deepen the memorization, not the signal. So neither cheap wrapper acts on samples or features
deliberately, and deliberate action on exactly those two axes is the whole point. That justifies the
two-module framework, and now I derive each module against the failure I measured.

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

At `epochs=28` rounds per member the "first 10%" and "last 10%" of the loss curve each average three
snapshots, so `h2 = rank(l_end/l_start)` is small when the loss collapsed early and large when it
stalled. The binned weight `1/(decay^k·h_avg + 0.1)` with `decay=0.5` leans hard toward the learnable
bins on the first member — a well-behaved bin outweighs a stalled one several-fold — and flattens
toward uniform as `k` grows, so the early members specialize on the learnable core and the later ones
re-broaden rather than overcommit to one slice. The `+0.1` floor keeps even the worst bin's weight
finite, so no sample is ever fully dropped.

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

Sampling the five reliance bins at `[0.8,0.7,0.6,0.5,0.4]` keeps about `158·0.6 ≈ 95` factors per
member, the most-reliant bin keeping most of its factors and the least-reliant still keeping a tail —
never zero, so every member carries a dozen-plus currently-dormant factors as optionality against a
regime turn. The reliance itself is standardized, `g = mean(Δloss)/std(Δloss)`, so a factor's rank
depends not on the raw scale of the loss increase but on how *reliably* shuffling it hurts —
consistency of reliance, not peak reliance, is what should survive a regime change, and that is the
right thing to bin on. So each member is a tree over a different ~95 raw factors, the weak tail
deliberately never emptied — the concrete instrument for the "keep dormant factors alive" idea.

The two modules alternate off the current ensemble's loss: train member `k`, evaluate the
ensemble-so-far on the training data to get the loss values and member `k`'s loss curve, run SR to set
member `k+1`'s sample weights and FS to set its feature subset, train member `k+1`, repeat. The last
member needs neither (nothing follows it). At inference the whole SR/FS apparatus is gone — `predict`
just runs each member on its own stored feature subset and averages, weighted by `sub_weights`
(uniform). So, like the adaptive models, all the adaptation is training-time; unlike them, the
inference is a plain tree-ensemble average with no encoder to drift.

That training-time-only structure is exactly why I expect it to hold the far-window win. The encoders
failed *at inference*: TRA's router and AdaRNN's GRU both carry a learned representation into test
time, so when the 2019–2020 marginals move the very transform producing the prediction is evaluated on
inputs it was never shaped for. Here the SR/FS apparatus exists only during `fit`; by test time each
member is a frozen tree over scale-invariant threshold splits, averaged uniformly, with no
representation for drift to corrupt. The drift-robustness is bought in how the members are trained and
spent through an inference path with no moving parts.

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
variance reduction and short members resist memorizing the noise. The full module is in the answer.

The `epochs=28` choice is the one that most changes the character of the base learner, so let me sit
with the arithmetic. The winning single tree ran up to 1000 boosting rounds (early-stopped on the
2015–2016 validation), so it was a *deep* additive model whose late rounds needed the enormous
`λ1`/`λ2` just to not memorize residuals. Here each member is capped at 28 rounds, and there are three
of them: `3·28 = 84` trees total against the single tree's several hundred. That is a deliberate
inversion of where the model's power comes from. A 28-round LightGBM under the same heavy penalties is a
*shallow, high-bias, low-variance* learner — it captures the first-order cross-sectional structure and
stops well before the rounds where a deep booster starts chasing noise. Stacking three such shallow
members, each on its own ~95-factor subset and SR-reweighted samples, moves the variance reduction from
"1000 regularized rounds inside one tree" to "three short trees averaged," which is the healthier place
to spend it on a low-SNR panel: bias stays controlled by the penalties, variance is killed by the
ensemble average rather than by ever-heavier shrinkage on ever-deeper trees. It also means the FS
feature diversity actually bites — a 28-round tree cannot route around a dropped factor through 900
later rounds the way a deep one could, so forcing each short member onto a different subset genuinely
diversifies what the ensemble attends to.

`num_models=3` with uniform `sub_weights=[1,1,1]` is the modest end of the design, and I want to be
honest that it caps the upside. If the three short members were independent, averaging would cut their
variance to about a third; they are not independent — they share the base learner, the bulk of the
factor set (each keeps ~60%), and most of the sample weight after the `decay` annealing flattens it —
so the realized variance reduction is well short of `1/3`, more a gentle smoothing than a dramatic one.
Three is genuinely the smallest count at which "ensemble" means anything, and the uniform weights mean
I am not even tilting toward the members that saw the better feature subsets. That framing sets my
expectation: this finale should deliver a *measured* lift over the single tree — enough to clear the
≈0.571 aggregate and close the shifted-window gap — not a step change, because with three coupled short
members the framework's whole contribution is the deliberate sample/feature shaping, not brute
ensemble size. If it needed many more members to separate from the single tree, that would itself be a
finding about how little headroom the two axes have on this panel.

One interaction between the two modules deserves a wary look, since they are driven by the same loss
and could fight. FS drops a factor from member `k+1`; any sample that leaned on it now predicts worse,
and SR — which up-weights poorly-handled samples through its `h1` term — pushes member `k+1` toward
exactly those newly-starved samples using the factors it *does* have. Read pessimistically that is a
feedback loop that could thrash. But two things damp it: SR blends `h1` with the curve-shape `h2`, so a
sample made hard purely by a dropped-but-informative factor (its curve still collapses early where that
factor is present in other members) scores differently from one that is hard because it is noise; and
the `decay=0.5` annealing weakens the reweighting toward uniform by the third member, just as the
accumulated feature subsets start covering the factor space between them. So the modules are coupled
but damped, not runaway.

The bar is LGBM's real numbers, and the expectation is falsifiable against them. The aggregate must
beat LGBM's ≈0.571 or the framework added nothing, and the mechanism predicts where the gain comes
from. On the long `csi300` window I expect a modest lift in IC/rank IC over the single tree (0.0457 /
0.0569) and, more importantly, the already-shallow −0.0579 max drawdown held or improved, since SR
down-weighting of noise and FS diversity should make the portfolio no less steady. On `csi300_recent`
I expect the far-window win held — LGBM already crushed the adaptive pair's 0.48 there at 1.10 — with
the FS tail of dormant factors precisely the exposure a distant regime rewards. On `csi300_shifted`,
the one window where AdaRNN's alignment beat the tree, the ensemble is my chance to close the
1.30-to-1.66 gap by reweighting toward the samples the mild shift makes informative. The honest risk is
over-thinning: with only three short members each on a sampled feature subset, the ensemble could
*lose* a little IC on the long window if the per-member subsets drop too much signal — a `csi300` IC
below 0.0457 would be the tell that the FS ratios are too aggressive for a 158-factor set on this
sample size. But built from the exact tree that won and only made deliberate about samples and
features, the expectation is a uniform-or-better lift across all three regimes with the drawdown held.
The `csi300` IC and the `csi300_recent` annualized return are the two numbers I would check first.
