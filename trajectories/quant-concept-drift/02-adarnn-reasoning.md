TRA landed where a real but heavy routing model lands, and the *shape* of its three-regime result
is the thing I have to read before I pick the next rung. On the long-horizon `csi300` window it gets
IC 0.0420, rank IC 0.0504, information ratio 1.01; on `csi300_shifted` it does best — IC 0.0513,
rank IC 0.0546, info ratio 1.59; and on `csi300_recent` it collapses — IC 0.0237, rank IC 0.0397,
info ratio 0.48, less than a third of its shifted info ratio. That gradient is the diagnosis. TRA is
identical across the three seeds (its backbone is seeded deterministically), so this is not seed
noise; it is structural. The recent window (2019–2020) is the regime furthest from the 2008–2014
training period, and it is exactly there that TRA is weakest. Routing did what I expected — it
separated in-distribution trading patterns and beat the averaged linear default — but the router's
selector is learned from cached prediction-error history on the *training* distribution, and when
the test regime drifts far from training, the very signal the router relies on (which predictor has
been doing well lately) is computed on a distribution that no longer holds, so it routes the
shifted-future samples to the wrong heads. TRA spent its capacity on *which pattern is active now*
and almost none on *what survives when the distribution moves*. The 0.48 info ratio on
`csi300_recent` is that gap made numeric. So the next rung should not route harder; it should
explicitly force the model to learn what is *invariant* across time, so a drifted test window
inherits a relation that was trained to be stable rather than a router trained to a stale signal.

Let me read TRA's table more slowly, because the *rate* at which each metric decays across the three
windows tells me which kind of structure survives drift and which does not. Line up the information
ratios: 1.5869 on shifted, 1.0081 on `csi300`, 0.4798 on recent. Normalizing to the near window,
`csi300` keeps 64% of the shifted IR and recent keeps only 30%. Now line up the plain IC: 0.0513,
0.0420, 0.0237 — recent keeps 46% of shifted. The portfolio metric collapses almost twice as fast as
the signal metric (30% vs 46%), and that gap is not noise: the top-50/drop-5 backtest is a hard
threshold on the *top* of the cross-section, so a roughly uniform shrinkage of IC translates into a
super-linear loss of the tail separation the strategy actually trades. That already tells me a method
that only lifts average IC will underdeliver in the backtest; I need one that protects the *ordering*
at the top of the book, not the mean correlation. And the ordering metric backs this up: rank IC runs
0.0546, 0.0504, 0.0397 — recent keeps 73% of shifted, far more than plain IC's 46%. So the *monotone*
content of TRA's signal is markedly more drift-stable than its linear content; the drift is corrupting
the calibrated magnitude of the prediction more than its rank. That is a direct hint toward a method
that aligns *distributions* rather than trusting the raw regression scale — the direction the
invariance framing points anyway.

One more decomposition before I commit. ICIR — the IC divided by its day-to-day standard deviation —
runs 0.4283, 0.3355, 0.1923 across the windows; recent keeps only 45% of shifted, essentially the
same decay as plain IC. So on the far regime TRA's daily IC is not just smaller on average, it is
proportionally *noisier* day to day: the sign of the edge flickers. A single deterministic encoder
trained on 2008–2014 produces a signal whose very stability erodes as the test window walks away from
training. This is the numeric face of "the router reads a stale error-history signal": both the level
and the temporal consistency of the edge decay together, and they decay hardest on the window furthest
in time.

Let me get the structure of that right, because naming it picks the tool. Write the joint law
`P(x, y)`. Over the training stream 2008–2014 the marginal `P(x)` of the factors drifts hard with
the regime, but the economic regularity turning factors into returns — `P(y|x)` — is far more
stable. The thing worth carrying to a shifted future is `P(y|x)`; the thing that betrays me is
`P(x)`. That is temporal covariate shift: `P_train(x) ≠ P_test(x)` while `P_train(y|x) =
P_test(y|x)`, and the stream is not one shift but a sequence of shifts at unknown times. The
classic covariate-shift fix — reweight by the test/train density ratio — is dead twice over: the
test is a genuinely unseen future window with no density to estimate, and the posed fix has no
notion of *when* inside a stream the distribution turns over. So the right move is to make the model
*extract what is invariant across time* by aligning the periods' representations, which is exactly
the failure TRA exposed: routing aligns nothing across regimes, it only partitions within one.

Before I name a mechanism I want to be honest about the two or three moves actually on the table and
kill the tempting ones with arithmetic rather than taste. The first is to route harder — more states,
a sharper selector. I reject it on the numbers I just read: TRA is deterministic and still collapses
on recent, so the failure is not selector variance I could average away; it is that the selector's
input, cached prediction-error history, is itself computed on a stale distribution. More states only
partition the training distribution more finely and give the drifted test *more* ways to be routed
wrong. The second is the textbook covariate-shift fix — reweight each training sample by the
test/train density ratio `w(x) = p_test(x)/p_train(x)`. It dies twice here: the test is a genuinely
unseen future window, so `p_test` has no sample to estimate from at training time, and the ratio has
no notion of *when* within the 2008–2014 stream the distribution turned, so it cannot express a
*sequence* of shifts at unknown change points — it treats the stream as one static source.

The third tempting move is invariant risk minimization: partition the stream into environments and
penalize the variance of the per-environment optimal predictor, so the learned representation admits
one head that is simultaneously optimal everywhere. It is the *right* instinct — invariance across
time is exactly what I want — but it fits this scaffold badly, and I can see why concretely. IRM needs
explicit environment labels and a penalty weight I would have to sweep, and its gradient-norm penalty
is famously fragile; more damningly, on a view where each sample is a single timestep `[N, 1, 158]`
the representation the penalty acts on is one hidden vector, and with only a two-way chronological
split the environment count is two — the smallest number at which a variance penalty is estimable at
all. The penalty would be balancing two noisy per-half gradients through a hand-tuned coefficient on a
panel whose IC is about 0.05. That is a lot of fragile machinery for the same two-environment
information I can use more directly. The direct use is: do not penalize the *gradient* of a shared
predictor across halves, just push the halves' *representations* together and let a shared head sit on
top — a distribution-matching loss between periods. Same invariance goal, a smoother objective, and it
drops straight onto a recurrent encoder. That is the family I take.

Now the model, derived against this scaffold — and the scaffold immediately changes the generic
AdaRNN story in a way I have to honor. The generic idea splits the training stream into `K` periods
chosen to be maximally *dissimilar* (maximum-entropy / worst-case diversity, found by a greedy
search over candidate cut points), then aligns the periods' hidden-state trajectories at every
timestep with a learned, boosting-updated per-state importance weight. Three pieces: period
discovery, per-state distribution matching, boosting weights. The task fill keeps the *machinery* —
the gated AdaRNN network, the boosting weight update, the full `TransferLoss` family — but pins two
hyperparameters that simplify it sharply, and I must derive the model that the harness actually
runs, not the generic one. First, `len_seq = 1`. Alpha158 gives 158 *flat* factors per (stock, day);
the fill's `data_loader` does `.unsqueeze(1)` to make `[N, 1, 158]`, a single-timestep sequence. So
the GRU sees one step, the hidden trajectory has length one, and the "match the distribution at
*every* hidden state" idea collapses to matching at the single state. The whole argument that an
endpoint-only match wastes the recurrent trajectory still motivates the *architecture*, but on this
task there is exactly one state, so per-state weighting and the local matching window `len_win`
degenerate — there is nothing to weight across. Second, the periods are not discovered by a
diversity search at all: `n_splits = 2`, and `fit` literally does `np.array_split(days, 2)` — two
equal consecutive halves of the training days, 2008–2011 and 2012–2014. No greedy max-distance
search, no `K` sweep. So the temporal-distribution-characterization step is replaced by a fixed
two-way chronological split. That is the honest description of this baseline: align the
representations of the first and second halves of training so the GRU learns a factor→return map
that is invariant between the two halves, in the hope it then transfers to a third (test) half.

Given those pins, here is the model. The backbone is a stack of single-layer GRUs (`num_layers=2`,
hidden 64) so each layer exposes its own hidden state; no bottleneck (`use_bottleneck=False`), a
linear head on the last state to the scalar return prediction. For a pair of periods (here the only
pair, halves 0 and 1) I run both through the GRU, take the per-layer hidden state, and add a
distribution-matching loss between the two halves' representations to the prediction loss. The
matching distance is `cosine` here (`loss_type="cosine"`, `d = 1 - cos(mean(h_s), mean(h_t))`) —
the cheapest of the `TransferLoss` family, the sane default on the large CSI300 panel; the family
also carries linear/RBF MMD, CORAL, and the gradient-reversal adversarial distance, all available
but unused at this setting. The objective per layer is `L_pred + λ·L_match`, with `dw = 0.5` as the
matching trade-off `λ` and `L_pred` the MSE on each half's predictions. Because the relationship I
want is the one *shared* between the halves, pushing their hidden representations together forces the
GRU toward the invariant conditional.

Let me size this network against the sample it must learn from, because the drift argument only
matters if the model is not already starved. The stacked GRU has two single-layer cells: the first
maps 158 inputs to 64 hidden, costing `3·(64·158 + 64·64 + 2·64) = 3·14336 ≈ 43.0k` parameters across
its three gates; the second maps 64 to 64, `3·(64·64 + 64·64 + 2·64) ≈ 25.0k`; the linear head adds
65. Call it about 68k parameters. The CSI300 training panel is roughly 300 names over ~1700 trading
days, order `5·10^5` samples, so the parameter-to-sample ratio is comfortable — the model is not
capacity-starved. What it is starved of is *signal per sample*: an IC around 0.05 is an R² near
0.0025, so roughly 99.7% of each label's variance is noise the encoder must not fit. That reframes the
whole exercise — the danger here is never underfitting the mean relation, it is the encoder spending
its ample capacity memorizing the noise, and the distribution-matching loss is as much a regularizer
that ties the two halves' representations together as it is a drift fix.

The matching distance itself is worth pinning down, because the fill's `cosine` reduces each half's
hidden batch to its *mean* before comparing — it aligns the first moment (the mean direction) of the
two halves' representations, not their full distribution. That is the coarsest member of the available
`TransferLoss` family (linear/RBF MMD would match higher moments, CORAL the covariance, the
gradient-reversal adversary the whole density), and it is the cheapest to compute on the large CSI300
batch. On a single-state, first-moment match between two chronological halves I am asking very little
of the alignment: keep the average hidden representation of 2008–2011 pointed the same way as
2012–2014. Cheap, stable, and — I suspect — too weak to reach a 2019–2020 regime that sits outside
both halves, which is precisely the tension my falsifiable prediction turns on.

The two-phase training is the load-bearing part the fill keeps. A chicken-and-egg sits between the
state-importance weights and the GRU parameters: the weights judge the importance of hidden states,
but early in training the states are meaningless, so the weights learn garbage and steer the GRU
wrong. So `pre_epoch = 40`: for the first 40 epochs the fill runs `forward_pre_train`, which produces
the prediction output and a *gate-weighted* matching loss (a small per-layer linear gate maps the
concatenated source/target states to the state weights), getting the representations sensible before
the boosting weights take over. After epoch 40 it switches to `forward_Boosting`, which computes the
prediction output, the matching loss weighted by the current weight matrix, and the per-state
distance matrix; then `update_weight_Boosting` ratchets up the weight on any state whose cross-half
distance *grew* from last epoch — `weight *= 1 + sigmoid(d_new − d_old)`, a multiplier in (1, 2) so
weights only ever increase on the stalling states and a single noisy epoch cannot blow them up,
re-L1-normalized to stay a convex weighting so `dw` alone controls strength. This is the boosting
instinct: lean into the worst-aligned state. With `len_seq=1` there is a single state per layer, so
the boosting update is doing little real work here — it is the period *alignment*, not the per-state
reweighting, that carries this baseline on this task.

It is worth tracing that boosting update once at this setting to be sure I am not fooling myself about
what it does. The weight vector for a layer has one entry per hidden state; with `len_seq = 1` that is
a length-one vector, say `w = [1]` after the L1 normalization the fill applies. At the next epoch the
update multiplies it by `1 + sigmoid(d_new − d_old)`, some number in `(1, 2)` — giving `w = [m]` with
`m ∈ (1, 2)` — and then re-L1-normalizes, `w/‖w‖_1 = [m]/m = [1]`. The single weight returns to 1
every epoch: the boosting reweighting is *exactly idempotent* here, a no-op by construction, for each
of the two layers independently. So whatever this baseline earns over a plain GRU comes entirely from
the matching loss `dw·L_match` pulling the two halves' single hidden states together — the celebrated
per-state boosting is inert at `len_seq = 1`, and I should read the result as a test of coarse
two-half representation alignment, nothing more.

The schedule's arithmetic reinforces which part does the work. Of the 200-epoch budget the first 40
(20%) run `forward_pre_train`, stabilizing the representations under a gate-weighted match before any
boosting weight engages; the remaining 160 run `forward_Boosting`. But I just showed the boosting
reweight is idempotent at one state, so those 160 epochs are, in effect, still just the matching loss
`0.5·L_match` plus the prediction MSE, now under a frozen uniform state weight. With `early_stop = 20`
on validation loss the run almost certainly halts before epoch 200 on this noisy panel, so the entire
trained model is: a two-layer GRU fit to predict returns from a single 158-vector, regularized by a
first-moment alignment of two chronological halves at matching strength 0.5. Stated that plainly, it
is a modest regularized regressor, and my expectation for its ceiling should be modest to match —
competitive with TRA precisely because both are, underneath their different adaptation stories, small
encoders extracting the same thin signal from the same panel. Training is Adam `lr=1e-3`,
`n_epochs=200`,
`early_stop=20` on validation loss, gradient value-clip at 3.0, `batch_size=800`, `dw=0.5`. The
`seed` is `None`, so unlike TRA this model is *not* deterministic across seeds — its weights
initialize differently each run, which I will need to read into its variance. At inference none of
the matching machinery runs: `predict` just unsqueezes the test features to `[N, 1, 158]` and runs
one GRU forward pass through `model.predict`, so prediction costs a vanilla recurrent net. The full
module — the GRU stack, the `TransferLoss` family, the gate and boosting weight update, the
two-phase `fit` over the two chronological halves, and the batched `infer` — is the fill of
`custom_model.py`; it is in the answer.

One more thing the scaffold changes relative to TRA that bears on what I expect: AdaRNN sees the
*full* 158-factor Alpha158 view (the fill edits only `custom_model.py`, leaving the default
`DatasetH`/Alpha158 processor block untouched), whereas TRA saw only the 20-column `FilterCol`
slice. So AdaRNN has more raw factor information and a cheaper, more direct mechanism (align two
halves, predict from one state) than TRA's heavy OT-distilled router. That cuts two ways and frames
my falsifiable expectations.

Let me make these predictions numeric enough that the next rung's feedback can actually falsify them.
"Near TRA on shifted" means I expect shifted IC in the 0.048–0.055 band around TRA's 0.0513 and
shifted info ratio at or a little above TRA's 1.5869 — the mild-shift window is where two-half
alignment should pay, since 2012–2014 is genuinely close to a 2016–2018 test. "Still weak on recent"
means recent info ratio should land in TRA's neighborhood of about 0.48, not the ~1.0 it would need to
count as rescued, because aligning two halves inside 2008–2014 encodes nothing about the shape of
2019–2020. And because `seed = None` makes this model non-deterministic, I expect the recent window —
the lowest-SNR of the three — to show the widest seed spread of any cell in the table; a worst seed
there flirting with zero info ratio would be the sharpest confirmation that alignment bought stability
in-distribution and nothing out of it. If instead recent info ratio jumps cleanly above 1.0 across all
three seeds, my whole "alignment cannot reach a regime outside both halves" reading is wrong and I
would have to credit two-half matching with real extrapolation.

I expect AdaRNN to be *roughly competitive with TRA on the aggregate, with a different regime
profile and more seed variance*. Concretely: on `csi300_shifted` it should be in TRA's
neighborhood or a touch better — its info ratio there should land high (TRA's was 1.59), because
the shifted window is closest to training and explicit two-half alignment should help where the
shift is mild. On `csi300_recent` — the window that killed TRA at info ratio 0.48 — alignment is the
right idea, but with only two coarse chronological halves and a single-timestep GRU, I do not expect
it to *rescue* the far regime; I expect it to stay weak there too, possibly weaker on some seeds,
because aligning 2008–2011 to 2012–2014 says nothing about a 2019–2020 regime that lies outside both
halves. The falsifiable claim against the prior numbers: if AdaRNN's `csi300_recent` info ratio
comes in around TRA's 0.48 or below — and especially if its *worst seed* on `csi300_recent` lands
near zero (an aligned-but-uninformed model on the farthest regime) — then explicit two-half
alignment, like routing, does not buy generalization to a genuinely distant future, and the lesson
is that *both* adaptive sequence models are spending their machinery on the wrong axis. The
sharper, more surprising possibility I should hold open: that on this low-SNR CSI300 panel a heavy
sequence model with elaborate adaptation does *not* beat what a strong, plain, full-feature learner
would do — that the adaptation overhead costs more (in fit-able signal on a thin sample) than the
drift-robustness it buys. If AdaRNN's three-regime aggregate sits at or below TRA's rather than
clearly above it, that points the next rung away from sequence-model adaptation entirely and toward
making the *strong non-adaptive control* robust by other means.
