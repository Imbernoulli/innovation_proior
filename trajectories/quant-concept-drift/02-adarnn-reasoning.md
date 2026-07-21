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

ICIR — IC over its day-to-day standard deviation — tells the same story: 0.4283, 0.3355, 0.1923,
recent keeping only 45% of shifted. So on the far regime the daily edge is not just smaller but
proportionally noisier, its sign flickering; both the level and the temporal consistency of the
signal decay together, and hardest on the window furthest in time.

Let me get the structure of that right, because naming it picks the tool. Write the joint law
`P(x, y)`. Over the training stream 2008–2014 the marginal `P(x)` of the factors drifts hard with
the regime, but the economic regularity turning factors into returns — `P(y|x)` — is far more
stable. The thing worth carrying to a shifted future is `P(y|x)`; the thing that betrays me is
`P(x)`. That is temporal covariate shift: `P_train(x) ≠ P_test(x)` while `P_train(y|x) =
P_test(y|x)`, and the stream is not one shift but a sequence of shifts at unknown times. So the right
move is to make the model *extract what is invariant across time* by aligning the periods'
representations — exactly the failure TRA exposed: routing aligns nothing across regimes, it only
partitions within one.

Before I name a mechanism, the tempting alternatives deserve to be ruled out. The first is to route
harder — more states, a sharper selector. The numbers reject it: TRA is deterministic and still
collapses on recent, so the failure is not selector variance I could average away; it is that the
selector's input, cached prediction-error history, is itself computed on a stale distribution. More
states only partition the training distribution more finely and give the drifted test *more* ways to
be routed wrong. The second is the textbook covariate-shift fix — reweight each training sample by the
density ratio `w(x) = p_test(x)/p_train(x)`. It dies twice: the test is a genuinely unseen future
window, so `p_test` has no sample to estimate from at training time, and the ratio has no notion of
*when* within the 2008–2014 stream the distribution turned, so it cannot express a *sequence* of
shifts at unknown change points — it treats the stream as one static source.

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

The network is not capacity-starved: the stacked GRU runs on the order of 68k parameters against a
training panel of roughly 300 names over ~1700 trading days, order `5·10^5` samples. What it is
starved of is *signal per sample* — an IC around 0.05 is an R² near 0.0025, so roughly 99.7% of each
label's variance is noise the encoder must not fit. That reframes the whole exercise: the danger here
is never underfitting the mean relation, it is the encoder spending its ample capacity memorizing
noise, and the distribution-matching loss is as much a regularizer tying the two halves'
representations together as it is a drift fix.

The `cosine` match is worth pinning down: it reduces each half's hidden batch to its *mean* before
comparing, so it aligns only the first moment — the mean direction — of the two halves'
representations, the coarsest and cheapest choice in the family. On a single-state, first-moment
match between two chronological halves I am asking very little of the alignment: keep the average
hidden representation of 2008–2011 pointed the same way as 2012–2014. Cheap, stable, and — I suspect
— too weak to reach a 2019–2020 regime that sits outside both halves, which is precisely the tension
my falsifiable prediction turns on.

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
instinct: lean into the worst-aligned state.

But at `len_seq = 1` that update is a no-op. The weight vector has one entry per hidden state, so it
is length-one, `w = [1]` after L1 normalization. The next epoch multiplies by `1 + sigmoid(d_new −
d_old)`, some `m ∈ (1, 2)`, giving `w = [m]`, then re-L1-normalizes: `[m]/m = [1]`. The single weight
returns to 1 every epoch — the boosting reweighting is exactly idempotent, a no-op by construction,
for each layer independently. So whatever this baseline earns over a plain GRU comes entirely from the
matching loss `dw·L_match` pulling the two halves' single hidden states together; I should read the
result as a test of coarse two-half representation alignment, nothing more.

So of the 200-epoch budget the first 40 run `forward_pre_train`, stabilizing the representations under
a gate-weighted match before the boosting phase engages; the remaining epochs run `forward_Boosting`,
whose per-state reweight is inert here — leaving, in effect, the matching loss `0.5·L_match` plus
prediction MSE throughout. With `early_stop = 20` on validation loss the run halts well before epoch
200 on this noisy panel. So the trained model is a two-layer GRU fit to predict returns from a single
158-vector, regularized by a first-moment alignment of two chronological halves at strength 0.5 — a
modest regularized regressor, and my expectation for its ceiling is modest to match: competitive with
TRA precisely because both are, underneath their different adaptation stories, small encoders
extracting the same thin signal from the same panel. Training is Adam `lr=1e-3`, `n_epochs=200`,
`early_stop=20` on validation loss, gradient value-clip 3.0, `batch_size=800`, `dw=0.5`. The `seed` is
`None`, so unlike TRA this model is *not* deterministic across seeds — its weights initialize
differently each run, which I will read into its variance. At inference none of the matching machinery
runs: `predict` unsqueezes the test features to `[N, 1, 158]` and runs one GRU forward pass, a vanilla
recurrent net. The full module is in the answer.

One more thing the scaffold changes relative to TRA that bears on what I expect: AdaRNN sees the
*full* 158-factor Alpha158 view (the fill edits only `custom_model.py`, leaving the default
`DatasetH`/Alpha158 processor block untouched), whereas TRA saw only the 20-column `FilterCol`
slice. So AdaRNN has more raw factor information and a cheaper, more direct mechanism (align two
halves, predict from one state) than TRA's heavy OT-distilled router. That cuts two ways and frames
my falsifiable expectations.

So my expectation, falsifiable against the next rung's numbers: AdaRNN should be *roughly competitive
with TRA on the aggregate, with a different regime profile and more seed variance*. On `csi300_shifted`
— the mild-shift window closest to training, where explicit two-half alignment should pay — I expect
its strongest showing, in the neighborhood of TRA's 1.59 info ratio or a touch above. On
`csi300_recent`, the window that killed TRA, alignment is the right idea but two coarse chronological
halves and a single-timestep GRU encode nothing about a 2019–2020 regime lying outside both, so I do
not expect a rescue: it should stay in the weak zone the encoders occupy rather than climbing toward
the ~1.0 that would count as fixed. And because `seed = None` makes this model non-deterministic, I
expect more run-to-run variance than deterministic TRA showed, most of it landing on the low-SNR
recent window; a recent window that swung wildly across seeds would confirm alignment bought stability
in-distribution and little out of it. If instead recent climbs cleanly and consistently above the
collapse zone, my "alignment cannot reach a regime outside both halves" reading is wrong. The sharper
possibility I hold open: that on this low-SNR panel a heavy sequence model with elaborate adaptation
does *not* beat a strong, plain, full-feature learner — that the adaptation overhead costs more, in
fit-able signal on a thin sample, than the drift-robustness it buys. If AdaRNN's aggregate sits at or
below TRA's rather than clearly above, that points the next rung away from sequence-model adaptation
entirely and toward making the *strong non-adaptive control* robust by other means.
