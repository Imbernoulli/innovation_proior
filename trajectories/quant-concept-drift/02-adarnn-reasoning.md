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

Now the model, derived against this scaffold — and the scaffold immediately changes the generic
AdaRNN story in a way I have to honor. The generic idea splits the training stream into `K` periods
chosen to be maximally *dissimilar* (maximum-entropy / worst-case diversity, found by a greedy
search over candidate cut points), then aligns the periods' hidden-state trajectories at every
timestep with a learned, boosting-updated per-state importance weight. Three pieces: period
discovery, per-state distribution matching, boosting weights. The task fill keeps the *machinery* —
the gated AdaRNN network, the boosting weight update, the full `TransferLoss` family — but pins two
hyperparameters that simplify it sharply, and I must derive the model that the harness actually
runs, not the paper's. First, `len_seq = 1`. Alpha158 gives 158 *flat* factors per (stock, day);
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
reweighting, that carries this baseline on this task. Training is Adam `lr=1e-3`, `n_epochs=200`,
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
