The scaffold hands me a Ridge regression over Alpha158 and a frozen backtest, and the research
question is concept drift — so before I write a line I have to decide what the *first serious* rung
should be, the one whose failure mode will teach me what the next rung must fix. Two things about how
this task is scored shape that decision, and I want them in front of me before I pick a model. The
aggregate is a geometric mean across three temporal regimes of each regime's equally-weighted,
sigmoid-mapped seven-metric mean, and both of those operations carry meaning. The geometric mean
over regimes punishes a collapse multiplicatively — a method that wins two windows and dies on the
third is dragged down toward its worst regime, not rewarded for its best — so what the score really
buys is *consistency across when I am tested*, not a peak on the friendly window. And the per-metric
logistic squashes each raw number: an IC of 0.04 maps to about 0.51, a rank IC of 0.05 to about 0.51,
both barely off the 0.5 floor, because the sigmoid is nearly flat near zero (its slope there is only
1/4). The metrics that actually carry spread through that map are the risk-adjusted ones — ICIR,
rank-ICIR, and above all the information ratio, which lives around 1 and maps near 0.73, a full 0.22
above the floor. So the objective is quietly telling me two things: raw predictive correlation is
almost free to move, and what genuinely moves the score is *stable, risk-adjusted* behavior held
across every regime. That is a drift objective wearing a scoreboard.

Let me pin down how much the geometric mean over regimes actually bites, because it decides whether
"consistency" is a slogan or a real budget. Say a method reaches a per-regime seven-metric mean of
0.58 on two windows but only 0.54 on the third. The arithmetic mean would be 0.567; the geometric
mean is `(0.58·0.58·0.54)^{1/3} = 0.5665`, and the gap to a flat method scoring 0.57 everywhere
(gmean 0.5700) is a real 0.0035, most of it manufactured by the single weak regime rather than by any
average. Push the weak regime down to 0.50 and the gmean drops to `(0.58·0.58·0.50)^{1/3} = 0.5520`,
losing another 0.014 for a four-point dip in one window. The lesson is arithmetic, not rhetorical: on
this scoreboard the cheapest points are bought by *lifting the worst regime*, and the most expensive
mistake is to let any one window collapse. That is precisely the failure mode concept drift produces
— a model that is fine on windows near its training distribution and falls apart on the one that
drifted — so the first rung should go straight at instability across time rather than chase a higher
in-sample IC on the friendly window.

The default Ridge is a single linear relation `p(y|x)` fit by average squared loss over 2008–2014,
and that is exactly the assumption I distrust: one fixed joint `P`, one mapping from factors to
returns. The market is not one `P`. Two of the best-documented cross-sectional effects point in
opposite directions at the same time — momentum (winners keep winning) and reversal (losers rebound)
— and which one dominates rotates with the regime. Let me make the damage concrete instead of
asserting it. Suppose the training stream is an equal blend of two sub-regimes: in regime A the
truth is `y = +β·r + ε` (momentum), in regime B it is `y = −β·r + ε` (reversal), where `r` is some
past-return factor standardized to unit variance and equal mass falls in each. A single least-squares
coefficient is `θ = Cov(r, y)/Var(r)`, and the covariance pools the two: `Cov(r,y) = ½·(+β·Var(r)) +
½·(−β·Var(r)) = 0`. The one coefficient the pooled objective can fit is *zero* on the very factor
that is perfectly predictive inside each regime — it forecasts nothing, IC ≈ 0, precisely because it
is forced to average two contradictory truths into their midpoint. A bigger single model does not
escape this: a deeper network with one output head still fits one conditional and still cancels the
two signs wherever they overlap in `x`. So the first rung must not be a bigger single model; it
should be a model that *holds several distinct relations at once and decides, per sample, which one
applies*. That is a routing model over a sequence encoder, and it is the first thing I will fill into
`CustomModel`. Let me derive it against this exact scaffold, because the scaffold constrains it in
ways the generic idea does not.

The data really is `P = Σ_k ν_k P_k`, a mixture of trading patterns, each with its own `p_k(y|x)`
and its own time-varying share `ν_k`. If I knew each sample's pattern `k` this would be plain
multi-domain learning. I do not — the patterns are latent, and worse, even the tempting cheat of
using the *label* to assign each training sample to its best-fitting pattern dies at test time, where
there are no labels and the test window is the whole point. So the constraint is sharp: I need a model
that (a) carries several relations, (b) routes each sample to one of them, and (c) makes that routing
decision from signals available *without the current label* — the sample's latent representation and
whatever past prediction-error state is already cached. Property (c) is the one that kills the naive
solutions, and I will keep coming back to it.

Start with "one model, several behaviors." A mixture-of-experts is the right shape: `K` experts and a
gate. But what should the experts be here? `K` full backbones would multiply parameters by `K` and,
on a noisy, smallish CSI300 panel, overfit hard — so let me count the cost before I commit. The
backbone the scaffold pins is an attentive LSTM: input width 20, hidden 64, two layers. An LSTM
layer carries roughly `4·(hidden·(in+hidden)+hidden)` parameters; layer one is `4·(64·(20+64)+64) =
4·5440 = 21,760`, layer two is `4·(64·128+64) = 33,024`, and the attention head (a `W` of `64×64` and
a context vector `u`) adds about `4,200`, so the shared encoder is ≈59k parameters. A single linear
predictor from the 128-wide latent is `129` parameters. Replicating only that head `K` times costs
`K·129` — negligible. Replicating the *whole encoder* `K` times costs `K·59k`; at `K=3` that is
≈177k versus the ≈64k the shared-encoder design uses once the router (≈5k, computed below) is added.
Roughly 2.7× more parameters to fit from the same 2008–2014 sample, for capacity I do not need — the
patterns differ in the *relation* far more than in how to *encode* a 60-day window. So I share one
encoder `ψ` and replicate only the cheap part into `K = num_states` linear predictors: `ŷ_ik = θ_k ∘
ψ(x_i)`, `ψ` shared, `θ_k` a linear head. Near-zero extra parameters, and a free safety net I can
actually verify rather than assert: at `num_states = 1` the router's softmax over one logit is 1, the
one-hot selection is `[1]`, the transport with a single column marginal `ν_1 = 1` forces every
sample's mass onto that one head, and the distillation term `Σ_k P_ik log q_ik = 1·log 1 = 0`, so the
whole objective collapses to plain task loss and the model *is* a single attentive LSTM. Routing can
only add capacity, never subtract it. In this task `num_states = 3` (the workflow's `MTSDatasetH` is
configured with `num_states: 3`) — enough heads to separate the main regimes, few enough that each
head still gets a third of the data rather than a starving sliver.

Now the encoder, and here the scaffold pins the details in ways I have to honor rather than default
around. The reference fill uses an **attentive LSTM** — `rnn_arch="LSTM"`, two layers, hidden 64,
`use_attn=True` — not the GRU the generic routing idea reaches for. The encoder projects the input
only when it must compress, and here it does not, since the input width (20) is below the hidden
width (64); it runs the LSTM, takes the mean of the last hidden state across layers, and concatenates
that with an attention-pooled vector `att = Σ_t softmax_t(u^T tanh(W·h_t))·h_t`. Two 64-wide vectors
concatenated make the latent fed downstream `2·64 = 128` wide — that is the shape the predictors and
the router both consume, and I will keep it straight because the router's `fc` depends on it. The
input width is the next thing the scaffold fixes and it is emphatically *not* all of Alpha158: the
editable processor block runs a **`FilterCol`** that keeps exactly 20 named columns (RESI5, WVMA5,
RSQR5, KLEN, …, KLOW) before `RobustZScoreNorm` and `Fillna`, so `input_size = 20`. That matters —
the routing model does not see the full 158-factor view the linear default sees; it sees a curated
20-factor slice over a 60-day sequence. The sequence itself comes from `MTSDatasetH` with `seq_len:
60`, `batch_size: 1024`, `memory_mode: "sample"`, `drop_last: true` — the editable dataset block is
swapped from `DatasetH` to this multi-timeseries sampler so each sample is a (stock, time) window
carrying its own cached state, which is precisely the object the router needs.

The router is the heart of it, and property (c) governs its inputs strictly. It must run at test, so
it can only touch label-free signals. The first is the latent `h_i = ψ(x_i)` — I route on `h` rather
than raw factors because `h` is already tuned toward prediction. But a single window's latent
underdetermines the regime: the difference between "we are in a momentum regime" and "we are in a
reversal regime" is not cleanly written into one stock's 60-day vector. I need a second signal, and
the honest one comes from how a real investor actually rotates strategies — not by deducing the macro
regime from today's prices, but by watching *how their strategies have been performing lately*. The
analogue is each predictor's *recent prediction error*, which is exactly the cached `state` the
`MTSDatasetH` carries. So the router reads the per-sample history of the `K` predictors' losses,
encoded by a small LSTM (`tra_config` hidden 32, one layer) whose last hidden state is concatenated
with the 128-dim latent (`src_info="LR_TPE"`: latent representation plus temporal prediction errors)
and mapped by a linear `fc` to `K` logits. Counting this: the TPE-LSTM is `4·(32·(3+32)+32) ≈ 4,608`
parameters and the `fc` from `160` to `3` adds `483`, so the router is ≈5.1k on top of the ≈59k
encoder — the number I used above. One causality subtlety the sampler must respect or the whole thing
leaks: the most recent usable error is from *before* the label horizon, never the current step, and
batches flow forward in time and are never shuffled across the time axis, or the router would be
reading its own future.

How does the router *use* the logits? A soft `softmax` blend would quietly re-create the single-model
problem: every predictor would receive gradient from every sample and all three would drift back
toward the same averaged relation — the very averaging I proved kills the momentum/reversal signal.
To force specialization the selection has to be *discrete*: one predictor per sample, so a head can
swing its coefficients all the way to reversal without being dragged back by momentum samples.
Discrete selection is `argmax`, which has no gradient. The escape is the **Gumbel-softmax**: draw
`one_hot(argmax_k(a_ik + g_ik))` with `g` i.i.d. Gumbel noise, replace the argmax with a temperature
softmax at `tau=1.0`, and use the straight-through form — a hard one-hot `choice` in the forward pass
carrying the soft gradient on the backward. The router also emits a plain `prob = softmax(a/tau)` for
test time, where I drop the noise and take `argmax(prob)`. Training prediction is `(all_preds *
choice).sum`; test prediction is `all_preds[argmax(prob)]` — label-free, property (c) satisfied at
inference.

Train this end to end and it collapses: every sample routes to one predictor and the other two
starve — the self-reinforcing gating pathology. It is worth tracing why so I fix it with the right
tool. If head `k` is even slightly better on a batch, the gate hands it more mass, its gradient
signal is larger, it improves faster, and the gate concentrates further next step; the fixed point is
winner-take-all. A soft load-balancing penalty does *not* fix this, because it only equalizes mass —
it says nothing about *which* sample belongs to *which* predictor, so an arbitrary balanced
assignment discovers no patterns at all. So I have to state the real objective: assign each sample to
the predictor that fits it *best*, subject to keeping the predictors balanced. Pack the predictor
errors into a loss matrix `L ∈ R^{N×K}`, seek an assignment `P` minimizing `⟨P, L⟩` with each sample
getting one predictor (row sums one) and each predictor a prescribed share (column sums `ν_k N`). Row
marginals fixed, column marginals fixed, nonnegative entries, linear cost — that is exactly an
**optimal-transport** problem, and "do not collapse" *is* the column-marginal constraint, not a
bolted-on penalty. The exact LP is too slow per minibatch, so use the entropic relaxation: the
`sinkhorn` step exponentiates `exp(-L_h/ε)` and runs three iterations of column-then-row
normalization.

The sign in that exponent is the one place I refuse to hand-wave, because getting it backwards
silently trains the anti-model, so let me trace a tiny case. Take two samples, two heads, with loss
matrix `L = [[0.1, 0.9],[0.9, 0.1]]` — sample 0 is fit well by head 0, sample 1 by head 1. With `ε =
0.1`, `exp(-L/ε)` gives `[[e^{-1}, e^{-9}],[e^{-9}, e^{-1}]] ≈ [[0.368, 0.0001],[0.0001, 0.368]]`.
Normalizing columns then rows leaves the mass essentially on the diagonal — sample 0 → head 0, sample
1 → head 1 — which is what I want: low loss earns high transport mass. Flip the sign to `exp(+L/ε)`
and the mass lands on the off-diagonal, routing every sample to the head that fits it *worst* — a
model that trains each predictor on exactly the samples it is bad at. So the minus sign is
non-negotiable, and the `L_h` fed to Sinkhorn is a blend `L_h = α·norm(L) + (1−α)·norm(hist)` with
`α = 0.5`, fusing the noisy single-batch loss with a smoother historical loss so the transport target
does not thrash from minibatch to minibatch.

The transport `P` needs labels to compute `L`, so it cannot itself be the test selector — it is a
*teacher*. The router, fed only LR and TPE, is trained to predict the assignment the OT oracle would
produce, by cross-entropy: add `reg = Σ_k P_ik log q_ik` and minimize `loss = task_loss − λ·reg`,
which is task loss plus a cross-entropy from the router's `q` to the OT target `P`. Two parameters
need pinning. The pattern shares `ν_k` are unknown, so the maximally noncommittal prior is equal
shares `1/K`, which also maximally fights collapse — but equal shares is *wrong* in the limit, since
the real regimes are not equal mass, so I decay the teacher's grip: `λ_t = λ_0·ρ^{⌊step/100⌋}` with
`ρ = 0.99` and `λ_0 = 1.0`. Early on the equal-share transport forces diverse, balanced predictors;
later the shrinking `λ` lets the task loss settle into the data's real share structure. A cold router
also cannot bootstrap off random predictors, so the fill **pretrains** (`pretrain=True`): first train
the encoder and predictors with the OT oracle assignment used directly (`transport_method="oracle"`,
`λ=0`) so the heads specialize, then reset the optimizer and train the router
(`transport_method="router"` plus the OT cross-entropy). And because recomputing every predictor's
error over all `N` samples each step would cost a full forward pass, the model keeps an external
memory updated in two stages (`memory_mode: "sample"`): a full refresh before each epoch and
on-the-fly patches for the current minibatch (`assign_data`). Training is `lr=1e-3` Adam,
`n_epochs=100`, early stop 20 on validation IC. The whole model — attentive-LSTM encoder, the TRA
router with `K` predictors and the TPE-LSTM, the Sinkhorn solver, the two-stage memory, the
pretrain-then-route loop, and the label-free argmax inference — is the fill of `custom_model.py`; the
full module is in the answer.

Now the part that makes this rung genuinely *first*: what I expect it to do, stated sharply enough
that its measured result diagnoses the next move rather than merely grading this one. This is a heavy
machine — an attentive LSTM, three predictors, an OT-distilled router, a pretraining phase, a 60-day
sampler over a 20-factor slice — and it carries real risk on this data. The panel is not thin in
sample *count* (roughly 300 names over ~1700 training days is on the order of 5×10^5 windows), but it
is desperately thin in *signal*: an IC of 0.04 is an R² near 0.0016, so under a fifth of a percent of
the cross-sectional variance is learnable on a given day. Discrete routing splits that already faint
signal into three per-head slices, and the router must learn a label-free selector from cached error
history that is itself low-signal. The OT teacher fights collapse but also imposes a balance the data
may not support, and `FilterCol` to 20 columns discards most of Alpha158 — a steep price if the
dropped factors carried regime information. So my honest expectation is that TRA *works* — it is a
complete drift-routing model and should clear the averaged linear default, especially on the
risk-adjusted metrics the scoreboard rewards — but that its IC and rank-IC land in the modest range
and its information ratios are respectable rather than dominant, with the three regimes treated
unevenly. The falsifiable shape I will look for: because the router's selector is learned from error
history on the *training* distribution, I expect it to be strongest on the window closest to training
(`csi300_shifted`, 2016–2018) and weakest on the window furthest from it (`csi300_recent`,
2019–2020). If `csi300_recent`'s information ratio comes in clearly below the `csi300` window's — say
less than half of it — that is the tell: the routing capacity is being spent on separating
in-distribution patterns and is not, by itself, *aligning* the model to a shifted future. TRA is
deterministic across seeds (its backbone is seeded), so anything I see there is structural, not
noise, which makes that gradient a clean read. And that diagnosis is exactly what the next rung — a
model that explicitly forces what is *invariant* across time rather than routing within a single
distribution — would be built to address.
