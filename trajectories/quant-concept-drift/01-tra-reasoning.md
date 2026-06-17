The scaffold hands me a Ridge regression over Alpha158 and a frozen backtest, and the research
question is concept drift — so before I write anything I have to decide what the *first serious*
rung should be, the one whose failure mode will teach me what the next rung must fix. The default
Ridge is a single linear relation `p(y|x)` fit by average squared loss over 2008–2014, and that is
exactly the assumption I distrust: one fixed joint `P`, one mapping from factors to returns. The
market is not one `P`. Two of the best-documented cross-sectional effects point in opposite
directions at the same time — momentum (winners keep winning) and reversal (losers rebound) — and
which one dominates rotates with the regime. I can make this concrete without a new experiment: fit
the sign of "past return → future return" on different years of CSI300 and it flips. A single
coefficient vector cannot be both signs; the best a single relation does is average two
contradictory ones and fit neither. So the first rung should not be a bigger single model; it
should be a model that *holds several distinct relations at once and decides per sample which one
applies*. That is a routing sequence model, and it is the first thing I will fill into
`CustomModel`. Let me derive it against this exact scaffold, because the scaffold constrains it in
ways the generic idea does not.

The data really is `P = Σ_k ν_k P_k`, a mixture of trading patterns, each with its own
`p_k(y|x)` and its own time-varying share. If I knew each sample's pattern `k` this would be plain
multi-domain learning. I do not — the patterns are latent, and worse, even if I cheated and used
the *label* to assign each training sample to its best-fitting pattern, that trick dies at test
time where there are no labels and the test period is the whole point. So the constraint is sharp:
I need a model that (a) carries several relations, (b) routes each sample to one of them, and (c)
makes that routing decision from signals available *without the current label* — the sample's
latent representation and whatever past prediction-error state is already cached. Property (c) is
the one that kills the naive solutions.

Start with "one model, several behaviors." Mixture-of-experts is the right shape: `K` experts and a
gate. But what are the experts here? `K` full backbones would multiply parameters by `K` and, on a
noisy, smallish CSI300 panel, overfit hard. The patterns differ in the *relation* far more than in
how to *encode* a 60-day window, so I share one backbone and replicate only the cheap part — the
final linear head — into `K = num_states` linear predictors. `ŷ_ik = θ_k ∘ ψ(x_i)`, `ψ` shared,
`θ_k` a linear head. Near-zero extra parameters, and a free safety net: `num_states = 1` collapses
to the plain backbone, so I can never do worse than a single sequence model by construction. In this
task `num_states = 3` (the workflow's `MTSDatasetH` is configured with `num_states: 3`), which is a
deliberate, modest choice: enough heads to separate the main regimes, few enough that each head
still gets data.

Now the backbone, and here the scaffold pins the details. The reference fill uses an **attentive
LSTM** — `rnn_arch="LSTM"`, two layers, hidden 64, `use_attn=True` — not the GRU the generic
routing idea defaults to. The encoder projects the input only when compressing (here it does not,
since the input width is below hidden), runs the LSTM, takes the mean of the last hidden state over
layers, and concatenates that with an attention-pooled vector `att = Σ_t softmax_t(u^T tanh(W·h_t))
· h_t`, so the latent fed to the predictors and router has width `2·64 = 128`. The input width is
the next thing the scaffold fixes and it is *not* all of Alpha158: the editable processor block
runs a **`FilterCol`** that keeps exactly 20 named columns (RESI5, WVMA5, RSQR5, KLEN, …, KLOW)
before `RobustZScoreNorm` and `Fillna`, so `input_size = 20`. That matters — the routing model does
not see the full 158-factor view the Ridge default and the tree baselines see; it sees a curated
20-factor slice over a 60-day sequence. The sequence itself comes from `MTSDatasetH` with
`seq_len: 60`, `batch_size: 1024`, `memory_mode: "sample"`, `drop_last: true` — the editable
dataset block is swapped from `DatasetH` to this multi-timeseries sampler so each sample is a
(stock, time) window with its cached state, which is precisely what the router needs.

The router is the heart of it, and property (c) governs its inputs. It must run at test, so it can
only use label-free signals. The first is the latent `h_i = ψ(x_i)` — I route on `h` rather than
raw factors because `h` is already tuned toward prediction. But a single window's latent
underdetermines the regime; the difference between "we are in a momentum regime" and "we are in a
reversal regime" is not cleanly written in one stock's 60-day vector. I need a second signal, and
the right one comes from how a real investor rotates strategies: not by deducing the macro regime
from today's prices but by watching *how their strategies have been performing lately*. The
analogue is each predictor's *recent prediction error*. So feed the router the per-sample history of
the `K` predictors' losses — the cached `state` the `MTSDatasetH` carries — encoded by a small LSTM
(`tra_config` hidden 32, one layer) whose last hidden state is concatenated with the 128-dim latent
(`src_info="LR_TPE"`: latent-representation plus temporal-prediction-errors) and mapped by a linear
`fc` to `K` logits. One causality subtlety the sampler must respect: the most recent usable error is
from before the label horizon, never the current step, or the router leaks the future; batches flow
forward in time, never shuffled across the time axis.

How does the router *use* the logits? A soft `softmax` blend would re-create the single-model
problem — every predictor would see gradients from every sample and all drift back to the average
relation. To force specialization the routing has to be *discrete*: one predictor per sample, so a
head can swing its coefficients to, say, reversal without being dragged back by momentum samples.
Discrete selection is `argmax`, which has no gradient. The escape is the **Gumbel-softmax**: draw
`one_hot(argmax_k(a_ik + g_ik))` with `g` i.i.d. Gumbel, replace the argmax with a temperature
softmax at `tau=1.0`, and use the straight-through form — a hard one-hot `choice` in the forward
pass with the soft gradient in the backward. The router also emits a plain `prob = softmax(a/tau)`
for test time, where I drop the noise and take `argmax(prob)`. Training prediction is
`(all_preds * choice).sum`; test prediction is `all_preds[argmax(prob)]` — label-free, property (c)
satisfied.

Train this end to end and it collapses: every sample routes to one predictor, the rest starve —
the self-reinforcing gating pathology. A soft load-balancing penalty does not fix it, because it
equalizes mass without saying *which* sample belongs to *which* predictor, and arbitrary balanced
assignment discovers no patterns. State the real objective: assign each sample to the predictor that
predicts it *best*, subject to keeping the predictors balanced. Pack predictor errors into a loss
matrix `L ∈ R^{N×K}`, want an assignment `P` minimizing `<P, L>` with each sample getting one
predictor (row sums one) and each predictor a prescribed share (column sums `ν_k N`). Row marginals
one, fixed column marginals, nonnegative entries, linear cost — that is exactly an **optimal-
transport** problem, and "do not collapse" *is* the column-marginal constraint. The hard LP is too
slow per minibatch, so use the entropic relaxation: the scaffold's `sinkhorn` exponentiates
`exp(-L_h/ε)` (cost enters with a minus sign so low loss → high mass) and runs three iterations of
column-then-row normalization. The negative sign is non-negotiable.

The transport `P` needs labels, so it cannot be the test selector — it is a *teacher*. The router,
fed only LR and TPE, learns to predict the assignment the OT oracle would produce, by cross-entropy:
add `reg = Σ_k P_ik log q_ik` and minimize `loss = task_loss − λ·reg`, which is task loss plus a
cross-entropy to the OT target. Two parameters need pinning. The pattern shares `ν_k`: unknown, so
the maximally noncommittal prior is equal shares `1/K`, which also maximally fights collapse — but
equal shares is wrong in the limit, so I decay the teacher's grip, `λ_t = λ_0·ρ^{⌊step/100⌋}` with
`ρ = 0.99` and `λ_0 = 1.0`; early on the equal-share transport forces diverse balanced predictors,
later the task loss settles into the data's real share structure. And `α = 0.5` fuses the noisy
single-batch loss matrix with the smoother historical loss (`L_h = α·norm(L) + (1−α)·norm(hist)`)
before Sinkhorn, so the transport target is stable.

Two more scaffold-pinned practicalities. The router's TPE input — every predictor's recent errors —
would cost a full forward pass over all `N` samples every step if recomputed; instead the model
keeps an external memory updated in two stages (`memory_mode: "sample"`): a full refresh before each
epoch and on-the-fly patches for the minibatch (`assign_data`). And a cold router cannot bootstrap
off random predictors, so the fill **pretrains** (`pretrain=True`): first train the backbone and
predictors with the OT oracle assignment used directly (`transport_method="oracle"`, `λ=0`) so the
predictors specialize, then reset the optimizer and train the router (`transport_method="router"`
plus the OT cross-entropy). `lr=1e-3` Adam, `n_epochs=100`, early stop 20 on validation IC. The
whole model — RNN backbone, TRA router with `K` predictors and the TPE-LSTM, the Sinkhorn solver,
the two-stage memory, the pretrain-then-route loop, and the label-free argmax inference — is the
fill of `custom_model.py`; the full module is in the answer.

Now the part that makes this rung *first*: what I expect it to do, so its measured result diagnoses
the next move. This is a genuinely heavy machine — an attentive LSTM, three predictors, an
OT-distilled router, a pretraining phase, a 60-day sampler over a 20-factor slice — and it carries
real risk on this data. The CSI300 panel is small and noisy; routing splits an already-thin sample
into three even thinner per-predictor slices, and the router has to learn a label-free selector from
cached error history that is itself low-signal. The OT teacher fights collapse but also forces
balance the data may not support, and the `FilterCol` down to 20 factors throws away most of
Alpha158 — a steep price if the dropped factors carried regime information. So my honest
expectation is that TRA *works* — it is a real, complete drift-routing model and should beat the
linear default — but that its IC and rank-IC land in the modest range and its information ratios are
respectable rather than dominant, with the three regimes treated unevenly: a routing model trained
on 2008–2014 may route the 2019–2020 (`csi300_recent`) test window poorly, since that regime is the
furthest from training and the router's error-history signal is least reliable there. If the
`csi300_recent` numbers come in clearly weaker than the long-horizon `csi300` ones, that is the
tell: the routing capacity is being spent on in-distribution pattern separation and is not, by
itself, *aligning* the model to a shifted future. That diagnosis is exactly what the next rung — a
model that explicitly matches distributions across time rather than routing within them — is built
to address.
