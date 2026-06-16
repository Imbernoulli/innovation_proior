# Context: learning stock-return predictors under multiple coexisting trading patterns (circa 2020)

## Research question

A stock-return predictor is trained the way every supervised model is trained: collect observations
`{(x_i, y_i)}_{i=1}^N` — `x_i` the features available at time `t` for stock `s`, `y_i` the future return (or its
cross-sectional rank) — and fit one estimator parameterized by `θ` by empirical risk minimization, implicitly
assuming every `(x_i, y_i)` is an i.i.d. draw from a single fixed joint distribution `P`. The trouble is that
the market is the aggregate of many participants running different strategies, so the causal relation between
features and future return — the *trading pattern*, the conditional `p(y | x)` — is not one relation but
several, and which one is in force changes over time. Two of these patterns are not just different but
*opposite*: the **momentum** effect (stocks that rose keep rising) and the **reversal / mean-reversion** effect
(stocks that fell rebound). A single estimator confronted with samples from contradictory patterns can only fit
their average, which fits neither.

The precise goal is a predictor that (1) can represent several distinct feature-to-return relations at once rather
than one averaged relation; (2) decides, for each sample, which relation applies — *and can do so at test time,
where no labels are available to reveal the pattern*; (3) discovers these patterns and their assignment with no
ground-truth pattern identifiers anywhere in the data; (4) is cheap enough to bolt onto an existing backbone
(LSTM / Transformer) without multiplying its parameter count; and (5) degrades gracefully — if there really is
only one pattern, it must not do worse than the plain backbone. The open problem is to get all five properties
in one supervised stock predictor.

## Background

By 2020 the dominant recipe for machine-learning stock prediction is a sequence model over a fixed feature
window. Features are engineered factors (market cap, book-to-market, momentum, reversal, volatility, …) or raw
price/volume series, usually transformed to cross-sectional ranks per trading day so the target is uniform and
robust to market-wide swings; the label is the next-period cross-sectional return rank, making the task a
ranking problem scored mainly by the **information coefficient** (IC), the cross-sectional correlation between
predicted and realized ranks. The backbone is typically an attention-augmented LSTM or a Transformer encoder
that compresses the lookback window into a latent vector, followed by a linear head that emits the score.

The load-bearing background fact is that the market data is *non-stationary in its conditional law*, and this is
documented, not conjectured. Two strands of empirical finance establish it. First, the coexistence of
contradictory cross-sectional effects: the momentum effect — past winners continue to outperform — is
well-documented (Jegadeesh & Titman; Fama & French 2012, Asness et al. 2013), and so is the opposite
reversal effect — past losers rebound (Jegadeesh 1990; Poterba & Summers 1988 on mean reversion). Both are
real and both appear in the same market, so the sign of the relationship between past return and future return
is not fixed. Second, classic style factors lead in different epochs: the annualized excess returns of *size*
(buy small caps), *value* (buy high book-to-market), and *momentum* (buy the trailing-12-month winners) take
turns dominating — for instance size leads before 2017 while momentum leads from 2019 on — which is direct
evidence that the active pattern rotates over time.

A sharper, model-level diagnostic makes the contradiction concrete. Fit a simple linear model of next-month
return rank on three features (size rank, value rank, 12-month momentum rank) separately on data from different
years and read off the coefficients. The momentum coefficient comes out **negative** when fit on 2009 (that
year's pattern: high past return predicts *low* future return — reversal) and **positive** when fit on 2013
(momentum). A single set of linear coefficients cannot be simultaneously negative and positive; no single model
— linear or not, with a fixed parameter vector — can hold both regimes. This observation forces the question:
the data genuinely carries multiple patterns, so one model is structurally insufficient.

Two conceptual frames are on the table for "one model, several behaviors." The first is **conditional
computation**: instead of one monolithic function, keep several sub-functions and a gate that, per input,
activates a subset, so different inputs are processed by different parameters at near-constant cost. The second
is **balanced assignment**: when labels or clusters are free to move, unconstrained self-assignment can collapse
into a single group, while a fixed marginal constraint keeps all groups populated. Those are mature tools, but
the stock-prediction setting adds a sharper constraint: the grouping signal cannot rely on the current label at
test time.

## Baselines

These are the prior methods a new approach would be measured against and react to.

**Single-backbone sequence predictors (ALSTM; Transformer).** The Attentive LSTM (Qin et al. 2017; Feng et al.
2018) maps the feature window through a per-timestep projection into an LSTM, then *temporal-attention-pools*
the hidden states — attention scores `softmax_t(u^T tanh(W h_t))` weight the sequence into one vector, which is
concatenated with the final hidden state and passed to a linear head. The Transformer variant (Ding et al.
2020, after Vaswani et al. 2017) replaces the recurrence with positional encoding plus self-attention and reads
out the last position. Both are strong, but each is a *single* estimator with a *single* output head fit by
ERM: under the i.i.d. assumption it can only learn the average relation, so when momentum and reversal samples
are mixed it splits the difference. **Gap:** one parameter vector, one relation — it cannot represent
opposite-signed patterns simultaneously, and it has no mechanism to recognize which regime a sample belongs to.

**Frequency-decomposed recurrence (SFM; Zhang et al. 2017).** The State Frequency Memory network decomposes the
LSTM cell state into multiple frequency components, aiming to capture trading patterns operating on different
time scales. **Gap:** the decomposition is along *frequency*, fixed by architecture; it does not discover
arbitrary latent patterns nor route a sample to a chosen sub-model, and it remains a single fused predictor.

**Non-adaptive references (Linear, MLP, LightGBM; Ke et al. 2017).** Linear regression, a plain MLP, and
gradient-boosted trees over the same features. **Gap:** all are single-model ERM with no notion of multiple
patterns at all; included as the floor.

**Conditional computation / mixture-of-experts gating (Shazeer et al. 2017).** Keep `K` expert sub-networks
and a trainable gating network `G(x)` that produces a sparse combination per example; train everything jointly
by backprop. This is exactly the "several sub-functions, one gate" frame. But the gate has a well-documented
failure mode: it "converges to a state where it always produces large weights for the same few experts," and
the imbalance is *self-reinforcing* — the favored experts get more gradient, so they are favored even more,
until the rest go unused. Shazeer et al. counter this with a **soft auxiliary penalty**: an importance loss
equal to the squared coefficient of variation of the per-expert gate mass, `L_importance = w·CV(Importance)^2`,
plus a separate load loss, added to the objective to push the experts toward equal importance. **Gap:** the
penalty is *soft* — it nudges the gate toward balance but does not enforce it, and Shazeer et al. observe that
even equalized importance can leave the *number of examples* per expert highly unequal; and the gate is trained
only to balance, not to send a sample to the expert that actually predicts it best. For a problem where the
sub-models must end up specialized to genuinely distinct patterns, "nudge toward balance" is not enough.

**Degeneracy-avoidance by balanced assignment as optimal transport (Asano et al. 2020; Caron et al. 2020).**
In simultaneous clustering / self-labelling, taking the model's own predictions as classification targets and
minimizing cross-entropy *collapses*: the loss is "trivially minimized by assigning all data points to a single
label." The fix is to add the constraint that "the labels must induce an equipartition of the data," which
turns label assignment into an optimal-transport problem — `min_P <P, L>` over the transportation polytope
`U(r,c) = {P ≥ 0 : P\mathbf{1} = r, P^T\mathbf{1} = c}`, with the column marginal forcing equal-sized groups.
Solved exactly this is an `O(d^3 log d)` linear program, too slow to run per minibatch; the entropic
relaxation of Cuturi (2013) makes the optimum a normalized exponential `P = diag(u)·exp(-L/ε)·diag(v)` whose
scaling vectors `u, v` enforce the row and column marginals through Sinkhorn–Knopp matrix scaling; in a
minibatch implementation this is just repeated normalization, which is GPU-vectorizable and fast — three
iterations suffice in related online clustering practice (Caron et al. 2020). **Gap:** this machinery balances
and assigns *clusters/labels* in representation-learning settings where the assignment can be used directly.
It does not by itself supply a supervised stock predictor whose sample assignment must also be reproduced at
test time without the current label.

**Supporting primitive — differentiable discrete selection (Jang et al. 2016; Maddison et al. 2016).** A gate
that truly separates patterns must make a *discrete* choice of one sub-model, but `argmax` is not
differentiable. The Gumbel-Max trick draws a categorical sample as `one_hot(argmax_i (g_i + log π_i))` with
`g_i ~ Gumbel(0,1)`; replacing the `argmax` with a temperature-`τ` softmax,
`y_i = exp((log π_i + g_i)/τ) / Σ_j exp((log π_j + g_j)/τ)`, gives the Gumbel-softmax — a continuous relaxation
on the simplex that anneals to one-hot as `τ→0` and is differentiable, optionally with a straight-through
estimator for hard samples. This is the available tool for a differentiable-but-discrete gate.

## Evaluation settings

The natural yardsticks are:

- **Universe and features.** The qlib Alpha158 workflow uses the CSI300 universe from the China market data
  bundle. It filters the feature group to 20 Alpha158 columns
  (`RESI5`, `WVMA5`, `RSQR5`, `KLEN`, `RSQR10`, `CORR5`, `CORD5`, `CORR10`, `ROC60`, `RESI10`, `VSTD5`,
  `RSQR60`, `CORR60`, `WVMA60`, `STD5`, `RSQR20`, `CORD60`, `CORD10`, `CORR20`, `KLOW`), applies robust
  z-score normalization and fillna to features, and applies cross-sectional rank normalization to the label.
  Label: `Ref($close, -2) / Ref($close, -1) - 1`.
- **Temporal split.** Train / validation / test are split in strict chronological order:
  train `2008-01-01` to `2014-12-31`, validation `2015-01-01` to `2016-12-31`, and test `2017-01-01` to
  `2020-08-01`. The sequence dataset feeds 60-step windows and preserves temporal ordering for evaluation.
- **Ranking metrics.** Information Coefficient `IC = corr(ŷ, y)` averaged over trading days; its stability
  ratio ICIR = mean(IC)/std(IC); rank-IC and rank-ICIR; and regression errors MSE / MAE. Higher IC/ICIR is
  better.
- **Portfolio metrics from a backtest.** Feed the predicted scores into a fixed top-k/drop strategy
  (TopkDropout, top 50 / drop 5) and a day-by-day backtest, reporting annualized return and information /
  Sharpe ratio (higher better) and maximum drawdown (closer to zero better). Computed by the standard
  signal-analysis and portfolio-analysis records.

## Code framework

The new scoring module plugs into an existing qlib training harness. What already exists: a dataset object that
prepares time-series minibatches (a sequence per sample over a lookback window, with labels and bookkeeping
indices), a backbone sequence encoder that maps a window to a latent vector, an Adam optimizer, a mean-squared
prediction loss, and the `Model` interface (`fit(dataset)` / `predict(dataset, segment)` returning a
`pd.Series` indexed by `(datetime, instrument)`). The starting predictor is just: encode the window, apply
one linear head, regress to the label. Everything about *how to represent and select among multiple patterns*
is the empty slot.

```python
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


class Backbone(nn.Module):
    """A pre-existing sequence encoder (e.g. attentive LSTM / Transformer) that maps a
    feature window [batch, seq_len, d_feat] to a latent vector [batch, hidden]."""
    def __init__(self, input_size, hidden_size, num_layers, use_attn=True, dropout=0.0):
        super().__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True, dropout=dropout)
        self.use_attn = use_attn
        if use_attn:
            self.W = nn.Linear(hidden_size, hidden_size)
            self.u = nn.Linear(hidden_size, 1, bias=False)
            self.output_size = hidden_size * 2
        else:
            self.output_size = hidden_size

    def forward(self, x):
        rnn_out, (h, _) = self.rnn(x)
        last = h.mean(dim=0)
        if self.use_attn:
            score = self.u(self.W(rnn_out).tanh()).softmax(dim=1)
            att = (rnn_out * score).sum(dim=1)
            last = torch.cat([last, att], dim=-1)
        return last                                   # latent representation h_i = psi(x_i)


class OutputHead(nn.Module):
    """Given the backbone latent and any per-sample state available in the dataset,
    produce a scalar score per sample."""
    def __init__(self, input_size):
        super().__init__()
        # TODO: the scoring object we will design goes here.
        pass

    def forward(self, hidden, state):
        # TODO: turn the latent and state into a per-sample score.
        pass


def train_loss(pred, label):
    mask = ~torch.isnan(label)
    return (pred[mask] - label[mask]).pow(2).mean()   # existing MSE loss


class CustomModel:
    """qlib Model interface. fit() trains backbone + head on time-series minibatches;
    predict() returns a pd.Series of scores indexed by (datetime, instrument)."""
    def __init__(self):
        self.backbone = None
        self.head = None
        self.optimizer = None

    def fit(self, dataset):
        train_set, valid_set, test_set = dataset.prepare(["train", "valid", "test"])
        # for each minibatch: hidden = backbone(data); pred = head(hidden, state)
        # loss = train_loss(pred, label); loss.backward(); optimizer.step()
        # TODO: the training procedure we will design.
        pass

    def predict(self, dataset, segment="test"):
        test_set = dataset.prepare(segment)
        # TODO: produce per-sample scores using the trained head.
        pass
```

The empty head and training / prediction bodies are the only unspecified pieces; the backbone, optimizer, loss,
dataset interface, and `Model` interface are already fixed.
