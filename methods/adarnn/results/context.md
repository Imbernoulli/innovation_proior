# Context: forecasting non-stationary time series under shifting distributions (circa 2020-2021)

## Research question

A great many forecasting problems hand us a single long stream of measurements — air-quality
sensors, household power meters, daily financial factors — and ask us to predict the next few
steps. The statistical properties of these streams are not fixed: the mean, the variance, the
correlations among input variables all drift as time passes. A model fit on one stretch of
history is therefore being asked to predict a stretch whose input distribution differs from the
data it trained on. Write the joint law of inputs and target as `P(x, y)`. Over a long series the
marginal `P(x)` changes with time — regimes, shocks, seasonal and structural shifts — and the test
stretch at the end has a marginal `P(x)` unlike the bulk of the training data. The relationship
`P(y | x)` — the underlying law mapping a configuration of inputs to its outcome (in markets, the
economic regularity that turns factors into returns; in air quality, the physics turning conditions
into pollutant levels) — tends to persist across the series.

The goal is a forecaster for this setting: one that captures the temporal dependency a sequence
carries while predicting a test stretch whose input distribution differs from the training bulk,
where the drift is unlabeled in time — nobody tells us where one regime ends and the next begins,
or how many regimes the training stream contains. Each existing tool below addresses part of this
setting.

## Background

Forecasting non-stationary series has a long pedigree. Classical statistical models —
autoregressive integrated moving average (ARIMA), hidden Markov models, dynamic Bayesian
networks, Kalman filters — model the temporal structure with explicit parametric assumptions:
linear or low-order dynamics, hand-built features, fixed noise models, and data pre-processing.
They have carried much of the practical load for decades.

The deep-learning approach is the recurrent neural network. A vanilla RNN, and its gated
descendants the LSTM and the GRU (Chung et al. 2014), process a sequence one step at a time,
carrying a hidden state forward: `h^t = delta(x^t, h^{t-1})`, where `delta` is the cell's update
(for a GRU, the reset/update-gate computation). The hidden state accumulates information about the
sequence so far, so the network can find highly nonlinear, long-range temporal dependence without
the analyst hand-specifying the dynamics. The GRU in particular is a workhorse: fewer parameters
than an LSTM, robust to train, strong on real data. By this time the recurrent net is the default
backbone for sequence forecasting, often dressed up with attention or seq2seq decoding for
multi-step output.

These methods — classical and recurrent alike — share one assumption: that training and test data
are independent draws from a *single* distribution, the I.I.D. assumption. The generalization
theory for non-I.I.D., non-stationary sequences (Kuznetsov & Mohri 2014) studies the case where it
does not hold: when the training distribution and the target distribution diverge, the usual
learning guarantees weaken, and the bounds depend on how the training data's distributions are
arranged — diversity across the training stream is a quantity that enters the guarantee.

A second body of work supplies the language for "two datasets, different input distributions, same
input-to-output law." Shimodaira (2000) named this *covariate shift*: training and test share the
conditional `P(y | x)` but differ in the marginal `P(x)`. The classic covariate-shift remedy is
importance-reweighting the training loss by `P_test(x) / P_train(x)`; the formalism is posed for
ordinary (non-sequential) data, with a single fixed train/test pair. The richer machinery for
distribution gaps comes from domain adaptation and domain generalization, which learn
*domain-invariant representations* — features whose distribution is forced to look the same across
domains. Two measurement tools recur there and matter here:

- **Maximum Mean Discrepancy (MMD)** (Borgwardt et al. 2006; Gretton et al. 2012). Map each
  distribution to its mean embedding in a reproducing-kernel Hilbert space and measure the
  distance between embeddings: `MMD(P, Q) = || mu_P - mu_Q ||_H`. Empirically, with a kernel
  `k`, the squared MMD over samples `{x_i}` from `P` and `{y_j}` from `Q` is
  `(1/n^2) sum_{i,j} k(x_i, x_j) + (1/m^2) sum_{i,j} k(y_i, y_j) - (2/nm) sum_{i,j} k(x_i, y_j)`.
  With a linear kernel `k(a,b) = <a,b>` this collapses to `|| mean(X) - mean(Y) ||^2`; with a
  universal RBF kernel `k(x,x') = exp(-||x-x'||^2 / (2 sigma^2))` it senses all moments. MMD is
  differentiable in the features, so it can be added as a regularizer that pulls two
  representations' distributions together.
- **Domain-adversarial training** (Ganin et al. 2016, "RevGrad"/DANN). Attach a small domain
  classifier `D` that tries to tell which domain a feature came from, and train the feature
  extractor to *fool* it — a minimax game whose equilibrium is a representation in which the
  domains are indistinguishable. Mechanically this is implemented with a gradient-reversal layer
  (identity on the forward pass, gradient multiplied by `-lambda` on the backward pass), so one
  ordinary backprop simultaneously trains `D` to separate domains and the features to confuse it.
  The adversarial discrepancy is `d_adv = -( E[log D(h_s)] + E[log(1 - D(h_t))] )`.
- **CORAL** (Sun et al. 2016) aligns second-order statistics directly:
  `d_coral = (1/(4 q^2)) || C_s - C_t ||_F^2`, with `C` the feature covariance and `q` the
  feature dimension — a kernel-free distribution distance.
- A plain **cosine** distance on the mean feature vectors,
  `1 - <h_s, h_t> / (||h_s|| ||h_t||)`, is the cheapest alignment signal.

Two further principles are on the table. The first is the **principle of maximum entropy**
(Jaynes 1982): among all hypotheses consistent with what you actually know, commit to the
least-committal one — the one that assumes no structure beyond your constraints. The second is
**boosting** (Schapire 2003): an iterative scheme that repeatedly reweights to concentrate effort
on the parts of a problem that are currently handled worst, driving the overall error down by
attacking the hard cases rather than re-polishing the easy ones.

Empirically, the observations going in are these. (i) On a long non-stationary stream, a model
trained under the I.I.D. assumption is evaluated on the later, distributionally different stretch.
(ii) Domain-adaptation alignment methods were built and validated for *image classification with
CNNs* on a single, given source/target pair; the domains are handed to them, and there is no
sequence. (iii) When such an alignment regularizer is applied to a recurrent forecaster, an RNN
produces a *sequence* of intermediate states, not just one summary.

## Baselines

These are the prior methods a new forecaster would be compared against and would react to.

**Classical statistical forecasters (ARIMA; FBProphet; HMM / Kalman).** ARIMA models a series as
autoregressive + moving-average on a differenced (integrated) signal; Prophet fits trend +
seasonality + holidays; HMMs and Kalman filters posit a latent state with linear-Gaussian or
discrete dynamics. Core idea: impose an explicit, low-order parametric dynamic and estimate its
coefficients.

**GRU / LSTM recurrent forecasters (Chung et al. 2014).** Run a gated recurrent cell over the
sequence, take the final hidden state (or each state, for multi-step), and read out a prediction
with a linear head; train end to end by minimizing prediction error (MSE for regression). Core
idea: let the network learn the nonlinear temporal dependence with no parametric dynamics imposed.
The network is trained by minimizing average error over the whole training stream.

**Latest sequence models — LSTNet (Lai et al. 2018), STRIPE (Le Guen & Thome 2020),
Transformer (Vaswani et al. 2017, masks removed for forecasting).** LSTNet mixes convolution,
recurrence and an autoregressive skip to capture short- and long-period patterns; STRIPE shapes
the predicted trajectory with structured shape-and-time losses; a mask-free Transformer uses
self-attention to relate all positions. Core idea: richer architectures for sharper pattern
extraction, optimizing a single-distribution objective.

**Domain-adaptation / domain-generalization alignment on a recurrent backbone ("MMD-RNN",
"DANN-RNN").** Take an RNN forecaster and add a distribution-alignment regularizer — MMD, or a
domain-adversarial term — on the network's output representation, in the spirit of CNN-based
domain adaptation. Core idea: force the learned representation's distribution to look the same
across the two stretches so the predictor transfers. These methods take a given labelled source
and target, were designed for CNN classification on images, and apply the alignment to a recurrent
model's single output summary.

## Evaluation settings

The natural yardsticks already in use:

- **Air-quality forecasting** (Beijing multi-station hourly data, 2013-2017): six features
  (PM2.5, PM10, SO2, NO2, CO, O3), predict a pollutant; train/valid/test split chronologically
  per station; metrics RMSE and MAE.
- **Household electric power consumption** (one-minute measurements, 2006-2010, ~2M rows after
  cleaning): several power/voltage attributes, predict next-day value from the previous few days;
  6:2:2 chronological split; metric RMSE.
- **Human activity recognition** (UCI smartphone, accelerometer/gyroscope/magnetometer): sliding
  window of length 128 over 9 channels, classify one of six activities; metrics accuracy,
  precision, recall, F1, AUC. (A classification yardstick that stresses train/test distribution
  difference.)
- **Stock-return forecasting**: a large panel of daily financial factors (hundreds of features)
  over many years, predict the next-period cross-sectional return, split chronologically with the
  most recent years held out as test. Signal metrics: information coefficient (IC) =
  `corr(f_{t-1}, r_t)` and rank IC = `corr(order^f_{t-1}, order^r_t)`, with their information
  ratios IC/std (ICIR, RankICIR); portfolio metrics from a top-k/drop backtest (annualized return,
  information ratio, max drawdown). Higher IC/ICIR/IR is better.
- Protocol: same chronological train/valid/test split across methods; GRU as the standard
  recurrent backbone; hyperparameters tuned on validation; results averaged over several random
  runs.

## Code framework

The new forecaster plugs into the recurrent time-series training harness already in use for the
baselines. The available substrate is a data pipeline that turns a stream into `(feature, label)`
segments, a generic recurrent backbone (stacked GRU cells producing a hidden state per timestep,
then a linear head), a per-step prediction loss (MSE), and a minibatch training loop with an
optimizer.

```python
import torch
import torch.nn as nn


class RecurrentForecaster(nn.Module):
    """Generic recurrent backbone: stacked GRU cells emit a hidden state per
    timestep; a linear head reads out the prediction."""

    def __init__(self, n_input, n_hiddens=(64, 64), n_output=1, dropout=0.0):
        super().__init__()
        in_size = n_input
        cells = []
        for h in n_hiddens:
            cells.append(nn.GRU(input_size=in_size, hidden_size=h,
                                num_layers=1, batch_first=True, dropout=dropout))
            in_size = h
        self.cells = nn.ModuleList(cells)
        self.head = nn.Linear(n_hiddens[-1], n_output)

    def hidden_states(self, x):
        """Run the stacked GRU; return the per-layer sequences of hidden states
        [h^1, ..., h^V] and the final-layer last state."""
        out = x
        per_layer = []
        for cell in self.cells:
            out, _ = cell(out.float())
            per_layer.append(out)            # [batch, V timesteps, hidden]
        last = out[:, -1, :]
        return per_layer, last

    def forward(self, x):
        _, last = self.hidden_states(x)
        return self.head(last).squeeze(-1)


def split_into_periods(train_stream):
    """The training stream is one long non-stationary series. How (and whether)
    to partition it along time is part of what we will design."""
    # TODO: decide how to organize the training stream for what comes next.
    pass


def training_objective(model, periods):
    """Train the forecaster given whatever organization of the stream we chose.
    The prediction loss already exists; the rest of the objective is still open."""
    pred_loss = 0.0
    for D in periods:
        x, y = D["feature"], D["label"]
        pred_loss = pred_loss + nn.functional.mse_loss(model(x), y)
    # TODO: the rest of the objective we will design.
    extra = 0.0
    return pred_loss + extra


# existing minibatch training loop the forecaster plugs into
def train(model, periods, optimizer, n_epochs):
    for _ in range(n_epochs):
        optimizer.zero_grad()
        loss = training_objective(model, periods)
        loss.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), 3.0)
        optimizer.step()
```

The backbone, the per-step MSE loss, and the training loop are the known substrate; how to
organize the non-stationary training stream and what extra objective to train under are the open
slots.
