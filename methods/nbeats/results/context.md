# Context: univariate point forecasting and the M4 competition (circa 2018-2019)

## Research question

Given a single univariate series observed over a look-back window of `L` steps — a history
`x = [x_1, ..., x_L]` of one variable — predict the next `H` steps `[x_{L+1}, ..., x_{L+H}]`.
The regime of interest is *short-horizon, many-series* forecasting as embodied by the M4
competition: 100,000 real-world series spanning very different sampling frequencies (Yearly,
Quarterly, Monthly, Weekly, Daily, Hourly), each short, each its own univariate problem, scored
by a percentage error. The question is whether a *pure deep-learning* model — with no hand-built
time-series components — can compete on this benchmark, which has been dominated by statistical
models and by hybrids that bolt a neural network onto a statistical core.

## Background

**The two ways to produce a multi-step horizon.** A multi-step forecaster is built either by
*iterated multi-step* (IMS) — learn a one-step predictor and roll it forward `H` times, feeding
each prediction back — or by *direct multi-step* (DMS) — learn a single map from the length-`L`
history to the whole length-`H` horizon at once. On short M4 horizons either is feasible. DMS is
a natural fit for a feed-forward network whose output is the whole horizon.

**The M4 competition and its winners.** M4 (Makridakis, Spiliotis & Assimakopoulos, 2018) is the
standard short-univariate-many-series benchmark, scored by **sMAPE** (symmetric mean absolute
percentage error) and **OWA** (overall weighted average of sMAPE and MASE, normalized against the
naive-2 benchmark). The M4 winner (Smyl 2018) was a *hybrid* — an Exponential-Smoothing/RNN that
wove a statistical ES model together with a recurrent network and per-series parameters. The
runner-up was a weighted *combination* of statistical methods.

**Seasonal-trend decomposition as a modelling primitive.** Classical time-series analysis splits a
series into a slowly varying trend and a repeating seasonal part (additive `y = trend + seasonal +
remainder`), because each component is individually more regular and predictable than their sum
(STL, Cleveland et al. 1990). Statistical forecasters and the M4 hybrids lean on this heavily,
usually with hand-chosen seasonal periods per frequency.

**Residual / boosting-style learning.** A long line of methods fits a target by a sum of
successive corrections — gradient boosting fits each new learner to the residual of the running
sum; deep residual networks (He et al. 2016) add a learned correction to an identity path so very
deep stacks remain trainable. The shared idea is *sequential refinement*: each block handles what
the previous blocks left unexplained. This is a reusable structural primitive a forecaster could
adopt, applied to the look-back window rather than to images.

**Basis expansion.** A classical way to constrain a function to a meaningful family is to express
it as a linear combination of fixed basis functions: low-degree *polynomials* describe a smooth
trend, *Fourier* sine/cosine pairs describe periodicity. If a network is made to output the
*coefficients* of such a basis rather than the forecast values directly, the resulting forecast is
constrained to the corresponding family (a trend, or a seasonal pattern) — which is the mechanism
by which a learned model could be made interpretable without a hand-imposed decomposition.

## Baselines

These are the forecasters a new pure-deep M4 method would be measured against and reacts to.

**Statistical benchmarks (naive-2, Theta, ETS, ARIMA, Holt).** The classical per-series models M4
uses to define OWA. Each is a low-capacity model with hand-set seasonality and no sharing across
the 100,000 series.

**The M4 winner: ES-RNN hybrid (Smyl 2018).** A per-series exponential-smoothing model whose
level/seasonality parameters are co-trained with a shared dilated-LSTM stack; the RNN forecasts
the de-seasonalized, normalized series and the ES part re-applies level and seasonality. The
statistical component is hand-built and the architecture is frequency-specific (different recipes
per M4 subset).

**Plain machine-learning models (MLPs, RNNs applied naively).** The pure-ML entrants that M4
reported as underperforming the statistical benchmarks.

## Evaluation settings

The yardstick already in use for M4:

- **Datasets.** The six M4 subsets by frequency (Yearly, Quarterly, Monthly, Weekly, Daily,
  Hourly), 100,000 series total, with the official train/test splits and per-frequency forecast
  horizons (`H` = 6 Yearly, 8 Quarterly, 18 Monthly, etc.).
- **Task.** Univariate-in / univariate-out: one series, one channel. Look-back length is a small
  multiple of the horizon (commonly `L` in `2H..7H`).
- **Metrics.** sMAPE (primary) and MASE, combined into OWA against naive-2; lower is better.
- **Protocol.** Trained by direct multi-step regression on the percentage metric, Adam optimizer,
  with the official M4 evaluation on the held-out horizon.

## Code framework

A direct-multi-step univariate forecasting harness already exists: the pipeline windows each
series into (look-back, horizon) pairs, an sMAPE objective scores a predicted horizon, and a
training loop drives Adam. The empty slot is the architecture inside `Model` — the map from a
length-`L` history to a length-`H` horizon for one channel.

```python
import torch
import torch.nn as nn


class Model(nn.Module):
    """Direct multi-step univariate forecaster: map a look-back of length seq_len
    to a horizon of length pred_len, one channel, in a single forward pass.
    Input  x_enc : [batch, seq_len,  1]
    Output       : [batch, pred_len, 1]
    """

    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        # TODO: the forecasting architecture we will design — the map
        #       [batch, seq_len, 1] -> [batch, pred_len, 1].
        pass

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        # TODO: produce the horizon and return [batch, pred_len, 1]
        pass


# existing DMS training loop the model plugs into
def train(model, data_loader, optimizer):
    def smape(pred, true):
        return (200.0 * (pred - true).abs() / (pred.abs() + true.abs() + 1e-8)).mean()
    for x_enc, y_true in data_loader:          # (look-back, horizon) window pair
        optimizer.zero_grad()
        y_pred = model(x_enc)                   # [batch, pred_len, 1]
        loss = smape(y_pred, y_true)            # direct multi-step sMAPE over the whole horizon
        loss.backward()
        optimizer.step()
```

The standardized pipeline and the sMAPE objective are pre-existing; the architecture that fills the
`Model` slot is what remains to be designed.
