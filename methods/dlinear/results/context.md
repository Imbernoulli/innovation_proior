# Context: long-term multivariate time-series forecasting (circa 2021-2022)

## Research question

Given a multivariate time series with `C` variates observed over a look-back window of `L`
steps — historical data `X = {X^t_1, ..., X^t_C}` for `t = 1..L`, where `X^t_i` is the value
of variate `i` at step `t` — predict the next `T` steps `{X^t_1, ..., X^t_C}` for
`t = L+1..L+T`, for every variate. The regime of interest is *long-term* forecasting,
`T >> 1` (horizons of dozens to hundreds of steps), which is far harder than one-step-ahead
prediction: errors have a long way to compound, and the model must keep representing structure
well past the last observed point.

The field has converged on increasingly elaborate architectures for this, each reporting a new
state of the art. The precise question is whether that elaboration is doing the work it is
credited with. A solution would be a forecaster that (1) maps a length-`L` history to a
length-`T` horizon for all channels in one shot, (2) actually *uses* the temporal order of the
look-back — more history should help, not hurt — and (3) is honest about where its accuracy
comes from, so a reported gain can be attributed to the mechanism that produced it rather than
to a confounded change in the experimental setup. Whether such a forecaster needs a
high-capacity sequence model at all is exactly what is unsettled.

## Background

**The forecasting setup and two ways to produce a horizon.** A multi-step forecaster can be
built in two fundamentally different ways. *Iterated multi-step* (IMS) forecasting
(Taieb & Hyndman 2012) learns a single-step predictor and applies it recursively, feeding each
prediction back as input to produce the next, `T` times. Because it is one model rolled
forward, IMS predictions have relatively low variance — but each step is fed its own
(imperfect) previous prediction, so a biased single-step model accumulates error as the
roll-out lengthens, which is precisely the failure mode of long horizons. *Direct multi-step*
(DMS) forecasting (Chevillon 2007) instead optimizes the full `T`-step objective in one shot: a
single map from the length-`L` history to the length-`T` horizon. DMS has no recursion and
therefore no recursive error accumulation, and is preferable when `T` is large or a clean unbiased
single-step model is hard to obtain — i.e. exactly the long-term regime. This IMS/DMS
distinction is a property of the *training strategy*, orthogonal to the architecture that
implements the map.

**Seasonal-trend decomposition.** A classical and standard move in time-series analysis is to
split a series into a slowly varying *trend-cyclical* component and a *seasonal / remainder*
component (the additive model `y = trend + seasonal + remainder`), because each component is
individually more regular and therefore more predictable than their sum (Cleveland et al.'s STL,
1990). The trend-cycle is commonly estimated by a moving average of the raw series;
the remainder is what is left after subtracting it. Autoformer (Xu et al., NeurIPS 2021) folded
this into a deep network as a `series_decomp` block — a moving-average kernel extracts the
trend, the residual is taken as the seasonal part, and the block is inserted between neural
layers — and FEDformer (Zhou et al., ICML 2022) extended it to a mixture of moving-average
kernels of several sizes. Decomposition is thus already a known, reusable preprocessing
primitive in this literature.

**The dominant architecture and a property of it that matters here.** Self-attention
(Vaswani et al., 2017) is the engine behind the recent surge of forecasting models. Its core
operation computes pairwise affinities between all elements of a sequence and forms each output
as an affinity-weighted sum of value vectors. A structural fact about that operation: before any
positional signal is added, it is permutation-equivariant — permute the input tokens and the
output tokens are permuted in the same way, with no other change, because the affinities depend
only on the content of pairs, not on their positions. Order information must therefore be injected
separately, via positional encodings added to the token embeddings. For language and vision this is benign: the tokens
carry rich semantic content, and the meaning of a sentence is largely preserved even if some
words are reordered, so attention's job is to find content correlations and a little positional
hinting suffices. Numerical time series are different in kind: an individual value like a
temperature or an electricity reading carries almost no standalone semantic content, and the
information lives almost entirely in the *order and spacing* of the points. On such data,
positional encodings restore order only partially, and any operation built on top of an
order-blind core risks discarding the very signal that matters.

**Diagnostics that a fair comparison must expose.** Two checks are especially relevant before
crediting a complex sequence model. First, vary the look-back window `L`: if a model really
extracts temporal relations from longer context, more history should help rather than merely add
noise. Second, inspect the chronological train/test split for level shifts: on datasets such as the
hourly electricity-transformer-temperature series, the overall level can drift over time, and a
forecaster that regresses absolute values needs a causal way to re-anchor to the current level. The
method slot below should be judged under these diagnostics, not only by headline benchmark
tables.

## Baselines

These are the prior forecasters a new long-term method would be measured against and reacts to.

**Vanilla Transformer for forecasting (Vaswani et al., 2017, applied to TSF).** Encoder-decoder
self-attention over the time steps, with an autoregressive decoder producing the horizon one
step at a time. **Gap:** attention is `O(L^2)` in time and memory, punishing at the long
sequence lengths LTSF needs; and the autoregressive decoder is IMS, so it accumulates error over
a long horizon — exactly the regime it is being asked to handle.

**Informer (Zhou et al., AAAI 2021).** Reduces attention cost with a *ProbSparse* self-attention
that keeps only the dominant query-key interactions, giving `O(L log L)` complexity, plus a
self-attention distilling operation; and crucially replaces the autoregressive decoder with a
*generative-style* decoder that emits the whole horizon in one forward pass (a DMS decoder).
**Gap:** the accuracy improvement it reports is entangled — it changed *both* the attention
mechanism and the forecasting strategy (to DMS) relative to the older baselines it beats, so it
is not isolated which change produced the gain. The model is still a high-capacity attention
stack whose use of temporal order is unverified.

**Autoformer (Xu et al., NeurIPS 2021).** Introduces an internal `series_decomp` block
(moving-average trend + seasonal residual) placed throughout the network, and replaces
dot-product attention with an *Auto-Correlation* mechanism that aggregates information at the
series' dominant lags found via the FFT. Also DMS. **Gap:** it brings two reusable ideas
(decomposition, DMS) bundled with a still-complex sequence model, and again does not separate
how much of its accuracy comes from the decomposition and the DMS strategy versus from the
auto-correlation sequence modeling itself.

**Pyraformer (Liu et al., ICLR 2022) and FEDformer (Zhou et al., ICML 2022).** Pyraformer uses
a pyramidal attention with `O(L)` complexity to capture multi-scale dependencies; FEDformer
attends in the frequency domain (Fourier/wavelet-enhanced blocks) with `O(L)` complexity and a
mixture-of-experts decomposition with multiple kernel sizes. Both DMS. **Gap:** the same
pattern — ever more intricate attention variants, each reporting a new best, each leaving open
whether the intricate part is what is responsible. The look-back-window and order-sensitivity
diagnostics above are still unresolved for this whole family.

A common thread across these: the *non-Transformer* baselines they were compared against in
their own evaluations were IMS forecasters (autoregressive statistical and RNN models), which
carry the long-horizon error-accumulation handicap. So a comparison that pits a DMS Transformer
against IMS baselines conflates "Transformer" with "DMS," and the prior art leaves this
confound unresolved.

## Evaluation settings

The natural yardstick already in use for long-term multivariate forecasting:

- **Datasets.** ETTh1 (7 variates, hourly electricity-transformer oil temperature and load,
  Zhou et al. 2021), Weather (21 meteorological indicators, 10-minute granularity), and ECL /
  Electricity (321 clients' hourly consumption); broader studies in this area also use ETTm,
  Traffic, Exchange-Rate, and ILI. These datasets and their public splits predate any new
  method.
- **Task.** Multivariate-in / multivariate-out (`features = M`): all channels are predicted. The
  standard long-term study sweeps horizons (96 / 192 / 336 / 720 for most datasets, shorter
  horizons for weekly ILI) and look-back windows, so both accuracy and use of additional history
  can be inspected.
- **Preprocessing.** Channel-wise standardization (zero-mean, unit-variance using training-set
  statistics); chronological train/validation/test split; sliding-window sampling of
  (look-back, horizon) pairs.
- **Metrics.** Mean squared error (MSE) and mean absolute error (MAE) computed over all channels
  and all horizon steps on the standardized series; lower is better.
- **Protocol.** Trained by direct multi-step regression with the MSE loss under a unified data
  pipeline so that architectural choices can be compared head-to-head.

## Code framework

A direct-multi-step forecasting harness already exists: the data pipeline windows each series
into (look-back, horizon) pairs and standardizes them, an MSE objective scores a predicted
horizon against the true one, and a training loop drives Adam over minibatches. What is *not*
settled is the architecture inside the `Model` — the map from a length-`L`, `C`-channel history
to a length-`T`, `C`-channel horizon. That is the single empty slot; everything around it is the
generic DMS scaffolding any forecaster would plug into.

```python
import torch
import torch.nn as nn


class Model(nn.Module):
    """Direct multi-step forecaster: map a look-back of length seq_len to a
    horizon of length pred_len, for all C channels, in a single forward pass.
    Input  x_enc : [batch, seq_len,  C]
    Output       : [batch, pred_len, C]
    """

    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.channels = configs.enc_in
        # Architecture slot: the map [batch, seq_len, C] -> [batch, pred_len, C].
        pass

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        # x_enc: [batch, seq_len, C]
        # Architecture slot: produce the horizon and return [batch, pred_len, C].
        pass


def moving_average_pad(x, kernel_size):
    """Smoothing primitive available in the toolbox: a length-preserving 1-D
    average pool. Replicate-pad both ends so the output length equals the input
    length, then average-pool with stride 1."""
    front = x[:, 0:1, :].repeat(1, (kernel_size - 1) // 2, 1)
    end = x[:, -1:, :].repeat(1, (kernel_size - 1) // 2, 1)
    x = torch.cat([front, x, end], dim=1)
    x = nn.functional.avg_pool1d(x.permute(0, 2, 1), kernel_size, stride=1)
    return x.permute(0, 2, 1)


# existing DMS training loop the model plugs into
def train(model, data_loader, optimizer):
    loss_fn = nn.MSELoss()
    for x_enc, y_true in data_loader:          # (look-back, horizon) window pair
        optimizer.zero_grad()
        y_pred = model(x_enc)                   # [batch, pred_len, C]
        loss = loss_fn(y_pred, y_true)          # direct multi-step MSE over the whole horizon
        loss.backward()
        optimizer.step()
```

The smoothing helper and the standardized pipeline are pre-existing primitives; the architecture
that fills the `Model` slot — and whether it needs to be a high-capacity sequence model at all —
is what remains to be designed.
