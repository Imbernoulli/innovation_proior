# Context: Transformer-based multivariate time series forecasting

## Research question

Given a multivariate time series — a panel of $N$ variates (channels) observed over $T$ historical
time steps, $\mathbf{X}=\{\mathbf{x}_1,\dots,\mathbf{x}_T\}\in\mathbb{R}^{T\times N}$ — predict the
next $S$ steps $\mathbf{Y}\in\mathbb{R}^{S\times N}$. The variates may carry different physical
meanings, units, and statistical distributions, and the events that drive them may reach different
channels with different delays. How should a sequence model be designed to produce accurate
multivariate forecasts from such panels?

## Background

**The standard Transformer.** Vaswani et al. (2017) built a sequence model from a small set of
reusable parts: scaled dot-product self-attention $\operatorname{softmax}(\mathbf{Q}\mathbf{K}^\top/\sqrt{d_k})\mathbf{V}$
that mixes information across a set of tokens, a position-wise feed-forward network (FFN) applied
identically to each token, layer normalization, and residual connections — plus an explicit
positional encoding, because attention itself is permutation-invariant and language has order. These
components have proven extraordinarily robust across NLP and vision.

**How forecasters used it.** When this machinery was carried over to time series, the near-universal
choice was to make a *token* out of one timestamp: at step $t$ the vector $\mathbf{X}_{t,:}\in\mathbb{R}^N$
of all variates is embedded into a $D$-dimensional "temporal token," and self-attention runs over the
$T$ such tokens to model temporal dependency. Autoformer (Wu et al. 2021), Informer (Zhou et al.
2021), FEDformer (Zhou et al. 2022) and others followed this temporal-token template and concentrated
their effort on the attention module — decomposition, sparsity, frequency-domain variants — mostly to
tame the $O(T^2)$ cost of attending over long lookbacks.

**What a temporal token contains.** The values packed into $\mathbf{X}_{t,:}$ are simultaneous
readings of channels that need not reflect the same underlying event: in real panels there are
systematic time lags between variates (a disturbance reaches different sensors at different delays),
and the channels differ in unit and distribution. A temporal token therefore mixes heterogeneous,
possibly time-misaligned quantities into one vector, and its receptive field is a single instant.

**Linear models as baselines.** Zeng et al. (2023) showed that a single linear layer over the
flattened lookback (DLinear, with a trend/seasonal decomposition) matches or beats these Transformers
on the standard benchmarks at a fraction of the cost. Revisiting-linear work (Li et al. 2023)
reinforced that simple dense weightings over time are strong baselines.

**Normalization for distribution shift.** Time series are non-stationary: the mean and scale of a
series drift over the window. Instance-normalization-style fixes that subtract each series' lookback
mean and divide by its lookback standard deviation, then restore them on the output, were shown to
help substantially (RevIN, Kim et al. 2021; Non-stationary Transformers, Liu et al. 2022). These are
applied as a wrapper around the forecaster.

## Baselines

**Vanilla temporal-token Transformer (Vaswani et al. 2017, as adapted; Autoformer/Informer/FEDformer
line).** Embed each timestamp's $N$ variates into a token; self-attention over the $T$ temporal
tokens; FFN per temporal token; layer norm over the variate mixture; encoder–decoder generative
formulation for multi-step output.

**DLinear / "Are Transformers Effective for Time Series Forecasting?" (Zeng et al. 2023).** Decompose
each series into trend and seasonal parts and apply a single linear map from the $T$-length lookback
to the $S$-length horizon, independently per channel (channel-independent).

**PatchTST (Nie et al. 2023).** Two ideas: *channel independence* — process every variate with a
single shared backbone, never mixing channels — and *patching* — group consecutive timesteps into a
patch and make the patch the token, enlarging the receptive field beyond a single instant and cutting
the number of tokens.

**Crossformer (Zhang & Yan 2023).** Explicitly models both cross-time and cross-dimension dependence
with a two-stage attention scheme and a redesigned architecture operating on patches.

**Informer / efficient-attention line (Zhou et al. 2021).** ProbSparse attention and a distilling
encoder to bring long-sequence attention below $O(T^2)$.

## Evaluation settings

Standard multivariate forecasting benchmarks: ECL (321 electricity-consumption channels), the ETT
family (Electricity Transformer Temperature, Zhou et al. 2021), Weather (21 meteorological channels),
Traffic, Solar-Energy (LSTNet, Lai et al. 2018), Exchange, and PEMS. The protocol fixes a lookback
$T$ and reports several horizons $S$ (e.g. $S\in\{96,192,336,720\}$ for most, $\{12,24,36,48\}$ for
PEMS). Series are split chronologically into train/val/test and z-score standardized by the data
loader. Metrics are MSE and MAE on the forecast (lower is better). In the exogenous setting the data
loader feeds all channels but only the designated target channel is scored. The optimizer is Adam
with an L2 (MSE) loss, batch size 32, a fixed small number of training epochs, early stopping on the
validation set, and a small learning rate. These datasets, splits, and metrics all predate any new
method and are the natural yardstick.

## Code framework

The forecaster slots into the Time-Series-Library harness: a data loader yields
`x_enc [B, T, N]` (lookback, all channels), `x_mark_enc [B, T, F]` (calendar/time features),
`x_dec`, `x_mark_dec`, and a target tensor; the training loop runs Adam on an MSE loss with early
stopping. The model is one `nn.Module` with a fixed signature; only its internals are open. The parts
below already exist as generic primitives; the contribution fills the single empty module. A known
per-instance normalization helper is available to the module, but using it, where to place it, and
what the Transformer block should operate over remain design choices.

```python
import torch
import torch.nn as nn

# --- primitives that already exist (generic, pre-method) ---
# nn.Linear, nn.LayerNorm, nn.Dropout, scaled dot-product multi-head attention, a position-wise FFN,
# residual connections — the standard Transformer building blocks.
# Instance normalization for distribution shift (subtract per-series lookback mean, divide by
# per-series lookback std; reverse on the output) is a known wrapper.

def instance_normalize(x):           # x: [B, T, N]
    means = x.mean(1, keepdim=True).detach()
    x = x - means
    stdev = torch.sqrt(x.var(1, keepdim=True, unbiased=False) + 1e-5)
    x = x / stdev
    return x, means, stdev

def instance_denormalize(y, means, stdev, length):   # y: [B, length, N]
    y = y * stdev[:, 0, :].unsqueeze(1).repeat(1, length, 1)
    y = y + means[:, 0, :].unsqueeze(1).repeat(1, length, 1)
    return y


class SeriesForecaster(nn.Module):
    """A multivariate forecaster. How to tokenize the panel, what self-attention and the FFN
    operate over, and how to map to the horizon is exactly what is to be designed."""

    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        # TODO: the tokenization + the encoder + the map-to-horizon we will design
        raise NotImplementedError

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # TODO: produce dec_out [B, pred_len, N]
        raise NotImplementedError

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return out[:, -self.pred_len:, :]
```
