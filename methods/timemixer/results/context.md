# Context: deep time-series forecasting under intricate temporal variations (circa 2021-2023)

## Research question

Given a length-`P` window of past observations of a series with `C` variates, predict the
length-`F` future. The hard part is not the regression head; it is that real-world series carry
*intricate, deeply mixed* temporal variations. In a single window the signal is simultaneously
increasing, decreasing, oscillating at several rhythms, and drifting, and these are entangled
inside the same scalar stream. A traffic sensor recorded every five minutes shows a sharp
intraday commute rhythm; the same road averaged to a daily series loses that rhythm and instead
shows weekday/weekend and holiday structure; averaged to a monthly series it shows mostly a slow
macroscopic drift. So the same physical process presents different patterns depending on the
sampling scale: fine scales expose microscopic detail, coarse scales expose macroscopic
structure. A forecaster that commits to a single resolution sees only one slice of this. The goal
is a cheap forecasting component for a fixed short look-back (`P=96`) and many variates that can
handle mixed seasonal, periodic, and trend-like variation without assuming one native sampling
resolution is the whole story, plugged into a fixed training/evaluation pipeline so that
architectural choices can be compared head-to-head.

## Background

By this time deep forecasters are organized around two questions: which backbone models the
temporal axis, and which structural prior is imposed to tame intricate variations.

On backbones, four families coexist. RNNs (LSTM/GRU, DeepAR) carry a recurrent state but have a
limited effective receptive field and serialize badly over long horizons. Temporal CNNs (TCN,
MICN) convolve along time but their receptive field is bounded by depth and kernel size.
Transformers (Informer, Autoformer, FEDformer, Pyraformer, PatchTST, the Non-stationary
Transformer) buy global temporal context through attention, at quadratic-ish cost that a long
literature then tries to cut. Pure-MLP forecasters (N-BEATS, N-HiTS, LightTS, DLinear) regress
the future from the past with linear/MLP layers and are strikingly competitive at a fraction of
the cost — DLinear in particular showed that a one-layer linear map on a decomposed series rivals
elaborate Transformers, which reset expectations about how much machinery the task actually needs.

On structural priors, two paradigms dominate. The first is **series decomposition**. It is an old
idea from classical analysis — STL (Cleveland et al. 1990) separates a series into seasonal and
trend components precisely because the two hold *distinct properties*: the seasonal part is
short-term, roughly stationary, repeating; the trend is long-term, non-stationary, slowly varying.
Autoformer (Wu et al. 2021) brought this into deep models as a differentiable block: a moving
average over time is the trend, the residual is the season, and the network processes the two
separately. FEDformer used multiple averaging kernels; DLinear (Zeng et al. 2023) decomposed as a
pre-processing step and then applied one linear layer to each component. The second paradigm is
**multiperiodicity**: N-BEATS fits trigonometric bases, FiLM projects onto Legendre polynomials,
and TimesNet (Wu et al. 2023) uses an FFT to discover dominant periods, reshapes the 1D series
into a 2D tensor indexed by (intra-period, inter-period), and runs 2D convolutions. Both paradigms
disentangle the signal along a single axis — season-vs-trend, or one period-length vs another.

The motivating empirical facts that sit underneath all of this are already well documented.
Coarsening a series by averaging is *not* information-preserving in a benign way: it
systematically removes high-frequency structure and retains low-frequency structure (an average
pool is a low-pass filter). So a downsampling ladder of a series produces views whose dominant
content shifts from microscopic to macroscopic as you climb. In seasonality analysis (Box & Jenkins
1970), larger periods are described through aggregations of smaller periods — a weekly cycle can
be understood through daily cycles — while slow trend analysis treats high-frequency movement as a
disturbance around a lower-frequency level. There is also a documented distribution-shift problem:
the mean and variance of these series drift between training and test windows, so a model fit on
raw values transfers poorly, and reversible per-instance standardization (RevIN; Kim et al. 2022;
the Non-stationary Transformer, Liu et al. 2022) was introduced to absorb it. These are facts
about what averaging, temporal aggregation, and drift do to a signal, knowable before any model.

## Baselines

These are the prior forecasters a new design is measured against and reacts to.

**Autoformer (Wu et al., NeurIPS 2021).** Embeds series decomposition as a module:
`trend = AvgPool1d(x, kernel)` (with edge-replication padding so length is preserved),
`season = x - trend`, and the Transformer processes season and trend with an Auto-Correlation
attention. Core idea: decomposition makes each component more predictable than the raw mixture.
Gap: the decomposition is applied at a *single resolution* — one moving-average kernel on the
original-scale series — so it never sees how the seasonal/trend split changes when you look at the
series more coarsely or more finely.

**DLinear (Zeng et al., AAAI 2023).** Decompose with a moving average into season + trend, then
apply *one linear layer per component* mapping length `P` to length `F`, channel-independently
(each variate handled by its own scalar-channel pipeline). Strikingly strong and cheap. Gap: a
single scale and a single linear map per component; there is no mechanism for information at one
resolution to inform another, so whatever the original sampling rate happens to be is the only
view the model ever forms.

**TimesNet (Wu et al., ICLR 2023).** FFT picks the top-`k` dominant frequencies; for each, the 1D
series is folded into a 2D tensor (period length × number of periods) and processed by Inception
2D convolutions, then the per-frequency outputs are amplitude-weighted and summed. Core idea:
disentangle along the *period* axis and reuse 2D vision backbones. Gap: it disentangles by period
rather than by sampling scale, and the 2D-conv Inception blocks are comparatively heavy; the
period decomposition does not produce the micro-to-macro scale ladder.

**N-HiTS (Challu et al., AAAI 2023).** A pure-MLP successor to N-BEATS that introduces multi-rate
data sampling and hierarchical interpolation across a stack of blocks, each block specializing to
a frequency band via pooling. Core idea: process the series at several rates with cheap MLPs. Gap:
its blocks pass a single residual down the stack and produce the forecast by hierarchically
interpolating and summing block outputs; it does not separate season from trend and route each
across scales with its own direction.

**MLP-Mixer (Tolstikhin et al., 2021).** Not a forecaster, but the relevant primitive: pure-MLP
information integration. It alternates a token-mixing MLP (across spatial positions) with a
channel-mixing MLP (across features), showing that *mixing* — a learned linear map along an axis,
wrapped in a nonlinearity — is enough to integrate information without attention or convolution.
This is the cheap mixing primitive available off the shelf.

**Pyraformer (ICLR 2022) and SCINet (NeurIPS 2022).** Both build *multiscale* temporal
representations: Pyraformer with a pyramidal attention over a coarsening tree, SCINet with a
bifurcate downsampling tree that interleaves and recombines even/odd subsequences. Core idea:
let the model see the series at several resolutions. Gap: although they form multiscale features
internally, the architecture ultimately relies on a re-merged representation for the forecast,
so the separate scale views are not retained as separate forecasting objects at the output
interface.

## Evaluation settings

The Time-Series-Library protocol (Wu et al., ICLR 2023) standardizes splits, per-channel z-score
standardization fit on the training split, and metric computation, so architectural contributions
are comparable. Long-term multivariate benchmarks, all with `features=M` (multivariate in →
multivariate out), look-back `seq_len=96`, horizon `pred_len=96`:

- **ETTh1** — 7 channels, hourly Electricity Transformer Temperature (Zhou et al., AAAI 2021).
- **Weather** — 21 channels, 10-minute meteorological observations.
- **ECL / Electricity** — 321 channels, hourly client electricity consumption.

Optimizer Adam with `(β1,β2)=(0.9,0.999)`; learning rate around `1e-2`; an L2 (MSE) training
loss. Metrics: MSE and MAE over all channels after the inverse standardization, lower is better.
(Settings only — the fixed yardstick, no outcomes.)

## Code framework

The component plugs into a fixed Time-Series-Library harness. A data loader yields four tensors
per window — `x_enc` (`[B, seq_len, enc_in]`, the past), `x_mark_enc` (`[B, seq_len, time_feat]`,
calendar features of the past), and the decoder-side `x_dec` / `x_mark_dec` — and the model must
implement `forecast(...)` returning `[B, pred_len, c_out]`; `forward` slices the last `pred_len`
steps. An Adam optimizer, an MSE loss, and a fixed training loop are provided. The existing
primitives are generic: an `AvgPool1d` (a parameter-free temporal low-pass / downsampler),
`nn.Linear` and `GELU` (the MLP mixing primitive), a moving-average series-decomposition block
that splits a window into `(season = x - movavg, trend = movavg)`, a no-positional-encoding
token (+ time-feature) embedding, and a reversible per-series instance-normalization layer that
standardizes a window on the way in and inverts it on the way out. What is *not* fixed is the
architecture between embedding the past and emitting the future — that is exactly the slot to
design.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class series_decomp(nn.Module):
    """Moving-average decomposition (Autoformer 2021): trend = AvgPool(x), season = x - trend."""
    def __init__(self, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, x):                                   # x: [B, T, C]
        front = x[:, :1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end   = x[:, -1:, :].repeat(1, self.kernel_size // 2, 1)
        xp    = torch.cat([front, x, end], dim=1)
        trend = self.avg(xp.permute(0, 2, 1)).permute(0, 2, 1)
        return x - trend, trend                            # season, trend


class Normalize(nn.Module):
    """RevIN (Kim et al. 2022): per-instance standardize on 'norm', invert on 'denorm'."""
    def __init__(self, num_features, eps=1e-5, affine=True):
        super().__init__()
        self.eps, self.affine = eps, affine
        if affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x, mode):                            # mode in {'norm','denorm'}
        if mode == 'norm':
            self.mean = x.mean(1, keepdim=True).detach()
            self.stdev = torch.sqrt(x.var(1, keepdim=True, unbiased=False) + self.eps).detach()
            x = (x - self.mean) / self.stdev
            if self.affine:
                x = x * self.affine_weight + self.affine_bias
            return x
        else:
            if self.affine:
                x = (x - self.affine_bias) / (self.affine_weight + self.eps * self.eps)
            return x * self.stdev + self.mean


class DataEmbedding_wo_pos(nn.Module):
    """Token (+ optional time-feature) embedding to d_model, NO positional encoding."""
    def __init__(self, c_in, d_model, embed, freq, dropout):
        super().__init__()
        # token conv (c_in -> d_model) + optional temporal embedding; details fixed by the harness
        ...

    def forward(self, x, x_mark):                          # -> [B, T, d_model]
        ...


class Model(nn.Module):
    """The forecasting architecture to design: from the [B, seq_len, enc_in] past
    produce the [B, pred_len, c_out] future. The downsampler, the GELU-MLP mixing
    primitive, the decomposition block, the embedding, and per-series RevIN already
    exist; how to arrange them between past and future is the open slot."""
    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len   = configs.seq_len
        self.pred_len  = configs.pred_len
        self.enc_in    = configs.enc_in
        self.c_out     = configs.c_out
        self.d_model   = configs.d_model
        # TODO: the architecture we will design here.

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # x_enc: [B, seq_len, enc_in]  ->  return [B, pred_len, c_out]
        # TODO: how to build past representations and produce the future.
        pass

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return out[:, -self.pred_len:, :]
```
