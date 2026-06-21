# Context

## Research question

Long-term multivariate forecasting asks for the next `S` steps of *every* channel of a
multivariate series given a fixed look-back window. The channels of a real multivariate system
play genuinely different roles relative to any one prediction: when forecasting electricity price,
grid load and wind-power forecasts are the causal levers; when forecasting a building's CO₂
concentration, the other meteorological readings drive it. For the channel currently being
predicted — call it the **target** — the other channels are **side** channels: observed information
that may sharpen *its* forecast. So a method that scores well on the whole multivariate task still
has to respect, per channel, the asymmetry between the one series whose exact future trajectory
matters and the others that are only there to inform it.

The concrete problem: for each channel, given its look-back `x_{1:T} ∈ R^T` and the other
channels as side series `z_{1:T_ex} ∈ R^{T_ex × C}` (in real settings possibly with `T_ex ≠ T`,
different sampling, missing values, temporal misalignment), predict the next `S` steps of that
channel, `x̂_{T+1:T+S} = F_θ(x_{1:T}, z_{1:T_ex})`, and do so for all channels at once, inside
a fixed look-back/horizon and a standard forecasting pipeline.

## Background

**Deep forecasting and the attention axis.** Transformer-based forecasters differ mainly in *what
a token is* and therefore *which axis attention runs over*. Point-wise models (Informer, Zhou et
al. 2021; Autoformer, Wu et al. 2021) make each timestamp a token and run attention over time, with
efficiency variants to fight the quadratic cost. Patch-wise models group consecutive steps into a
patch token to recover local semantics that a single time point lacks. Variate-wise models make a
whole series one token and run attention across channels. Each granularity captures a different kind
of dependency, and a recurring empirical fact about temporal-attention forecasters is that
extending the look-back window does not reliably help and often hurts — attention spreads thin over
ever more temporal tokens.

**Channel independence vs. cross-variate modeling.** A second axis of the field is whether channels
interact. Channel-independent backbones (one shared model per series, no cross-channel mixing) are
strong, well-regularized on small datasets, and transfer across a changing number of channels. Cross-variate models do mix channels, at the cost of either new attention machinery or quadratic cost in the number of channels.

**The learnable aggregator token.** In vision Transformers (Dosovitskiy et al. 2020) a single
learnable `[class]` token is prepended to the patch sequence; through self-attention it is forced to
collect information from the whole image and serves as the global descriptor for the downstream
head. It is a differentiable parameter, not a function of any one patch — a learned summary node
that lives inside the same attention as the patches.

**Cross-attention as a fusion primitive.** In multi-modal learning (e.g. ALIGN-style fusion, Li et
al. 2021) cross-attention lets one stream query another: queries come from stream A, keys and values
from stream B, so A adaptively reads from B without B reading back. This is the standard way to push
information *one direction* between two heterogeneous token sets.

**Normalization for non-stationarity.** Forecasting series drift in mean and scale across the
window. The Non-stationary-Transformer / RevIN recipe normalizes each series over its own time axis
before the model (subtract the per-series mean, divide by the per-series standard deviation, both
computed over time and detached) and de-normalizes the prediction with the same statistics, so the
network operates in a stationary, scale-free space and the real scale is restored at the end.

**Classical exogenous modeling.** Statistical models long handled side information: ARIMAX / SARIMAX
extend ARIMA with regressors. Deep covariate models (Temporal Fusion Transformer, Lim et al. 2021;
NBEATSx, Olivares et al. 2023; TiDE, Das et al. 2023) incorporate covariates, typically by
concatenating exogenous features onto endogenous features *at each time step* and projecting jointly.
That concatenation forces the endogenous and exogenous series to be aligned in time, equal in
length, and equally sampled.

## Baselines

**DLinear (Zeng et al. 2023, arXiv 2205.13504).** Decompose each series into a moving-average trend
and a remainder, apply one linear map per component from the `T`-step past directly to the `S`-step
future, sum them. Channel-independent, no attention. It matches or beats elaborate temporal-attention
Transformers on long-horizon benchmarks, which establishes that a linear map from representation to
horizon is sufficient for the generation step.

**PatchTST (Nie et al. 2023, arXiv 2211.14730).** Split each series into subseries-level patches,
embed each patch as a token, and run self-attention over the patch tokens, with a channel-independent
backbone shared across series. Patching gives a token a wider receptive field than a single instant
and recovers local temporal semantics, and channel independence regularizes well.

**iTransformer (Liu et al. 2024, arXiv 2310.06625).** Invert the token: embed each variate's whole
look-back as a single token via a linear `R^T → R^D` map, giving `C` tokens, and reuse the stock
encoder so self-attention runs *across* variate tokens (cross-variate correlation) and the FFN runs
*within* each token.

**Crossformer (Zhang et al. 2022).** Patch every series and run a two-stage attention across both
time and variate dimensions, modeling cross-variate structure by redesigning the attention layer and
surrounding architecture.

## Evaluation settings

- **ETTh1** — Electricity Transformer Temperature, hourly (Zhou et al. 2021). 7 channels (oil
  temperature plus six power-load features).
- **Weather** — meteorological observations recorded every 10 minutes. 21 channels.
- **ECL** — hourly electricity consumption of 321 clients. 321 channels.

Protocol fixed by the Time-Series-Library loaders: `features=M` (multivariate input →
multivariate output — every channel is predicted), `seq_len = 96`, `label_len = 48`,
`pred_len = 96`. Per-channel standardization, fixed train/val/test splits. Optimization with
Adam, learning rate `1e-4`, L2 loss, up to 10 epochs with early stopping (patience 3), batch
sizes per dataset. Metrics are MSE and MAE over all channels and the whole horizon; lower is
better.

## Code framework

The pipeline already provides the data loaders, the optimizer/loss, the standard attention and
embedding primitives, and the training loop. What must be filled in is one model class with a fixed
interface: `__init__(configs)`, forecasting routines for single-target and multivariate settings,
and a thin `forward(...)` that selects the right path and slices the horizon. Output is
`[batch, pred_len, c_out]`; in multivariate mode `c_out == enc_in`, so every channel is predicted and
scored.

Pre-existing primitives that can be reused unchanged:

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEmbedding(nn.Module):
    """Fixed sinusoidal positions, returns pe[:, :L] for a length-L token sequence."""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class FullAttention(nn.Module):
    """Scaled dot-product attention (unmasked here); returns (output, optional weights)."""
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1,
                 output_attention=False):
        super().__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, _ = values.shape
        scale = self.scale or 1.0 / math.sqrt(E)
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)
        return V.contiguous(), None


class AttentionLayer(nn.Module):
    """Multi-head wrapper: project Q,K,V into heads, run inner attention, project out."""
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None):
        super().__init__()
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)
        out, attn = self.inner_attention(queries, keys, values, attn_mask, tau=tau, delta=delta)
        return self.out_projection(out.view(B, L, -1)), attn


class Model(nn.Module):
    """Forecasting shell. The architecture that maps the provided history to the requested
    horizon is the open slot."""

    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.features = configs.features
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.c_out = configs.c_out
        # TODO: define the embedding(s), encoder, and head for the forecasting slot.

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Single-target convention: the target channel is selected by the model,
        # and the remaining channels/time marks may be used as side information.
        # TODO: implement the single-target forecasting computation.
        raise NotImplementedError

    def forecast_multi(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Multivariate convention: x_enc is [batch, seq_len, enc_in] and the
        # result is [batch, pred_len, enc_in].
        # TODO: implement the all-channel forecasting computation.
        raise NotImplementedError

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in ("long_term_forecast", "short_term_forecast"):
            if self.features == "M":
                dec_out = self.forecast_multi(x_enc, x_mark_enc, x_dec, x_mark_dec)
            else:
                dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]
        return None
```

The training loop, the Adam optimizer, the L2 objective, and the `features=M` multivariate I/O all
sit outside this file; the task is to fill the one architectural slot.
