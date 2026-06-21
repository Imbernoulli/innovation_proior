# Context: Transformer-based time series forecasting (circa 2021-2022)

## Research question

A time series presents, at each step `t`, a value (or for the multivariate case an
`M`-channel vector `x_t in R^M`), observed over a look-back window of length `L`. The task is
to predict the next `T` values `(x_{L+1}, ..., x_{L+T})`, often hundreds of steps ahead, while
also learning representations that can transfer to downstream time-series tasks. The
Transformer is the obvious tool — its attention learns relations between sequence elements with
no fixed locality assumption, and it powers the strongest sequence models in language, vision,
and speech. The question is how to best tokenize and route a raw time-series window through a
Transformer encoder so that attention operates over meaningful units, the look-back can be
kept long, and the resulting representations are useful beyond a single trained head.

## Background

By this point the Transformer (Vaswani et al. 2017) — multi-head scaled dot-product
self-attention `Attention(Q,K,V) = softmax(Q K^T / sqrt(d_k)) V`, stacked with position-wise
feed-forward blocks and residual connections — is the default sequence model, and has been
carried into time series. The standard recipe embeds **each time step** as one token: the
value (or the `M`-vector) at step `t` is linearly projected to a `D`-dimensional token, and
self-attention runs over the `L` step-tokens. Several considerations have shaped the field's
assumptions:

- **Attention cost scales with sequence length.** Vanilla attention is `O(n^2)` in time
  and memory for `n` tokens. With one token per step, `n = L`, so doubling the look-back
  quadruples the cost. This is the reason the efficient-Transformer line spent years
  sparsifying or low-ranking the attention matrix to make a long `L` affordable.

- **Local structure in time series.** A single value `x_t` in isolation carries limited
  information; the meaningful structure (a rising edge, a daily bump, a spike-and-decay)
  lives in a neighborhood of steps. Time series are also heavily temporally redundant:
  adjacent points are nearly equal, so consecutive step values are strongly correlated.

- **Grouping into patches elsewhere.** In vision, the Vision Transformer (Dosovitskiy et al.
  2021) cuts an image into 16x16 patches and linearly projects each patch to a token — this
  is what made Transformers work on images; BEiT and masked autoencoders (He et al. 2022)
  build masked pre-training on those patches. Sub-word tokens in language (BERT/WordPiece) and
  convolutional sub-sequence features in speech (wav2vec) are the same idea. The common thread
  is that the *unit of attention* already carries local semantics.

- **Distribution shift.** These series are non-stationary: the mean and scale of a window
  drift over time, so test windows sit at levels the training windows never visited.
  Instance-level normalization — normalize each input window to zero mean and unit variance,
  predict, then map the prediction back — was proposed to neutralize this drift (the operator
  from Ulyanov et al. 2016; the reversible time-series form from Kim et al. 2022, RevIN).

- **Normalization inside the encoder.** For Transformers trained on time series, batch
  normalization was found to outperform the layer normalization that language Transformers use
  (Zerveas et al. 2021): time series carry outlier timestamps with extreme values that a
  per-token layer-norm cannot temper, whereas a batch-norm over the batch/time dimension can.

A finding from the linear-model work (Zeng et al. 2023) is that complex Transformer baselines
do not reliably improve when given a longer window — their error stays flat or rises as `L`
grows — while the simple linear model can keep improving with longer `L`.

## Baselines

**Linear forecaster (Zeng et al., AAAI 2023; "Are Transformers Effective for Time Series
Forecasting?").** Decompose each univariate series into trend (moving average) and seasonal
(remainder), apply one shared linear layer per component mapping the length-`L` look-back
directly to the length-`T` horizon, and sum. It is **channel-independent**: the same linear
map is applied to every channel separately, with no cross-channel mixing. On the standard
benchmarks this matches or beats every published Transformer.

**Point-wise efficient Transformers — Informer (Zhou et al., AAAI 2021), Autoformer (Wu et
al., NeurIPS 2021), FEDformer (Zhou et al., ICML 2022), Pyraformer (Liu et al., ICLR 2022).**
All embed one time step as one token and attack the `O(L^2)` cost: Informer with ProbSparse
attention that keeps only dominant query-key pairs plus a distilling step that halves the
sequence; Autoformer with a series-decomposition block plus an auto-correlation mechanism that
aggregates over period-shifted subseries; FEDformer with attention in the frequency domain on
a few selected modes; Pyraformer with a pyramidal attention of inter- and intra-scale edges.
They use an encoder-decoder that emits the whole horizon at once.

**Patch-flavoured attempts — LogTrans (Li et al., NeurIPS 2019), Triformer (Cirstea et al.,
IJCAI 2022).** LogTrans replaces the point-wise dot product with a causal-convolution-derived
query/key so the score depends on a local neighborhood, but its *value* is still a single time
step. Triformer introduces a patch attention, using a pseudo-timestamp as the query within a
patch to cut complexity.

**Masked time-series Transformer (Zerveas et al., KDD 2021; "A Transformer-based Framework for
Multivariate Time Series Representation Learning", TST).** A Transformer for representation
learning whose input token is the per-step vector `x_t`; it pre-trains by masking values
at randomly chosen **single time steps** and reconstructing them, and it established that
batch-norm beats layer-norm here.

**Channel-independence precedents (Zheng et al. 2014 multichannel CNN; the linear forecaster
above).** Treating each channel as its own univariate signal with a shared model worked for
CNNs and for the linear model.

## Evaluation settings

- **Long-term multivariate benchmark (Autoformer release, Wu et al. 2021).** Eight datasets:
  Weather (21 channels), Traffic (862), Electricity (321), ILI (7), and the four ETT variants
  (ETTh1, ETTh2, ETTm1, ETTm2; 7 channels each). Horizons `T in {96, 192, 336, 720}` (and
  `{24, 36, 48, 60}` for the small ILI). Metrics MSE and MAE on the standard
  train/val/test split. Look-back `L` is a knob; a fair comparison fixes the same `L` across
  models.
- **Representation-learning protocol.** Self-supervised pre-train on the unlabelled series,
  then either linear-probe (train only the head) or end-to-end fine-tune, and forecast.

## Code framework

The model plugs into a fixed forecasting harness (a `Time-Series-Library`-style pipeline): a
data loader that yields standardized sliding windows `x_enc in [B, L, M]` with their calendar
marks and a decoder placeholder; an Adam optimizer; a horizon loss such as MSE; and a training
loop with validation-based early stopping. The
encoder/attention primitives already exist as reusable layers — a generic `EncoderLayer`
(multi-head attention + position-wise feed-forward + residual + a configurable normalization),
a `FullAttention` scaled-dot-product block, and an `AttentionLayer` that wraps the QKV
projections. What is **not** decided is how a raw window should be turned into tokens, what the
tokens should be, how the channels should be routed through the encoder, and how the encoded
representation becomes a length-`T` forecast. That is the single empty slot: the `Model` class.

```python
import torch
from torch import nn
# existing reusable encoder primitives in the library
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer


class Model(nn.Module):
    """Forecasting model for the fixed harness.

    configs carries: task_name (e.g. 'long_term_forecast'), seq_len (L),
    pred_len (T), enc_in / c_out (number of channels M),
    d_model (D), d_ff (F), e_layers, n_heads (H), dropout, activation, factor.

    forecast() receives x_enc [B, L, M] and must return [B, T, M].
    """

    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        # TODO: how to tokenize the window, route the channels through the
        #       encoder, normalize against distribution shift, and map the
        #       encoded representation to a T-step forecast — the design we will build.
        pass

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # x_enc: [B, L, M]  ->  return [B, T, M]
        # TODO: the forward computation we will design.
        pass

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return out[:, -self.pred_len:, :]


# existing training loop the model plugs into
def train(model, loss_fn, data_loader, optimizer):
    for x_enc, x_mark_enc, x_dec, x_mark_dec, target in data_loader:
        optimizer.zero_grad()
        out = model(x_enc, x_mark_enc, x_dec, x_mark_dec)   # [B, T, M]
        loss = loss_fn(out, target)                          # e.g. MSE on the forecast horizon
        loss.backward()
        optimizer.step()
```

The `TODO`s — tokenization, channel routing, normalization, and the horizon head — are
exactly the design slots that remain open.
