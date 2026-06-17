# PatchTST, distilled

PatchTST is a Transformer forecaster built on two ideas: **patching** — segment each
univariate series into subseries-level patches and feed those patches (not single time steps)
as the input tokens — and **channel-independence** — process every channel as its own
univariate series through one shared Transformer backbone, with no cross-channel mixing. The
backbone is a deliberately vanilla Transformer encoder; the contribution is the input
representation, not a new attention kernel. The result is competitive with — and on the
standard benchmarks beats — both the prior point-wise Transformers and the strong linear
baseline, while keeping the multi-layer representation a linear model cannot offer.

## Problem it solves

Forecast the next `T` values of a series from a length-`L` look-back, where (a) point-wise
attention tokenizes each step and so attends over units with no local meaning and pays
`O(L^2)` cost that forbids a long history, and (b) channel-mixing entangles all channels under
one attention pattern and overfits the modest datasets. For the short, univariate, many-series
M4 setting this specializes to one channel per series (`enc_in = c_out = 1`), trained on the
SMAPE metric.

## Key ideas

**Patching.** Split each univariate series `x in R^L` into patches of length `P` with stride
`S`. After padding `S` copies of the last value to the end, the number of patches is
`N = floor((L - P)/S) + 2`. This (1) makes each token a local *shape* attention can compare;
(2) cuts the token count from `L` to `~L/S`, so the `O(N^2)` attention cost drops by `~S^2`;
(3) makes a longer, more informative look-back affordable. Defaults `P=16, S=8` (half-overlap)
are robust across patch-length sweeps; e.g. `L=336 -> N=42`, `L=512 -> N=64`.

**Channel-independence.** Split the `M`-channel input into `M` univariate series and run each
through the *same* shared backbone independently. More preferable than channel-mixing because:
*adaptability* (each series gets its own attention map rather than one shared compromise);
*data efficiency* (only temporal structure is learned, not joint cross-channel interactions
that need much more data); *less overfitting* (mixing fits spurious cross-channel coincidences
and overfits after a few epochs). Implemented for free by folding the channel axis into the
batch axis: `[B, M, L] -> [B*M, N, P]`, run the encoder, reshape back. The weights are shared,
so the channel count at train time need not match test time — the univariate M4 case is the
native one (`M = 1`).

**Vanilla encoder over patches.** Linear-project each patch to `D` with `W_p in R^{D x P}` (no
bias), add a positional embedding (patches are otherwise an unordered set), then a standard
multi-head scaled dot-product encoder
`Attention(Q,K,V) = softmax(Q K^T / sqrt(d_k)) V` with a position-wise FFN `D -> F -> D` and
residuals. Normalization is **BatchNorm**, not LayerNorm: outlier time steps corrupt per-token
statistics, while BatchNorm dilutes a single outlier patch across the batch.

**Reversible instance normalization.** Before patching, per instance and per channel subtract
the look-back mean and divide by `sqrt(var + eps)` (biased variance, `eps = 1e-5`, statistics
detached); add the mean and scale back to the forecast. Decouples shape-learning from
level-tracking and counters train/test distribution shift.

**Head.** Flatten the `D x N` encoder output per series and project with one linear layer to
the horizon `T` (`head_nf = D * N`). This per-series shared head avoids the oversized
`(L*D) x (M*T)` joint head that a channel-mixing model needs and that overfits.

**Loss.** MSE per channel averaged over channels for the long-horizon benchmarks;
**SMAPE** for M4, `(200/T) * sum_t |y_t - yhat_t| / (|y_t| + |yhat_t|)`, training on the
percentage metric so large-magnitude series do not dominate. M4 config is small (a couple of
encoder layers, few heads, small `d_model`, Adam `lr=1e-3`, batch 16, ~10 epochs) because the
short single-variable series would otherwise overfit.

## Final architecture (forward pass)

```
x in R^{B x L x M}
  -> instance-normalize per (instance, channel): subtract mean, divide by sqrt(var+eps)
  -> permute to [B, M, L]
  -> pad S copies of last value; unfold into patches -> [B, M, N, P]; fold channels -> [B*M, N, P]
  -> linear patch embed W_p (R^P -> R^D, no bias) + positional embed -> [B*M, N, D]
  -> e_layers vanilla Transformer encoder layers (MHA + FFN + residual, BatchNorm) -> [B*M, N, D]
  -> reshape -> [B, M, N, D] -> permute -> [B, M, D, N]
  -> flatten head: (D*N) -> T  -> [B, M, T] -> permute -> [B, T, M]
  -> de-normalize: multiply by stdev, add mean
  -> forecast in R^{B x T x M}
```

## Self-supervised variant

Same encoder, head swapped for `D -> P`. Mask whole *non-overlapping* patches at random and
reconstruct them by MSE. Masking whole patches (not single steps, which are trivially
interpolable from neighbors) forces whole-signal understanding, so the representation
transfers; the shared channel-independent weights allow pretrain/fine-tune channel counts to
differ.

## Working code

The canonical Time-Series-Library implementation (what the short-term M4 run uses), filling
the harness `Model` slot. The encoder/attention layers are the library's reusable primitives.

```python
import torch
from torch import nn
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import PatchEmbedding


class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False):
        super().__init__()
        self.dims, self.contiguous = dims, contiguous

    def forward(self, x):
        if self.contiguous:
            return x.transpose(*self.dims).contiguous()
        return x.transpose(*self.dims)


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs, nvars, d_model, patch_num]
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class Model(nn.Module):
    def __init__(self, configs, patch_len=16, stride=8):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        padding = stride

        # patching + per-patch linear embedding + positional embedding
        self.patch_embedding = PatchEmbedding(
            configs.d_model, patch_len, stride, padding, configs.dropout)

        # vanilla Transformer encoder over patch tokens, BatchNorm normalization
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(configs.d_model), Transpose(1, 2))
        )

        # prediction head: width = d_model * number-of-patches
        self.head_nf = configs.d_model * int((configs.seq_len - patch_len) / stride + 2)
        if self.task_name in ('long_term_forecast', 'short_term_forecast'):
            self.head = FlattenHead(configs.enc_in, self.head_nf, configs.pred_len,
                                    head_dropout=configs.dropout)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # reversible per-instance normalization
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x_enc /= stdev

        # channel-independent patching: channels folded into the batch axis
        x_enc = x_enc.permute(0, 2, 1)                  # [bs, nvars, L]
        enc_out, n_vars = self.patch_embedding(x_enc)   # [bs*nvars, patch_num, d_model]

        enc_out, attns = self.encoder(enc_out)          # [bs*nvars, patch_num, d_model]
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))  # [bs, nvars, patch_num, d_model]
        enc_out = enc_out.permute(0, 1, 3, 2)           # [bs, nvars, d_model, patch_num]

        dec_out = self.head(enc_out)                    # [bs, nvars, T]
        dec_out = dec_out.permute(0, 2, 1)              # [bs, T, nvars]

        # de-normalization
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in ('long_term_forecast', 'short_term_forecast'):
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]       # [bs, T, nvars]
        return None
```

Patch embedding (pad last value, unfold into patches, fold channels into batch, linear-embed,
add positional embedding):

```python
class PatchEmbedding(nn.Module):
    def __init__(self, d_model, patch_len, stride, padding, dropout):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = nn.ReplicationPad1d((0, padding))
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)   # W_p
        self.position_embedding = PositionalEmbedding(d_model)             # W_pos
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):                                # x: [bs, nvars, L]
        n_vars = x.shape[1]
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)  # [bs, nvars, patch_num, patch_len]
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x), n_vars
```

For M4 short-term forecasting the harness sets `task_name='short_term_forecast'`,
`enc_in=c_out=1`, and the training loss to SMAPE; the architecture above is unchanged.
