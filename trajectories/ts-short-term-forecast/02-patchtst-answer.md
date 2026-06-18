**Problem.** The linear floor (DLinear: SMAPE 13.39 / 10.50 / 14.36) is honest but structurally
capped — an affine, un-normalized direct-multi-step map cannot model any nonlinear shape and cannot
decouple a series' shape from its per-window level. The goal is to keep the linear model's robustness
on these short M4 series while adding a learned nonlinear representation and per-instance
normalization, without reintroducing the overfitting that sank the heavy attention forecasters.

**Key idea.** Attention failed on forecasting because the *token* was a single time step (meaningless
in isolation), not because attention is wrong. Replace the token: cut each univariate series into
overlapping sub-series **patches** (`patch_len=16`, `stride=8`) — each patch is a local shape attention
can compare — and run a deliberately *vanilla* Transformer encoder over the patch tokens. Process each
channel **independently** through one shared backbone (channels folded into the batch axis); on M4 the
univariate case is native (`enc_in=c_out=1`). Wrap everything in **reversible instance normalization**
(subtract look-back mean, divide by `√(var+1e-5)`, denormalize the forecast) — the exact level-
decoupling DLinear lacked.

**Why.** Patch tokens give attention real shape-objects instead of meaningless scalars; on M4 the
short window makes the cost/look-back benefits of patching moot, so this benefit is the load-bearing
one. Channel-independence with shared weights matches DLinear's winning channel treatment while keeping
the Transformer's nonlinear depth. Instance normalization is the cheap, consistent gain that directly
attacks the per-series level drift the affine floor could not handle. BatchNorm (not LayerNorm) in the
encoder dilutes outlier patches across the batch instead of letting one spike corrupt its own token's
statistics.

**Hyperparameters.** `patch_len=16`, `stride=8`, padding `=stride`; number of patches
`N = int((seq_len − patch_len)/stride + 2)` (the replication-pad path keeps `N ≥ 1` even when
`seq_len < patch_len`, as on Yearly); flatten head width `head_nf = d_model · N`. Under the fixed
Custom protocol `d_model=512`, `d_ff=512`, `e_layers=2`, `n_heads=8`, `dropout=0.1`, `factor` and
`activation` from configs; Adam `lr=1e-3`, batch 16, 10 epochs, patience 3, SMAPE loss. (The 512-wide
encoder over 2–4 patch tokens is far wider than PatchTST's own M4 script — the instance norm, shared
channel-independent weights, and early stopping are the regularizers holding that capacity in check.)

**What to watch.** Should beat the linear floor on every regime, with the largest absolute gain on
Yearly (where DLinear was most strained). If the gain over DLinear is only marginal, that signals
attention over so few short-window patches is near its ceiling — and that the next rung should exploit
multi-scale structure rather than add generic capacity.

```python
import torch
from torch import nn
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import PatchEmbedding


class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False):
        super(Transpose, self).__init__()
        self.dims, self.contiguous = dims, contiguous

    def forward(self, x):
        if self.contiguous:
            return x.transpose(*self.dims).contiguous()
        return x.transpose(*self.dims)


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super(FlattenHead, self).__init__()
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
    """PatchTST: patch tokens + vanilla channel-independent Transformer + RevIN."""

    def __init__(self, configs, patch_len=16, stride=8):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.c_out = configs.c_out
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
                    activation=configs.activation,
                ) for l in range(configs.e_layers)
            ],
            norm_layer=nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(configs.d_model), Transpose(1, 2)),
        )

        # prediction head: width = d_model * number-of-patches
        self.head_nf = configs.d_model * int((configs.seq_len - patch_len) / stride + 2)
        self.head = FlattenHead(configs.enc_in, self.head_nf, configs.pred_len,
                                head_dropout=configs.dropout)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # reversible per-instance normalization
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x_enc = x_enc / stdev

        # channel-independent patching: channels folded into the batch axis
        x_enc = x_enc.permute(0, 2, 1)                  # [bs, nvars, L]
        enc_out, n_vars = self.patch_embedding(x_enc)   # [bs*nvars, patch_num, d_model]

        enc_out, attns = self.encoder(enc_out)          # [bs*nvars, patch_num, d_model]
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        enc_out = enc_out.permute(0, 1, 3, 2)           # [bs, nvars, d_model, patch_num]

        dec_out = self.head(enc_out)                    # [bs, nvars, T]
        dec_out = dec_out.permute(0, 2, 1)              # [bs, T, nvars]

        # de-normalization
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]       # [bs, T, nvars]
        return None
```
