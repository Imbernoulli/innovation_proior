## PatchTST reconstructor, distilled

A channel-independent, patch-tokenized vanilla Transformer that reconstructs the input window. Each
univariate series is instance-normalized, cut into overlapping sub-series patches (each a local shape
token), encoded by a plain BatchNorm Transformer over the patches, and decoded by a flatten-and-project
head back to the same `seq_len` window. The reconstruction MSE is the anomaly score.

## Problem it solves

Unsupervised reconstruction-based anomaly detection: map a normal window `[B, seq_len, C]` to a
same-length reconstruction, trained by MSE on normal windows. A point-wise Transformer reconstructor
tokenizes single time steps — meaningless in isolation, and on a mostly-normal window the attention is
dominated by normal points, washing out the rare abnormal pattern the score must keep sharp.

## Key idea

- **Patch the token.** A single step has no standalone meaning and neighboring steps are redundant, so
  group a contiguous stretch of `P` steps (stride `S`) into one token — a local shape attention can use.
  Token count drops from `L` to `≈ L/S`, attention cost by `≈ S²`. Pad the end by repeating the last
  value `S` times so the window tail is covered; `N = floor((L − P)/S) + 2 = 12` at `L=100, P=16, S=8`.
- **Channel-independent, weight-shared.** Run one backbone independently on each univariate series (fold
  channels into the batch axis) for per-series adaptability, data efficiency, and less overfitting; the
  shared weights serve any channel count (25 / 55 / 25).
- **Vanilla encoder, BatchNorm.** Keep the attention kernel plain (the input representation was the fix,
  not the kernel); use BatchNorm not LayerNorm so an outlier patch is diluted across the batch rather
  than corrupting its own per-token statistics.
- **Reconstruction head.** Flatten the `[D, N]` per-series representation and linear-project to
  `seq_len`; no decoder, no horizon. Reversible per-instance normalization wraps the whole thing.

## Why it should help over identity

Identity gives a flat, useless score. A patch Transformer reconstructs the smooth, near-periodic normal
structure well (high F1 on server metrics), and the reconstruction error spikes where the local shape
breaks. Channel-independence keeps the heterogeneous telemetry channels from contaminating one another.

## Hyperparameters

`patch_len=16`, `stride=8`, `d_model=512`, `d_ff=512`, `e_layers=2`, `n_heads=8`, `dropout=0.1`,
`seq_len=100`, `enc_in=c_out=C`; Adam `lr=1e-4`, `batch_size=32`, `train_epochs=3`. Encoder uses
`FullAttention` (non-causal) and BatchNorm over `d_model`.

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
        self.flatten = nn.Flatten(start_dim=-2)        # (D, N) -> (D*N)
        self.linear = nn.Linear(nf, target_window)      # (D*N) -> seq_len
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):                               # x: [bs, nvars, D, N]
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
        padding = stride                                # pad S copies of the last value (one extra patch)

        # patch + linear-project each patch to d_model (no bias) + positional embedding
        self.patch_embedding = PatchEmbedding(
            configs.d_model, patch_len, stride, padding, configs.dropout)

        # vanilla Transformer encoder over the N patch tokens; BatchNorm (not LayerNorm)
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor,
                                      attention_dropout=configs.dropout,
                                      output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=nn.Sequential(Transpose(1, 2),
                                     nn.BatchNorm1d(configs.d_model),
                                     Transpose(1, 2))
        )

        # head input width = d_model * number of patches N; N = (L - P)/S + 2
        self.head_nf = configs.d_model * int((configs.seq_len - patch_len) / stride + 2)
        # reconstruction: project the encoded patches back to the seq_len window
        self.head = FlattenHead(configs.enc_in, self.head_nf, configs.seq_len,
                                head_dropout=configs.dropout)

    def anomaly_detection(self, x_enc):
        # reversible per-instance normalization against distribution shift
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc / stdev

        # channel-independence: channels on dim 1, folded into the batch inside patch_embedding
        x_enc = x_enc.permute(0, 2, 1)                  # [bs, nvars, L]
        enc_out, n_vars = self.patch_embedding(x_enc)   # [bs*nvars, N, D]

        # vanilla encoder over patch tokens
        enc_out, attns = self.encoder(enc_out)          # [bs*nvars, N, D]
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))  # [bs, nvars, N, D]
        enc_out = enc_out.permute(0, 1, 3, 2)           # [bs, nvars, D, N]

        # reconstruction head -> seq_len, then back to [bs, seq_len, nvars]
        dec_out = self.head(enc_out)                    # [bs, nvars, seq_len]
        dec_out = dec_out.permute(0, 2, 1)              # [bs, seq_len, nvars]

        # de-normalize with the same window statistics
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'anomaly_detection':
            return self.anomaly_detection(x_enc)
        return None
```
