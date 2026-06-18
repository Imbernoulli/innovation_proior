**Problem (from step 1).** DLinear is loose on Weather and ECL (no exogenous fusion) but also leaves a
residual on ETTh1, where fusion barely matters — so the *per-channel temporal model* itself is a
bottleneck. Upgrade that, holding channel-independence fixed, so the next gap can be cleanly blamed on
fusion.

**Key idea.** Stop tokenizing per time step. Cut each channel's look-back into contiguous **patches**
(`patch_len=16`, `stride=8`), embed each patch as a token, and run a vanilla Transformer encoder over
the patch tokens, then a flatten-linear head to the 96-step horizon. A patch is a meaningful local
shape; point-wise attention compared meaningless scalars. The backbone is **channel-independent** —
the same patch-embed/encoder/head per channel, channels never attend — by folding the channel axis into
the batch.

**Why it works.** Patching fixes the tokenization failure that let a plain linear map beat the
point-wise Transformers: each token is now a local ramp/bump/level, attention compares shapes, and the
token count collapses so a long look-back is cheap. This gives a strong nonlinear, multi-scale read of
one channel's own dynamics — strictly more than DLinear's affine map — without touching cross-channel
structure.

**Scaffold edit / hyperparameters.** Per-instance normalization (Non-stationary style) in/out;
`PatchEmbedding(d_model, patch_len=16, stride=8, padding=stride)`; `e_layers=3`, `n_heads=4`,
`d_model=128`, `d_ff=256`, dropout 0.1; transpose-BatchNorm-transpose encoder norm; `FlattenHead` over
`d_model * patch_num` to `pred_len`. Adam `lr=1e-4`, 10 epochs, batch 32 (fixed by the loop).

**What to watch.** Should beat DLinear on all three (better temporal model) — ETTh1 into the high
0.05s is pure evidence the per-channel model was the bottleneck. On Weather and ECL it should improve
but leave a **residual gap** that only cross-variate fusion can close; that gap is the target for the
next rung.

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

    def forward(self, x):  # [bs, nvars, d_model, patch_num]
        x = self.flatten(x)
        x = self.linear(x)
        return self.dropout(x)


class Model(nn.Module):
    """Channel-independent patch Transformer (PatchTST)."""

    def __init__(self, configs, patch_len=16, stride=8):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        padding = stride

        self.patch_embedding = PatchEmbedding(
            configs.d_model, patch_len, stride, padding, configs.dropout)

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False), configs.d_model, configs.n_heads),
                    configs.d_model, configs.d_ff,
                    dropout=configs.dropout, activation=configs.activation,
                ) for _ in range(configs.e_layers)
            ],
            norm_layer=nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(configs.d_model), Transpose(1, 2)),
        )

        self.head_nf = configs.d_model * int((configs.seq_len - patch_len) / stride + 2)
        self.head = FlattenHead(configs.enc_in, self.head_nf, configs.pred_len,
                                head_dropout=configs.dropout)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # per-instance normalization
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc / stdev

        # patch + embed (channel folded into batch -> channel-independent)
        x_enc = x_enc.permute(0, 2, 1)
        enc_out, n_vars = self.patch_embedding(x_enc)        # [bs*nvars, patch_num, d_model]
        enc_out, _ = self.encoder(enc_out)
        enc_out = torch.reshape(enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        enc_out = enc_out.permute(0, 1, 3, 2)               # [bs, nvars, d_model, patch_num]

        dec_out = self.head(enc_out).permute(0, 2, 1)        # [bs, pred_len, nvars]

        # de-normalization
        dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        dec_out = dec_out + means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in ('long_term_forecast', 'short_term_forecast'):
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]
        return None
```
