**Problem (from step 4).** Four rungs split the two levers: TimeMixer owns ETTh1/Weather (intra-series
temporal detail, channel-independent) but ECL (0.1561) still trails iTransformer (0.1482, the only
channel-modeling rung, which itself lost ETTh1). No model has held fine intra-target temporal structure
*and* cross-variate correlation while keeping both virtues. The fix must respect a per-channel
asymmetry: a side channel should *inform* the target's forecast without overwriting its temporal detail
or leaking noise, and without paying O(C²) to model interactions into channels never predicted.

**Key idea.** Use different granularities for the target and the exogenous channels. Endogenous path:
patch the target finely (PatchTST's intra-target temporal modeling) plus one learnable **global token**;
self-attention runs within the target's patch+global tokens only. Exogenous path: embed each channel's
whole look-back as one variate token (iTransformer's coarse cross-variate token). Cross: the target's
global token — and only it — cross-attends to the exogenous tokens. Direct multi-step flatten head.

**Why it works.** Cross-variate information enters through one narrow gate (the global token), so the
patch tokens carrying the target's fine temporal detail are never overwritten, side-channel noise passes
through a single learned bottleneck, and the cross-attention costs O(patch_num·C), linear in channels —
iTransformer's correlation without its quadratic cost or detail loss. It is the first rung to hold both
levers.

**Hyperparameters / edit-surface notes.** `features='M'` → symmetric-multivariate path (`forecast_multi`):
every channel is endogenous and the full variate set is the exogenous pool. `patch_len` from `configs`
(default 16 → 6 non-overlapping patches at L=96), `use_norm` from `configs` (default 1). `EnEmbedding`
(local; target patches + per-channel global token + `PositionalEmbedding`), dual-attention
`EncoderLayer`, `FlattenHead`, `DataEmbedding_inverted`, `FullAttention`/`AttentionLayer` reused from the
scaffold layers. Fixed scaffold config `d_model=512`, `e_layers=2`, `batch_size=32`, `lr=1e-4` — note the
method's own script uses `d_model=256`, `batch_size=4`, `e_layers=1` on ETTh1/Weather, so the fixed 512
over-parameterizes the small datasets and the large batch is a regime mismatch.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted, PositionalEmbedding


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):                                       # x: [B, n_vars, d_model, patch_num+1]
        return self.dropout(self.linear(self.flatten(x)))


class EnEmbedding(nn.Module):
    """Target -> patch tokens (+positions) and one learnable global token per channel."""
    def __init__(self, n_vars, d_model, patch_len, dropout):
        super().__init__()
        self.patch_len = patch_len
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
        self.glb_token = nn.Parameter(torch.randn(1, n_vars, 1, d_model))
        self.position_embedding = PositionalEmbedding(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):                                       # x: [B, n_vars, T]
        n_vars = x.shape[1]
        glb = self.glb_token.repeat((x.shape[0], 1, 1, 1))
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.patch_len)   # non-overlapping patches
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        x = self.value_embedding(x) + self.position_embedding(x)
        x = torch.reshape(x, (-1, n_vars, x.shape[-2], x.shape[-1]))
        x = torch.cat([x, glb], dim=2)                                         # append global token
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        return self.dropout(x), n_vars


class Encoder(nn.Module):
    def __init__(self, layers, norm_layer=None, projection=None):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask, tau=tau, delta=delta)
        if self.norm is not None:
            x = self.norm(x)
        if self.projection is not None:
            x = self.projection(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(d_model, d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(d_ff, d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        B, L, D = cross.shape
        # self-attention within the target's patch+global tokens (intra-target temporal detail)
        x = x + self.dropout(self.self_attention(x, x, x, attn_mask=x_mask, tau=tau, delta=None)[0])
        x = self.norm1(x)
        # only the global token cross-attends to the exogenous variate tokens (narrow gate)
        x_glb_ori = x[:, -1, :].unsqueeze(1)
        x_glb = torch.reshape(x_glb_ori, (B, -1, D))
        x_glb_attn = self.dropout(self.cross_attention(
            x_glb, cross, cross, attn_mask=cross_mask, tau=tau, delta=delta)[0])
        x_glb_attn = torch.reshape(
            x_glb_attn, (x_glb_attn.shape[0] * x_glb_attn.shape[1], x_glb_attn.shape[2])).unsqueeze(1)
        x_glb = x_glb_ori + x_glb_attn
        x_glb = self.norm2(x_glb)
        y = x = torch.cat([x[:, :-1, :], x_glb], dim=1)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm3(x + y)


class Model(nn.Module):
    """TimeXer: endogenous patch self-attention + exogenous cross-attention through a global token."""

    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.features = configs.features
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.use_norm = getattr(configs, "use_norm", 1)
        self.patch_len = configs.patch_len
        self.patch_num = int(configs.seq_len // configs.patch_len)
        self.n_vars = 1 if configs.features == 'MS' else configs.enc_in

        self.en_embedding = EnEmbedding(self.n_vars, configs.d_model, self.patch_len, configs.dropout)
        self.ex_embedding = DataEmbedding_inverted(
            configs.seq_len, configs.d_model, configs.embed, configs.freq, configs.dropout)

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False),
                        configs.d_model, configs.n_heads),
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model, configs.d_ff,
                    dropout=configs.dropout, activation=configs.activation,
                ) for _ in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
        )
        self.head_nf = configs.d_model * (self.patch_num + 1)
        self.head = FlattenHead(configs.enc_in, self.head_nf, configs.pred_len,
                                head_dropout=configs.dropout)

    def forecast_multi(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc = x_enc / stdev

        en_embed, n_vars = self.en_embedding(x_enc.permute(0, 2, 1))    # every channel endogenous
        ex_embed = self.ex_embedding(x_enc, x_mark_enc)                 # full variate set exogenous

        enc_out = self.encoder(en_embed, ex_embed)
        enc_out = torch.reshape(enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        enc_out = enc_out.permute(0, 1, 3, 2)                           # [B, nvars, d_model, patch_num+1]

        dec_out = self.head(enc_out).permute(0, 2, 1)                   # [B, pred_len, nvars]

        if self.use_norm:
            dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
            dec_out = dec_out + means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast_multi(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]
        return None
```
