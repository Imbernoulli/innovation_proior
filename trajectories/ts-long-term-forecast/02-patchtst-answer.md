**Problem (from step 1).** The linear control is competitive on the temporal task (ETTh1 0.3962), so
the missing ingredient is either richer temporal structure or cross-channel structure. Settle the
temporal half first: can attention over time, done right, beat a linear temporal map? The heavy
Transformers tokenized **per step** — a permutation-invariant operator over meaningless single scalars
— which is why a linear map that reads whole-window shapes beat them.

**Key idea.** Change what a token is. Cut each channel's length-L series into contiguous length-P
patches with stride S, embed each patch as a token, and run a stock self-attention encoder over the N
patches — channel-independent, the same shared backbone applied to every channel as a batch. A patch
carries local temporal shape (the unit attention can actually compare), the token count drops from L to
≈L/S (an S² cut in attention cost), and the channel treatment is held at the linear model's setting so
this is a clean temporal-only test.

**Why it works.** Single steps are meaningless and the time axis is redundant; grouping a local stretch
into one token gives attention order-bearing semantic units, so the positional encoding does honest work
instead of papering over a permutation mismatch. Reversible per-window instance normalization
(subtract/divide by the look-back mean/std, restore after the head) handles level drift the loader does
not absorb. Direct multi-step head — flatten the N patch features and map to the whole horizon — keeps
the generation step that already worked for the linear model.

**Hyperparameters / edit-surface notes.** `patch_len=16` from `configs`, `stride=8` hardcoded (the loop
passes no stride); padding = stride so the final value is never dropped, giving N = floor((L−P)/S)+2 = 12
patches at L=96. `PatchEmbedding` and `Encoder`/`EncoderLayer`/`FullAttention`/`AttentionLayer` reused
from the scaffold layers; encoder norm is BatchNorm (steadier than LayerNorm at small N). Fixed scaffold
config `e_layers=2`, `n_heads=8`, `d_model=512`, `d_ff=512`, `dropout=0.1` — note the method's own
script tunes `e_layers`/`n_heads`/`batch_size` per dataset and uses a long look-back, neither available
here, which blunts patching's signature long-history advantage.

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

    def forward(self, x):                                       # x: [B, nvars, d_model, patch_num]
        return self.dropout(self.linear(self.flatten(x)))


class Model(nn.Module):
    """PatchTST: channel-independent self-attention over subseries patches, direct multi-step."""

    def __init__(self, configs, patch_len=16, stride=8):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        padding = stride                                        # pad end so the final value is kept

        self.patch_embedding = PatchEmbedding(
            configs.d_model, patch_len, stride, padding, configs.dropout)

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model, configs.d_ff,
                    dropout=configs.dropout, activation=configs.activation,
                ) for _ in range(configs.e_layers)
            ],
            norm_layer=nn.Sequential(
                Transpose(1, 2), nn.BatchNorm1d(configs.d_model), Transpose(1, 2)),
        )

        self.head_nf = configs.d_model * int((configs.seq_len - patch_len) / stride + 2)
        self.head = FlattenHead(configs.enc_in, self.head_nf, configs.pred_len,
                                head_dropout=configs.dropout)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # reversible per-window instance normalization
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc / stdev

        x_enc = x_enc.permute(0, 2, 1)                          # [B, nvars, L]; channels -> batch
        enc_out, n_vars = self.patch_embedding(x_enc)          # [B*nvars, patch_num, d_model]
        enc_out, attns = self.encoder(enc_out)
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        enc_out = enc_out.permute(0, 1, 3, 2)                  # [B, nvars, d_model, patch_num]

        dec_out = self.head(enc_out)                           # [B, nvars, pred_len]
        dec_out = dec_out.permute(0, 2, 1)                     # [B, pred_len, nvars]

        # de-normalize
        dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        dec_out = dec_out + means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]
        return None
```
