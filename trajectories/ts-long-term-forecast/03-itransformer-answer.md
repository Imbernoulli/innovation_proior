**Problem (from step 2).** PatchTST fixed the temporal lever (ETTh1 0.3794, below the linear control),
but ECL stayed the relative laggard (0.1819, worst MAE) because it — like the linear model — is
channel-independent and never lets one channel inform another. ECL's 321 coupled channels need
cross-variate structure. The standard temporal-token layout cannot supply it cleanly: a per-timestep
token is a fruit salad of incommensurable, time-misaligned channel values, LayerNorm blends unrelated
channels, and attention runs (wrongly) on the ordered time axis.

**Key idea.** Invert the token. Make each token the *whole look-back of one variate* (linear map
R^T→R^D), giving N variate tokens, and run self-attention **across variates** — h_i·h_j is a clean
cross-variate correlation. The position-wise FFN then acts within each variate token as the per-series
temporal feature extractor. Direct multi-step head: one linear d_model→pred_len per variate token.

**Why it works.** Attention now runs over the unordered variate set, so permutation invariance is
correct (no temporal positional encoding needed, the structural mismatch is gone); LayerNorm normalizes
per-series instead of blending channels; cost is O(N²) in variates, independent of T, so longer history
is cheap. Cross-variate correlation is exactly the lever PatchTST and the linear model could not touch —
the ECL lever.

**Hyperparameters / edit-surface notes.** `DataEmbedding_inverted` (R^T→R^D, marks appended as extra
tokens) and stock `Encoder`/`EncoderLayer`/`FullAttention`/`AttentionLayer` reused from the scaffold
layers, `norm_layer=LayerNorm(d_model)`. Reversible per-window instance norm wraps the forecast. Fixed
scaffold config `d_model=512`, `e_layers=2`, `n_heads=8`, `d_ff=512` — note the method's own script
tunes `d_model=128` on ETTh1 (7 channels), so the fixed 512 here is over-parameterized on the small
dataset and the inversion's payoff is expected on ECL, not ETTh1.

```python
import torch
import torch.nn as nn
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted


class Model(nn.Module):
    """iTransformer: attention across variate tokens; FFN as the per-series temporal extractor."""

    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        self.enc_embedding = DataEmbedding_inverted(
            configs.seq_len, configs.d_model, configs.embed, configs.freq, configs.dropout)

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
            norm_layer=torch.nn.LayerNorm(configs.d_model),
        )

        self.projection = nn.Linear(configs.d_model, configs.pred_len, bias=True)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # reversible per-window instance normalization
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc / stdev

        _, _, N = x_enc.shape
        enc_out = self.enc_embedding(x_enc, x_mark_enc)        # [B, N(+marks), d_model]
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :N]   # [B, pred_len, N]
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
