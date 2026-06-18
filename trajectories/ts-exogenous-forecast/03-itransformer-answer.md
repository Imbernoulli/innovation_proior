**Problem (from step 2).** PatchTST closed the temporal-model gap but is still channel-independent, so
a residual remains on Weather and ECL — the cost of not reading the other channels. That residual is
the fusion gap, now cleanly isolated. The naive fix (cross-attention on a token-per-timestamp layout)
re-creates the meaningless-token and channel-mixing-through-norm diseases.

**Key idea.** Invert the token: collapse each channel's **whole** look-back into one variate token via
a linear $\mathbb{R}^{T}\to\mathbb{R}^{D}$ map, then run the **stock** self-attention across the $N$
channel tokens. The attention score between two variate tokens is a clean cross-variate correlation —
the side channels can finally write into the target's token. The FFN models temporal representation
within each variate; project each token back $\mathbb{R}^{D}\to\mathbb{R}^{\text{pred\_len}}$.

**Why it works.** Cross-channel mixing happens only in the attention scores, where it is wanted;
LayerNorm now normalizes each *variate* token (per-channel, no smearing); each token has the whole
series in its receptive field. No new kernel — just the right tokenization for cross-variate structure.

**Scaffold edit / hyperparameters.** `DataEmbedding_inverted(seq_len, d_model)` (folds `x_mark_enc` as
extra tokens); stock `Encoder` with `e_layers=2`, `d_model=512`, `d_ff=512`, `n_heads=8`, dropout 0.1,
LayerNorm; `Linear(d_model, pred_len)` head per token; per-instance normalization in/out.

**What to watch.** Should beat PatchTST on **Weather** (covariates drive the target) and on **ECL**
MSE (cross-client structure) — that is the fusion gap closing. On **ETTh1** expect a tie or slight
loss: collapsing the series to one token discards the intra-series detail PatchTST's patches kept, and
ETTh1 barely needs fusion. The $O(N^2)$ fusion is indiscriminate (ECL's 321 channels) — MAE is the
riskier metric.

```python
import torch
import torch.nn as nn
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted


class Model(nn.Module):
    """Cross-variate attention forecaster (iTransformer): one token per channel."""

    def __init__(self, configs):
        super(Model, self).__init__()
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
                                      output_attention=False), configs.d_model, configs.n_heads),
                    configs.d_model, configs.d_ff,
                    dropout=configs.dropout, activation=configs.activation,
                ) for _ in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
        )
        self.projection = nn.Linear(configs.d_model, configs.pred_len, bias=True)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # per-instance normalization
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc / stdev

        _, _, N = x_enc.shape

        # one token per variate (whole series -> D); attention runs across channels
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, _ = self.encoder(enc_out, attn_mask=None)

        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :N]   # [B, pred_len, N]

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
