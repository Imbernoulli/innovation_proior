# iTransformer, distilled

## Problem

Multivariate time series forecasting: from a lookback $\mathbf{X}\in\mathbb{R}^{T\times N}$ ($T$
steps, $N$ variates) predict the next $S$ steps $\mathbf{Y}\in\mathbb{R}^{S\times N}$. The trouble
with the standard recipe — embed each timestamp's $N$ variates into a "temporal token" and run
attention over the $T$ tokens — is that the token fuses physically heterogeneous, often
time-misaligned channels at one instant, so attention learns uninformative instant-to-instant maps,
layer norm blends unrelated channels, a permutation-invariant operator sits on the ordered time axis,
cost is $O(T^2)$ in lookback, and accuracy does not improve with longer history. Plain linear models
beat these Transformers.

## Key idea

Invert the tokenization axis, leaving every Transformer component unchanged. Make each **variate** a
token: embed the whole lookback series of one channel into a $D$-vector. Then:
- **Self-attention runs across the $N$ variate tokens** → the score map is $N\times N$, a learned
  multivariate-correlation matrix; permutation invariance is now correct (variates are unordered).
- **The FFN runs per variate token** over its series representation → the temporal/nonlinear modeling,
  the job where linear/MLP forecasters excel; shared across variates (channel-independent temporal
  pathway).
- **Layer norm normalizes each variate token** across its features → puts series on a common scale,
  removing cross-variate measurement discrepancy and non-stationarity, instead of fusing channels.
- **No positional encoding** — temporal order is held in the embedding's per-timestep weights and the
  FFN; the only axis attention sees (variates) has no order.
- **Encoder-only**, with a single linear **projection $\mathbb{R}^D\!\to\!\mathbb{R}^S$** for one-shot
  multi-step output. Wrapped in reversible instance normalization (subtract/divide by per-series
  lookback mean/std, restore on output).

A consequence: lookback $T$ is the embedding's input width, not the token count ($N$), so longer
history helps and does not raise attention cost; efficient-attention variants drop in unchanged for
large $N$; the model is not tied to a fixed number of variates.

## Architecture

Per variate $n$:
$$\mathbf{h}^0_n=\operatorname{Embedding}(\mathbf{X}_{:,n}),\quad
\mathbf{H}^{l+1}=\operatorname{TrmBlock}(\mathbf{H}^l),\ l=0,\dots,L-1,\quad
\hat{\mathbf{Y}}_{:,n}=\operatorname{Projection}(\mathbf{h}^L_n),$$
with $\operatorname{Embedding}:\mathbb{R}^T\!\to\!\mathbb{R}^D$, $\operatorname{Projection}:\mathbb{R}^D\!\to\!\mathbb{R}^S$.
Each block (residual + post-LayerNorm per sublayer):
$$\mathbf{H}=\operatorname{LayerNorm}(\mathbf{H}+\operatorname{SelfAttn}(\mathbf{H})),\qquad
\mathbf{H}=\operatorname{LayerNorm}(\mathbf{H}+\operatorname{FFN}(\mathbf{H})).$$
Per-token layer norm: $\operatorname{LayerNorm}(\mathbf{H})=\{(\mathbf{h}_n-\operatorname{Mean}(\mathbf{h}_n))/\sqrt{\operatorname{Var}(\mathbf{h}_n)}\}$.
Attention scores: $\mathbf{A}_{i,j}=(\mathbf{Q}\mathbf{K}^\top/\sqrt{d_k})_{i,j}\propto\mathbf{q}_i^\top\mathbf{k}_j$, $\mathbf{A}\in\mathbb{R}^{N\times N}$.

## Code

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt


class DataEmbedding_inverted(nn.Module):
    """Embed each variate's whole series into a token; marks become extra tokens."""
    def __init__(self, seq_len, d_model, dropout=0.1):
        super().__init__()
        self.value_embedding = nn.Linear(seq_len, d_model)   # R^T -> R^D
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, x_mark):
        x = x.permute(0, 2, 1)                                # [B, T, N] -> [B, N, T]
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            x = self.value_embedding(torch.cat([x, x_mark.permute(0, 2, 1)], 1))
        return self.dropout(x)                                # [B, N(+marks), D]


class FullAttention(nn.Module):
    def __init__(self, attention_dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values):
        B, L, H, E = queries.shape
        scale = 1. / sqrt(E)
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)
        return V.contiguous()


class AttentionLayer(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        d_keys = d_model // n_heads
        self.inner_attention = FullAttention()
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_keys * n_heads)
        self.out_projection = nn.Linear(d_keys * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values):
        B, L, _ = queries.shape
        S = keys.shape[1]
        H = self.n_heads
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)
        out = self.inner_attention(queries, keys, values).view(B, L, -1)
        return self.out_projection(out)


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff, dropout=0.1, activation="relu"):
        super().__init__()
        self.attention = attention
        self.conv1 = nn.Conv1d(d_model, d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(d_ff, d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x):
        x = x + self.dropout(self.attention(x, x, x))        # cross-variate correlation
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))     # series representation, per token
        return self.norm2(x + y)


class Encoder(nn.Module):
    def __init__(self, layers, norm_layer=None):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        if self.norm is not None:
            x = self.norm(x)
        return x


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.dropout)
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(configs.d_model, configs.n_heads),
                    configs.d_model, configs.d_ff,
                    dropout=configs.dropout, activation=configs.activation,
                ) for _ in range(configs.e_layers)
            ],
            norm_layer=nn.LayerNorm(configs.d_model),
        )
        self.projection = nn.Linear(configs.d_model, configs.pred_len, bias=True)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc / stdev

        _, _, N = x_enc.shape
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out = self.encoder(enc_out)
        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :N]   # [B, S, N]

        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :]
```

Typical configuration on the standard benchmarks: `d_model=512`, `d_ff=512`, `e_layers=2`,
`n_heads=8`, `dropout=0.1`; Adam, MSE loss, batch size 32, learning rate $10^{-4}$, a fixed small
number of epochs with early stopping; lookback `seq_len=96`, horizon `pred_len=96`. For exogenous /
`features=MS` evaluation the model outputs all $N$ channels and the harness scores only the target
channel; covariate time-feature marks enter as additional tokens and are dropped by the `[:, :, :N]`
slice.
