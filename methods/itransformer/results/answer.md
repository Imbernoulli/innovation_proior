# iTransformer, distilled

## Problem

Multivariate time series forecasting: from a lookback $\mathbf{X}\in\mathbb{R}^{T\times N}$ ($T$
steps, $N$ variates) predict the next $S$ steps $\mathbf{Y}\in\mathbb{R}^{S\times N}$. The standard
Transformer recipe embeds each timestamp's $N$-variate slice as a temporal token and attends over
the $T$ tokens. That makes attention compare instantaneous, physically heterogeneous, often
time-lagged mixtures; it puts a permutation-invariant operation on the ordered time axis; it costs
$O(T^2)$ in lookback length; and longer history often fails to help.

## Key idea

Invert the token axis without inventing new Transformer modules. Make each **variate** a token by
embedding its whole lookback series into a $D$-vector:

- self-attention runs across the $N$ variate tokens, so its score map is a learned proxy for
  cross-variate dependence;
- the FFN runs independently on each variate token, so nonlinear temporal representation learning
  sits in the per-series pathway where linear/MLP forecasters are strong;
- LayerNorm acts on each variate token's feature vector, reducing per-variate representation scale
  discrepancies instead of normalizing a mixed timestamp token;
- no positional encoding is added on the variate axis; temporal order is fixed by the ordered
  coordinates of the `Linear(T, D)` embedding and subsequent per-token maps;
- output is encoder-only: a linear projection $\mathbb{R}^D\to\mathbb{R}^S$ maps each final variate
  token to the whole forecast horizon.

## Architecture

For variate $n$:
$$
\mathbf{h}^0_n=\operatorname{Embedding}(\mathbf{X}_{:,n}),\qquad
\mathbf{H}^{l+1}=\operatorname{TrmBlock}(\mathbf{H}^l),\ l=0,\dots,L-1,\qquad
\hat{\mathbf{Y}}_{:,n}=\operatorname{Projection}(\mathbf{h}^L_n).
$$

Here $\operatorname{Embedding}:\mathbb{R}^T\to\mathbb{R}^D$ and
$\operatorname{Projection}:\mathbb{R}^D\to\mathbb{R}^S$ are MLPs in general; the realization uses a
single linear layer for each. A block is post-norm:
$$
\mathbf{H}=\operatorname{LN}(\mathbf{H}+\operatorname{SelfAttn}(\mathbf{H})),\qquad
\mathbf{H}=\operatorname{LN}(\mathbf{H}+\operatorname{FFN}(\mathbf{H})).
$$

Schematically LayerNorm is
$(\mathbf{h}_n-\operatorname{Mean}(\mathbf{h}_n))/\sqrt{\operatorname{Var}(\mathbf{h}_n)}$ per token;
PyTorch `nn.LayerNorm(d_model)` implements the usual affine form with `eps=1e-5`.

For one attention head,
$$
s_{ij}=\frac{\mathbf{q}_i^\top\mathbf{k}_j}{\sqrt{d_k}},\qquad
\alpha_{ij}=\operatorname{softmax}_j(s_{ij}),\qquad
\mathbf{o}_i=\sum_j\alpha_{ij}\mathbf{v}_j.
$$
The pre-softmax score map can reveal variate-wise correlations because tokens represent normalized
whole-series summaries, but it is not a literal Pearson-correlation matrix.

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
    def __init__(self, attention_dropout=0.1, output_attention=False):
        super().__init__()
        self.dropout = nn.Dropout(attention_dropout)
        self.output_attention = output_attention

    def forward(self, queries, keys, values):
        B, L, H, E = queries.shape
        scale = 1. / sqrt(E)                                 # default 1/sqrt(head_dim)
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)
        return V.contiguous(), (A if self.output_attention else None)


class AttentionLayer(nn.Module):
    def __init__(self, inner_attention, d_model, n_heads):
        super().__init__()
        d_keys = d_model // n_heads
        self.inner_attention = inner_attention
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
        out, attn = self.inner_attention(queries, keys, values)
        return self.out_projection(out.view(B, L, -1)), attn


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
        new_x, attn = self.attention(x, x, x)                # cross-variate correlation
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))     # series representation, per token
        return self.norm2(x + y), attn


class Encoder(nn.Module):
    def __init__(self, layers, norm_layer=None):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer

    def forward(self, x):
        attns = []
        for layer in self.layers:
            x, attn = layer(x)
            attns.append(attn)
        if self.norm is not None:
            x = self.norm(x)
        return x, attns


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.dropout)
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention),
                        configs.d_model, configs.n_heads,
                    ),
                    configs.d_model, configs.d_ff,
                    dropout=configs.dropout, activation=configs.activation,
                ) for _ in range(configs.e_layers)
            ],
            norm_layer=nn.LayerNorm(configs.d_model),
        )
        self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc = x_enc / stdev

        _, _, N = x_enc.shape
        enc_out = self.enc_embedding(x_enc, x_mark_enc)        # [B, T, N] -> [B, N(+F), D]
        enc_out, attns = self.encoder(enc_out)                 # attention over variate tokens
        dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :N]

        if self.use_norm:
            dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
            dec_out = dec_out + means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)

        return dec_out, attns

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out, attns = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        return dec_out[:, -self.pred_len:, :]
```

`DataEmbedding_inverted` permutes `[B,T,N]` to `[B,N,T]`, optionally concatenates
`x_mark_enc.permute(0,2,1)` as extra covariate tokens, applies `nn.Linear(seq_len, d_model)`, and
drops out. `FullAttention` uses no causal mask; its scale is `1/sqrt(head_dim)`, and it returns
attention weights only when `output_attention=True`. The encoder layer is residual attention,
`LayerNorm`, a two-layer `Conv1d(kernel_size=1)` FFN, residual add, and second `LayerNorm`.

The experiment appendix uses Adam, L2/MSE loss, batch size 32, 10 epochs, learning rate in
`{1e-3, 5e-4, 1e-4}`, block count `L in {2,3,4}`, and token dimension `D in {256,512}`; other widths
such as `d_ff`, heads, dropout, and activation come from the configuration.
