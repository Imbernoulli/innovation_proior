The task is multivariate forecasting: from a lookback $\mathbf{X}\in\mathbb{R}^{T\times N}$ of $N$ variates over $T$ steps, predict the next $S$ steps $\mathbf{Y}\in\mathbb{R}^{S\times N}$. The thing that stings is that a single dense map over time — Zeng and colleagues' channel-independent linear forecasters that take each variate's length-$T$ history and project it straight to the length-$S$ horizon — matches or beats heavily engineered Transformers on the long-horizon benchmarks, at a fraction of the cost. That should not happen if temporal self-attention were doing the essential work. Two further observations sharpen the doubt: extending the lookback $T$ does not improve (and often degrades) Transformer accuracy even though more history should help a forecaster, and the cost grows like $O(T^2)$ in $T$. So before tuning another attention kernel, the real question is whether the standard recipe is pointing the existing Transformer components at the wrong objects.

It is. The default reflex comes from language — a sequence has positions, so make one token per position — and for a panel that means token $t$ is $\mathbf{X}_{t,:}$, the simultaneous reading of all $N$ variates at one timestamp. But that vector is not a word: it places temperature next to rainfall next to pressure, channels with different units, different distributions, and different delays from the same underlying event, so the same wall-clock instant can mean different phases of the same disturbance across sensors. Several symptoms then line up. The token's receptive field is a single instant, so the temporal shape of the series lives across tokens and attention must reconstruct it from a long list of instantaneous mixtures; LayerNorm applied to such a token centers and scales a bundle of unrelated variates, injecting interaction noise rather than removing nuisance scale; and the attention map is a time–time map computed from those scrambled vectors, telling us which instants look alike rather than which variables are related. There is also a clean structural objection: self-attention is permutation-equivariant over its token axis before positional information is added — that is why language Transformers need positional encodings — yet time is exactly the axis where order matters most, so the recipe puts a permutation-friendly operator on an order-sensitive axis and then patches the mismatch with positional codes. DLinear and the MLP forecasters expose that temporal dependency is better captured by dense per-series maps, while PatchTST keeps channels independent and so has no mechanism for cross-variate correlation, and Crossformer brings correlation back only by attending over time-segment patches drawn from different (and possibly misaligned) variates with a comparatively heavy redesign. What is missing is a method that uses cross-variate structure, keeps improving with more history, and stays affordable — without abandoning the native Transformer.

I propose iTransformer. The idea is not a new attention formula but an inversion of the token axis: do not invent new modules, just point the ordinary Transformer at the right objects. Which axis has no natural order? The variate axis — sensor 12 and sensor 37 carry no canonical sequence position the way timestep 12 and timestep 37 do. So make each variate a token by embedding its whole lookback series into a $D$-vector, $\mathbf{h}^0_n=\operatorname{Embedding}(\mathbf{X}_{:,n})$ with $\operatorname{Embedding}:\mathbb{R}^T\to\mathbb{R}^D$, producing tokens $\mathbf{H}=\{\mathbf{h}_1,\dots,\mathbf{h}_N\}\in\mathbb{R}^{N\times D}$. The encoder is a stack of ordinary post-norm blocks, $\mathbf{H}^{l+1}=\operatorname{TrmBlock}(\mathbf{H}^l)$ for $l=0,\dots,L-1$, and the readout is a direct projection $\hat{\mathbf{Y}}_{:,n}=\operatorname{Projection}(\mathbf{h}^L_n)$ with $\operatorname{Projection}:\mathbb{R}^D\to\mathbb{R}^S$. Embedding and projection are MLPs in general, but I realize each as a single linear layer — `nn.Linear(seq_len, d_model)` and `nn.Linear(d_model, pred_len)` — the minimal faithful version, and enough because the FFN inside each block already supplies the nonlinearity.

Once the axis is inverted, every native module lands in the right job, and each design choice beats its temporal-token counterpart for a concrete reason. Self-attention now runs over the $N$ variate tokens, so its score matrix is $N\times N$ and compares one variate's whole-series representation with another's — exactly the multivariate structure the channel-independent linear models lack. Because the variate axis is genuinely unordered, attention's permutation symmetry is now a feature rather than a defect, and no positional encoding is needed on the token axis; the temporal order has not been lost, it lives in the ordered coordinates of the length-$T$ vector that feeds the embedding, since `Linear(T, D)` holds a distinct weight per input position and makes timestep 1 and timestep 2 non-interchangeable. The FFN, which a Transformer applies identically to every token, now processes one series representation at a time instead of a mixed instantaneous snapshot, so the temporal pathway is channel-independent in the useful PatchTST sense — the same weights learn reusable temporal features that transfer across variates — while cross-variate information still flows through attention. This resolves the split the analysis exposed: temporal modeling belongs in dense per-series maps, multivariate correlation belongs in attention across variates. LayerNorm follows the same logic: in the inverted layout it normalizes each variate token's $D$ features,
$$
\operatorname{LN}(\mathbf{h}_n)=\gamma\,\frac{\mathbf{h}_n-\mu_n}{\sqrt{\sigma_n^2+\epsilon}}+\beta,\qquad
\mu_n=\operatorname{mean}_d(\mathbf{h}_{n,d}),\quad \sigma_n^2=\operatorname{var}_d(\mathbf{h}_{n,d}),
$$
with PyTorch's default $\epsilon=10^{-5}$ and learnable affine $\gamma,\beta$; the point is not that this Gaussianizes anything but that it normalizes one variate representation at a time and so stops blending unrelated channels inside a timestamp. The blocks stay vanilla post-norm,
$$
\mathbf{H}=\operatorname{LN}(\mathbf{H}+\operatorname{SelfAttn}(\mathbf{H})),\qquad
\mathbf{H}=\operatorname{LN}(\mathbf{H}+\operatorname{FFN}(\mathbf{H})),
$$
deliberately unchanged, because the contribution is the axis assignment, not another block.

The attention itself is the standard Vaswani computation, kept verbatim. For one head with queries, keys, and values in $\mathbb{R}^{N\times d_k}$,
$$
s_{ij}=\frac{\mathbf{q}_i^\top\mathbf{k}_j}{\sqrt{d_k}},\qquad
\alpha_{ij}=\operatorname{softmax}_j(s_{ij}),\qquad
\mathbf{o}_i=\sum_j\alpha_{ij}\mathbf{v}_j.
$$
The $1/\sqrt{d_k}$ factor is the original scaling: without it the dot-product variance grows with the head dimension and saturates the softmax. Because the tokens are normalized whole-series summaries, $s_{ij}$ can behave like a learned dependence score between variates and the pre-softmax map can reveal multivariate correlation structure — but it is a query–key compatibility score, not a literal Pearson-correlation matrix, and I am careful not to overclaim it. Calendar marks are not a separate decoder problem: in the embedding layer `x_enc [B,T,N]` is permuted to `[B,N,T]`, and if `x_mark_enc [B,T,F]` is present it is permuted to `[B,F,T]` and concatenated along the token axis, so the same `Linear(T,D)` embeds real variates and mark-series alike; after projection the output is sliced back to the first $N$ tokens, so covariate tokens inform the real variates through attention without becoming output channels.

Two design properties fall out cleanly and are worth stating precisely. A larger lookback $T$ does not increase the number of attention tokens — attention still sees $N$ variate tokens — so the quadratic part is independent of $T$; the only thing that changes is the input width of `Linear(T,D)`, so for a model configured with a longer lookback the added history widens the per-series embedding rather than diluting attention over more time tokens, which is exactly why longer history can now help. As $N$ grows the attention cost becomes $O(N^2)$ over variates, which is not free on panels with hundreds of sensors, but since the attention module is left untouched, sparse, linear, or hardware-accelerated variants drop straight in. The remaining piece is distribution shift, handled as a separate outer wrapper rather than baked into the blocks: when enabled, I subtract each series' lookback mean and divide by its lookback standard deviation before the encoder and restore them after projection,
$$
\mu=\operatorname{mean}_t(\mathbf{X}),\qquad
\sigma=\sqrt{\operatorname{var}_t(\mathbf{X}-\mu;\ \text{unbiased}=\text{False})+10^{-5}}.
$$
The mean is detached because it is a data statistic, not a learned parameter; the variance is the biased window variance (divide by $T$, since this is a normalization statistic over the window), and the $10^{-5}$ guard prevents division by zero on constant channels; after projection I multiply by $\sigma$ and add $\mu$ back, broadcast across the $S$ predicted steps. This is RevIN-like in spirit but kept as the plain Non-stationary-Transformer mean/std normalization, without RevIN's extra learned affine parameters. The whole resolution, then, is an assignment of axes — whole-series variate tokens, attention across variates, FFN per variate token, per-token LayerNorm, direct projection to the horizon, optional per-instance normalization around the forecast path, and no positional encoding over variate indices.

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
