Let me start from the thing that actually stings: a single dense map over time is beating my Transformer. Zeng and colleagues take each channel's lookback, optionally split it into trend and seasonal parts, and map the length-$T$ history straight to the length-$S$ future with linear layers. That should be an embarrassingly weak baseline if temporal self-attention is doing the essential work. Instead it is competitive or better on the long-horizon benchmarks. So before I change another attention kernel, I need to ask whether I am pointing the existing Transformer components at the wrong objects.

The default reflex comes from language: a sequence has positions, so make one token per position. For a multivariate panel $\mathbf{X}\in\mathbb{R}^{T\times N}$, that means token $t$ is $\mathbf{X}_{t,:}$, the vector of all $N$ variates at one timestamp. But what does that vector contain? It is not a word. It is temperature next to rainfall next to pressure, or road sensors from different places in a traffic network, or electricity clients with different scales. The components can have different physical units, different distributions, and different delays from the same underlying event. A traffic jam can hit one detector before another, so the same wall-clock timestamp can mean different phases of the same event across channels. I am packing those simultaneous but not necessarily comparable measurements into one token and asking attention to compare such tokens across time.

That makes several symptoms line up. First, the token's receptive field is one instant, so the representation has very little of the local or global shape of a time series inside it. The temporal information lives across tokens, and self-attention has to reconstruct it from a long list of instantaneous mixtures. Second, LayerNorm in that layout normalizes the feature vector of a timestamp token, so it centers and scales a mixed bundle of unrelated variates. If the variates are delayed or measured in incompatible units, this can inject interaction noise rather than remove nuisance scale. Third, the attention map is a time-time map computed from those mixed timestamp vectors. It tells me which instants look similar in the scrambled feature space, not directly which variables are related.

There is an even cleaner structural objection. Self-attention is permutation-equivariant over its token axis before positional information is added. That is why the original Transformer needs positional encodings for language order. Time is exactly the axis where order matters most; shuffling timesteps destroys the series. So I am putting a permutation-friendly operation on an order-sensitive axis, then adding position information to repair the mismatch. Meanwhile the cost grows like $O(T^2)$ as I enlarge the lookback window, and those larger windows often do not improve the temporal-token Transformer. More history should help a forecaster, so if more history mostly dilutes attention, the tokenization is suspect.

What axis has no natural order? The variate axis. Sensor 12 and sensor 37 do not have a canonical sequence position in the same way timestep 12 and timestep 37 do. That suggests the inversion: make one token out of the whole history of one variate. For channel $n$, take $\mathbf{X}_{:,n}\in\mathbb{R}^T$ and embed that length-$T$ series into a $D$-vector $\mathbf{h}_n$. Now the tokens are $\mathbf{H}=\{\mathbf{h}_1,\dots,\mathbf{h}_N\}\in\mathbb{R}^{N\times D}$. Each token is physically coherent: it is one variable over the whole lookback. The old "patching" idea enlarged a token from one point to a short segment; this pushes the patch to the whole series.

Once I do that, the native Transformer modules suddenly have sensible jobs. Self-attention runs over the $N$ variate tokens, so its score matrix is $N\times N$ and compares one variate's whole-series representation with another's. That is the multivariate structure the linear channel-independent models lack. The token axis is unordered, so attention's symmetry is now a feature rather than a problem, and I do not need positional encodings over variate indices. The temporal order has not disappeared; it is in the ordered coordinates of the length-$T$ vector that feeds the embedding. A `Linear(T, D)` has a distinct weight for each input position, so timestep 1 and timestep 2 are not interchangeable. The same is true downstream for the per-token feed-forward maps over the representation coordinates.

The FFN also lands in the right place. A Transformer FFN is applied identically to each token. With timestamp tokens, that means it processes mixed instantaneous snapshots. With variate tokens, it processes one series representation at a time. That is very close to the successful linear/MLP forecaster idea: learn reusable temporal features that transfer across channels. The same FFN weights process every variate token, so the temporal pathway is channel-independent in the useful PatchTST sense, but cross-variate information is still available through attention. This resolves the split that was bothering me: temporal modeling should be handled by dense per-series maps, and multivariate correlation should be handled by attention across variates.

The readout can be just as direct. If the final representation for variate $n$ is $\mathbf{h}^L_n$, a projection $\mathbb{R}^D\to\mathbb{R}^S$ can produce the whole horizon $\hat{\mathbf{Y}}_{:,n}$. I do not need an autoregressive decoder whose tokens are future timestamps. Recent dense forecasters already show that a direct projection to the horizon is a strong forecasting primitive, and here the projection follows a representation that has had a chance to interact with other variates.

So the shape is:
$$
\mathbf{h}^0_n=\operatorname{Embedding}(\mathbf{X}_{:,n}),\qquad
\mathbf{H}^{l+1}=\operatorname{TrmBlock}(\mathbf{H}^l),\ l=0,\dots,L-1,\qquad
\hat{\mathbf{Y}}_{:,n}=\operatorname{Projection}(\mathbf{h}^L_n).
$$
The general statement allows MLPs for the embedding and projection, but I will realize both as single linear layers, `nn.Linear(seq_len, d_model)` and `nn.Linear(d_model, pred_len)` — the minimal faithful version, and enough since the FFN inside the block already supplies the nonlinearity.

I need to be precise about LayerNorm, because it is easy to overclaim. In the inverted layout, LayerNorm acts on each variate token's $D$ features, not across a timestamp's mixed variates. Schematically, for token $\mathbf{h}_n$ it is
$$
\operatorname{LN}(\mathbf{h}_n)=\gamma\frac{\mathbf{h}_n-\mu_n}{\sqrt{\sigma_n^2+\epsilon}}+\beta,\qquad
\mu_n=\operatorname{mean}_d(\mathbf{h}_{n,d}),\quad \sigma_n^2=\operatorname{var}_d(\mathbf{h}_{n,d}).
$$
In PyTorch the default `LayerNorm(d_model)` uses $\epsilon=10^{-5}$ and learnable affine parameters. The useful point is not that this becomes a literal Gaussianizer or a complete distribution-shift solution. The useful point is that it normalizes one variate representation at a time, so it reduces representation-scale discrepancy without blending unrelated channels inside a timestamp token.

The actual non-stationary lookback normalization is a separate outer step. If I enable it, I subtract each series' lookback mean and divide by its lookback standard deviation before the inverted encoder, then restore those statistics after projection. For $\mathbf{X}\in\mathbb{R}^{B\times T\times N}$:
$$
\mu=\operatorname{mean}_t(\mathbf{X}),\qquad
\sigma=\sqrt{\operatorname{var}_t(\mathbf{X}-\mu;\ \text{unbiased}=False)+10^{-5}}.
$$
The mean is detached because it is a data statistic, not a learned parameter. The variance is the biased window variance (divide by $T$, not $T-1$, since this is a normalization statistic over the window), and the $10^{-5}$ guard prevents division by zero on constant channels. After projection, I multiply by $\sigma$ and add $\mu$ back, broadcasting across the $S$ predicted steps. This is RevIN-like in spirit, but I keep it as the plain Non-stationary-Transformer-style mean/std normalization, not a full RevIN module with extra learned affine parameters.

Now the attention math. For one head, queries, keys, and values have shape $\mathbb{R}^{N\times d_k}$. The raw score is exactly
$$
s_{ij}=\frac{\mathbf{q}_i^\top\mathbf{k}_j}{\sqrt{d_k}},
$$
then the weights are $\alpha_{ij}=\operatorname{softmax}_j(s_{ij})$, and the output is $\mathbf{o}_i=\sum_j\alpha_{ij}\mathbf{v}_j$. The $1/\sqrt{d_k}$ factor is still the Vaswani scaling: without it, the dot-product variance grows with the head dimension and can saturate the softmax. Because the tokens are whole-series summaries and are normalized inside the blocks, $s_{ij}$ can behave like a learned dependence score between variates. But I should not call it a Pearson correlation coefficient. It is a query-key compatibility score whose learned maps can reveal multivariate correlation structure.

The block ordering should stay vanilla post-norm, because the point is not to invent another block. The encoder layer computes attention with `x` as query, key, and value; adds dropout and a residual; applies `LayerNorm`; passes the result through a position-wise FFN implemented as two `Conv1d(kernel_size=1)` layers with activation and dropout; adds another residual; and applies a second `LayerNorm`. A final encoder norm is applied after the stack. This is the ordinary Transformer encoder machinery, just run on variate tokens.

Calendar marks and other time features are not a separate decoder problem. In the embedding layer, `x_enc [B,T,N]` is permuted to `[B,N,T]`; if `x_mark_enc [B,T,F]` exists, it is also permuted to `[B,F,T]` and concatenated along the token axis. The same `Linear(T,D)` embeds real variates and mark-series as tokens. After projection, I slice back to the first `N` tokens so the covariate tokens can inform the real variates but do not become output channels.

Let me check the longer-lookback claim carefully. A larger lookback $T$ does not increase the number of attention tokens; attention still sees $N$ variate tokens, so the quadratic attention part is independent of $T$. The embedding input width does change from `Linear(T,D)` to a different `Linear(T',D)`, so one trained weight matrix is not automatically length-polymorphic in the time dimension. The claim I can safely make is narrower and correct: for a model configured with a larger lookback, the added history widens the per-series embedding rather than making attention attend over more time tokens. That is exactly why longer history can help without the temporal-attention dilution problem.

As $N$ grows, the attention cost is now $O(N^2)$ over variates. That is not free; on Traffic-like data with hundreds of sensors, it can be heavier than attending over a short lookback. But the architecture leaves the attention module unchanged, so sparse, linear, or hardware-accelerated attention variants can be swapped in. The variate-token view also means the token count can differ between training and inference, because the same embedding, FFN, attention projections, and readout are shared across tokens. The model is tied to the lookback length through the embedding weights, but it is not tied in the same way to a fixed number of variates.

So the resolution is not a new attention formula. It is an axis assignment: whole-series variate tokens, attention across variates, FFN per variate token, per-token LayerNorm, direct projection to the horizon, optional per-instance normalization around the forecast path, no positional encoding over variate indices. I can write the forecast module in code.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt


class DataEmbedding_inverted(nn.Module):
    # Embed each variate's whole lookback series (length seq_len) into a D-dim variate token.
    # Time-feature marks are treated as extra series -> concatenated as extra tokens.
    def __init__(self, seq_len, d_model, dropout=0.1):
        super().__init__()
        self.value_embedding = nn.Linear(seq_len, d_model)   # R^T -> R^D, per variate
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, x_mark):
        x = x.permute(0, 2, 1)                                # [B, T, N] -> [B, N, T]
        if x_mark is None:
            x = self.value_embedding(x)                       # [B, N, D]
        else:
            x = self.value_embedding(torch.cat([x, x_mark.permute(0, 2, 1)], 1))
        return self.dropout(x)                                # [B, N(+marks), D]


class FullAttention(nn.Module):
    # Standard scaled dot-product attention; 1/sqrt(d_k) keeps logits unsaturated.
    def __init__(self, attention_dropout=0.1, output_attention=False):
        super().__init__()
        self.dropout = nn.Dropout(attention_dropout)
        self.output_attention = output_attention

    def forward(self, queries, keys, values):
        B, L, H, E = queries.shape
        scale = 1. / sqrt(E)
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)   # [B, H, N, N] variate-variate
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
        out = out.view(B, L, -1)
        return self.out_projection(out), attn


class EncoderLayer(nn.Module):
    # One inverted Transformer block: attention over variate tokens, then FFN per token,
    # each as residual + post-LayerNorm. FFN is two 1x1 convs == position-wise MLP D->d_ff->D.
    def __init__(self, attention, d_model, d_ff, dropout=0.1, activation="relu"):
        super().__init__()
        self.attention = attention
        self.conv1 = nn.Conv1d(d_model, d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(d_ff, d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)                   # per variate token, over D
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
        self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)  # R^D -> R^S

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.use_norm:                                    # per-series lookback normalization
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc = x_enc / stdev

        _, _, N = x_enc.shape                                 # number of real variates

        enc_out = self.enc_embedding(x_enc, x_mark_enc)       # [B, N(+marks), D]
        enc_out, attns = self.encoder(enc_out)               # [B, N(+marks), D]
        dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :N]   # [B, S, N], drop mark tokens

        if self.use_norm:                                    # restore mean/scale over the horizon
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out, attns

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out, attns = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        return dec_out[:, -self.pred_len:, :]
```
