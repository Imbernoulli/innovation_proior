Let me start from the thing that actually stings, which is that a single linear layer is beating my Transformer. Zeng and colleagues took each series, split it into a smooth trend and a remainder with a moving-average decomposition, ran one linear map per part from the $T$-length lookback straight to the $S$-length horizon, summed them, and that plain dense weighting outperformed Informer, Autoformer, FEDformer — the whole family of temporal-attention forecasters — on the standard long-horizon benchmarks, on accuracy and on cost both. If attention over the time axis were earning its keep, a linear map from past to future has no business winning. So before I build anything I want to understand what my attention is even doing, and whether I have been aiming it at the wrong axis the whole time.

Here is how everyone builds these models, including me. I have a panel $\mathbf{X}\in\mathbb{R}^{T\times N}$: $T$ timesteps, $N$ variates. The reflex from language is "a token per position in the sequence," and the sequence here is time, so I take the slice $\mathbf{X}_{t,:}\in\mathbb{R}^N$ — all $N$ variates at one instant $t$ — and embed it into a $D$-vector. That gives me $T$ tokens, one per timestamp, and I run self-attention over them to capture temporal dependency. It feels natural by analogy. But let me actually look at what is inside one of those tokens, because I think the analogy is lying to me.

A word token in language is a coherent semantic unit; the embedding has something real to represent. What is in $\mathbf{X}_{t,:}$? It is whatever every sensor happened to read at the same wall-clock instant. In a traffic panel those are occupancy readings from detectors scattered across a city; in a weather panel it is temperature next to rainfall next to pressure — different physical quantities, different units, different distributions, jammed into one vector. Worse, they are not even synchronized in the sense that matters: a congestion event hits an upstream detector and only minutes later the downstream one, so the "same timestamp" lumps together the early phase of the event at one sensor and a totally different phase at another. The token is a fruit salad of time-misaligned, incommensurable numbers, and its receptive field is a single instant — one tick of every channel. There is barely any temporal information in it at all; the temporal content lives *across* tokens, and I am asking attention to reconstruct it from a sequence of instantaneous snapshots.

Now I see why attention over these tokens can look meaningless and why the layer norm is actively harmful. Layer norm in this layout normalizes across the feature dimension of a token — which here is the variate mixture at a fixed $t$. So at every timestamp I am centering and scaling temperature against rainfall against pressure together, blending channels that have nothing to do with one another. That is not removing nuisance variation, it is injecting interaction noise between unrelated, possibly lagged processes, and across timesteps it oversmooths. And the attention map over $T$ temporal tokens is telling me which *instants* resemble which other instants in this scrambled feature space — not which *variables* drive which, which is the multivariate structure I actually care about.

There is a cleaner way to see that the temporal axis is the wrong place for attention. Attention is permutation-invariant in its tokens by construction: $\operatorname{softmax}(\mathbf{Q}\mathbf{K}^\top/\sqrt{d_k})\mathbf{V}$ does not know the order its tokens came in, which is exactly why language models bolt on a positional encoding. But time *has* order — shuffling the timesteps destroys the series. So putting a permutation-invariant operator on the temporal axis is a structural mismatch; I am fighting the operator's own symmetry and patching it with positional encodings just to undo the damage. Zeng's people pointed at exactly this. Meanwhile the cost is $O(T^2)$ in the number of temporal tokens, so when I lengthen the lookback to feed the model more history — which classical statistics says should *help* — I both blow up compute and dilute the attention over a longer scrambled sequence, and the accuracy flatlines or drops. Every symptom traces back to the same root: I tokenized along time and mixed variates inside each token.

So let me ask the inverse question. The thing with order that I should not permute is the time axis. The thing without an inherent order, where permutation invariance is *correct*, is the set of variates — channel 5 and channel 12 have no canonical ordering. What if I make the token the whole series of one variate, and let attention run across the variates? Take the column $\mathbf{X}_{:,n}\in\mathbb{R}^T$ — the entire lookback of channel $n$ — and embed *that* into a $D$-vector $\mathbf{h}_n$. Now I have $N$ tokens, one per variate, each one a description of a single physically coherent series over its whole history. The receptive field of a token is no longer one instant; it is the entire lookback window. This is the extreme of patching: PatchTST enlarged the receptive field by grouping a handful of consecutive timesteps into a token instead of one; push that all the way and the "patch" is the complete series. Each token finally has something real and homogeneous to represent.

Let me check that every part lands in the right place under this inversion. Self-attention now runs over the $N$ variate tokens, so the score map is $N\times N$ — it relates variable to variable. That is precisely the multivariate correlation structure I was missing, and now it is what attention literally computes. Permutation invariance over variates is fine, because variates have no order, so I do not even need a positional encoding on this axis — which is a relief, because a positional encoding over the variate index would have been meaningless anyway. Where did the temporal modeling go? It went into the per-token map. Each token $\mathbf{h}_n$ is an encoding of a single series, and the feed-forward network, applied identically to each token, transforms that series representation. The FFN is a nonlinear map on the temporal content of one channel — which is exactly the regime where the linear and MLP forecasters were winning. So the inversion gives me a division of labor that I could not get before: attention handles cross-variate correlation, the FFN handles the temporal/series representation per channel. The linear model wasn't beating me because attention is useless; it was beating me because I had attention doing the temporal job, which it is bad at, and nothing doing the cross-variate job. Flip the axes and each operator does the thing it is good at.

I want to nail the FFN-as-forecaster point because it is the crux of why this can beat DLinear rather than merely match it. The FFN is shared across all variate tokens — the same weights process every channel's series representation. That is channel independence in the temporal pathway, the PatchTST idea, but now combined with explicit cross-variate attention, which PatchTST threw away. A shared map over single-series representations is in spirit a stack of nonlinear linear-forecasters: by the universal approximation property of MLPs it can represent amplitude, periodicity, even frequency content of a generic series, and because it is shared it learns these as transferable properties of "any series" rather than memorizing one channel. So the temporal job is done by the descendant of the method that was beating me, and the cross-variate job — which that method lacked entirely — is added back via attention on top. And because the encoder is just "encode the lookback representation, then map to the future," the generation is one-shot: a single linear projection from the $D$-dim token to the $S$-dim horizon, no autoregressive decoder. The recent linear and dense-encoder forecasters already showed a direct linear projection to the whole horizon is competent, so I can drop the heavy encoder–decoder generative machinery entirely and keep an encoder-only stack.

Now the layer norm, which was poison in the old layout. Apply it per variate token: normalize each $\mathbf{h}_n$ across its own feature dimension,
$$\operatorname{LayerNorm}(\mathbf{H}) = \left\{\frac{\mathbf{h}_n - \operatorname{Mean}(\mathbf{h}_n)}{\sqrt{\operatorname{Var}(\mathbf{h}_n)}} \ \middle|\ n=1,\dots,N\right\}.$$
This is normalizing *within a single series*, not across the variate mixture — so it never blends temperature with rainfall. Each series token is brought to a common Gaussian-ish scale, which directly attacks the inter-variate measurement-discrepancy problem (channels at wildly different magnitudes are put on equal footing) and is the same medicine that the non-stationarity work prescribed for distribution shift. The very operation that was fusing unrelated channels in the temporal layout becomes, in the inverted layout, the thing that removes cross-variate scale discrepancy. Same module, opposite effect, because the axis it normalizes over flipped from "variates at one time" to "one variate over its representation."

Let me look harder at the attention scores under this inversion, because there is a reason they become interpretable and it is worth being precise about. With variate tokens $\mathbf{H}=\{\mathbf{h}_1,\dots,\mathbf{h}_N\}\in\mathbb{R}^{N\times D}$, linear projections give queries, keys, values $\mathbf{Q},\mathbf{K},\mathbf{V}\in\mathbb{R}^{N\times d_k}$, and a pre-softmax entry is
$$\mathbf{A}_{i,j} = \left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}}\right)_{i,j} \propto \mathbf{q}_i^\top \mathbf{k}_j.$$
Each token has already been normalized on its feature dimension by the layer norm, so $\mathbf{q}_i$ and $\mathbf{k}_j$ are projections of unit-ish-scale vectors, and the inner product $\mathbf{q}_i^\top\mathbf{k}_j$ behaves like a learned correlation between variate $i$ and variate $j$. The full map $\mathbf{A}\in\mathbb{R}^{N\times N}$ is then a learned multivariate correlation matrix, and the softmax-weighted aggregation pulls each variate's next representation toward the variates it correlates with, via $\mathbf{V}$. That is a genuinely interpretable object — I can read off which channels inform which — in a way the temporal-token map never was. The $1/\sqrt{d_k}$ is the usual scaling: $\mathbf{q}_i^\top\mathbf{k}_j$ is a sum of $d_k$ products, so its variance grows with $d_k$; dividing by $\sqrt{d_k}$ keeps the logits at a scale where the softmax is not saturated and gradients survive. I keep it exactly as is — the whole bet here is that I do not need to redesign any component, only to point them at the right axes.

Let me write the forward path per variate and make sure the shapes are coherent.
$$\mathbf{h}^0_n = \operatorname{Embedding}(\mathbf{X}_{:,n}),\quad \mathbf{H}^{l+1}=\operatorname{TrmBlock}(\mathbf{H}^l),\ l=0,\dots,L-1,\quad \hat{\mathbf{Y}}_{:,n}=\operatorname{Projection}(\mathbf{h}^L_n),$$
where $\operatorname{Embedding}:\mathbb{R}^T\mapsto\mathbb{R}^D$ and $\operatorname{Projection}:\mathbb{R}^D\mapsto\mathbb{R}^S$. The embedding takes a length-$T$ series to a $D$-token; I'll implement it as a linear map (an MLP is the general statement, a single `Linear(T, D)` is the realization). $L$ blocks of the encoder, each block
$$\mathbf{H} = \operatorname{LayerNorm}\big(\mathbf{H} + \operatorname{SelfAttn}(\mathbf{H})\big),\qquad \mathbf{H} = \operatorname{LayerNorm}\big(\mathbf{H} + \operatorname{FFN}(\mathbf{H})\big),$$
residual then norm in each sublayer, attention over the $N$ tokens and FFN per token. Then the projection maps each $D$-token to its $S$-length forecast. One subtlety on where the order of time lives: I claimed I do not need a positional encoding, but the timesteps obviously have order — where is it stored? It is in the embedding and the FFN weights. The embedding `Linear(T, D)` assigns a distinct weight column to each input timestep, so position-1 and position-2 of the series are handled by different parameters; the order is baked into the parameterization, fixed by the neuron-to-timestep assignment, not something attention has to recover. The FFN over the series representation likewise has a fixed neuron permutation. So order is implicit in the weights along the temporal axis, and the only axis attention sees — the variate axis — correctly has no positional encoding.

Before I commit, I should wrap the whole thing in the right normalization, because these series are non-stationary and the instance-normalization fixes were a clear win. The drift in mean and scale over the lookback window is nuisance I do not want the network spending capacity on. So per series, over the lookback, I subtract the mean and divide by the standard deviation, run the model on the standardized panel, and restore the statistics on the output. Concretely, with $\mathbf{X}\in\mathbb{R}^{B\times T\times N}$, take the mean over the time axis $\mathtt{means}=\operatorname{mean}_t(\mathbf{X})$ and detach it — I want it treated as a constant shift, not something gradients flow through, since it is a data statistic and not a parameter — subtract it, compute $\mathtt{stdev}=\sqrt{\operatorname{Var}_t(\mathbf{X}-\mathtt{means})+10^{-5}}$ with the biased variance (divide by $T$, not $T-1$, because this is a normalization statistic over the window, not an inferential estimate), and divide. The $10^{-5}$ guards a flat (constant) channel from a divide-by-zero. After the model produces the horizon, multiply back by $\mathtt{stdev}$ and add $\mathtt{means}$, broadcasting the per-series statistics across the $S$ forecast steps. This is reversible instance normalization, and it slots in as an outer wrapper around the inverted encoder.

One more practical thing: the harness hands me time-feature marks (calendar features) alongside the values, and in the multivariate/exogenous setting all channels come in together. Under the variate-token view these are not awkward — a time-feature mark is just another series over the lookback, so I can embed it as an *extra* token by concatenating it onto the variate axis before the embedding linear. The attention then lets the real variates attend to these covariate tokens too. After the projection I just keep the first $N$ output channels (the real variates) and drop the covariate tokens. So the embedding takes the transposed input, optionally concatenates the transposed marks along the token dimension, and applies the linear map; the projection's output is sliced back to $N$.

Now let me convince myself this actually answers the three things that were broken. Longer lookback: in the inverted model the lookback length $T$ is the *input feature dimension* of the embedding linear, not the number of tokens. The number of tokens is $N$, fixed by the panel. So lengthening the history does not multiply the token count or the attention cost — it just widens the embedding's input, and the FFN/linear pathway is exactly the kind of operator that classical statistics says should benefit from more history. So the model can use a longer lookback for more precise predictions instead of choking on it. Cross-variate structure: attention is now literally an $N\times N$ correlation map, so it is used, not discarded. Cost as $N$ grows: attention is $O(N^2)$ in variates now, but the components are untouched, so any efficient-attention drop-in — sparse, linear, hardware-accelerated — plugs straight in to bring that down when $N$ is large; nothing about the inversion forbids it. And a bonus falls out: because the token count $N$ is just the number of input series and the FFN is shared and applied per token, the model is not tied to a fixed $N$ — it can be trained on one set of variates and run on another, since the parameters live in the shared per-token maps and the projection, none of which bake in a particular variate count. The variate tokens are interchangeable as far as the weights are concerned.

So the resolution is not a new attention mechanism, not a new normalization, not a decomposition — it is the same Transformer block with its operators pointed at the inverted axes: tokenize each variate's whole series, attend across variates for correlation, feed-forward per token for the temporal representation, layer-norm per series for scale, project linearly to the horizon, all wrapped in reversible instance normalization, encoder-only, no positional encoding. Let me write it in the form the harness expects.

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
            # covariate marks become additional variate tokens
            x = self.value_embedding(torch.cat([x, x_mark.permute(0, 2, 1)], 1))
        return self.dropout(x)                                # [B, N(+marks), D]


class FullAttention(nn.Module):
    # Standard scaled dot-product attention; 1/sqrt(d_k) keeps logits unsaturated.
    def __init__(self, attention_dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values):
        B, L, H, E = queries.shape
        scale = 1. / sqrt(E)
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)   # [B, H, N, N] variate-variate
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
        new_x = self.attention(x, x, x)                      # cross-variate correlation
        x = x + self.dropout(new_x)
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
        self.projection = nn.Linear(configs.d_model, configs.pred_len, bias=True)  # R^D -> R^S

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # reversible instance normalization (per series, over the lookback)
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc / stdev

        _, _, N = x_enc.shape                                 # number of real variates

        enc_out = self.enc_embedding(x_enc, x_mark_enc)       # [B, N(+marks), D]
        enc_out = self.encoder(enc_out)                       # [B, N(+marks), D]

        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :N]   # [B, S, N], drop mark tokens

        # de-normalize: restore per-series mean and scale, broadcast over the horizon
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :]
```

The causal chain, start to finish: a single linear forecaster was beating temporal-attention Transformers, which told me attention was on the wrong axis — the temporal token mixed time-misaligned, incommensurable variates into one instant, so attention learned meaningless instant-to-instant maps and layer norm fused unrelated channels, while a permutation-invariant operator sat on the ordered time axis where it does not belong, and lengthening the history only diluted attention and exploded cost. Inverting the tokenization — one token per variate's whole series — puts attention on the unordered variate axis where its symmetry is correct and its $N\times N$ map is exactly the multivariate correlation I needed, moves the temporal modeling into the per-token FFN where the winning linear/MLP forecasters live, turns layer norm into per-series scale normalization that removes measurement discrepancy, makes the lookback a feature width rather than a token count so more history helps instead of hurting, and needs no positional encoding because order is held in the embedding and FFN weights — all of it the same untouched Transformer components, only re-pointed, wrapped in reversible instance normalization and read out by one linear projection to the horizon.
