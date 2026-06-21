TimesNet settled the PEMS04 question and drew the map for the strongest rung. It won PEMS04 decisively — MAE $21.8436$, comfortably under StemGNN's $24.2040$ and the best on the ladder — confirming that the flow residual really was a *temporal* problem, captured once I lifted the series into the 2D period space, with no explicit graph at all. It also took METR-LA to $3.8455$, the best speed MAE so far, and dropped METR-LA RMSE hard. But on PEMS-BAY it came in at $2.0343$ MAE — *worse* than SOFTS's $1.9621$, and losing on RMSE too. That is the gap. PEMS-BAY is the densest grid (325 tightly-correlated sensors), exactly where a cross-node mechanism pays most, and TimesNet has none worth the name — its only spatial touch is the dense per-timestep value-embedding conv, which mixes nodes channel-wise but learns no explicit node-to-node structure. SOFTS, which *does* model cross-node interaction through its core, beats it there. So the two leading rungs each own one half: TimesNet owns the temporal/flow half, SOFTS owns the dense-grid cross-node half, and neither owns both.

The strongest rung must therefore keep an explicit, learned cross-node mechanism — robust like SOFTS's core, but per-pair like StemGNN's graph — without SOFTS's lossy global averaging or StemGNN's spectral fragility. I propose iTransformer, the inverted Transformer: put genuine all-pairs attention back, but on the right axis. SOFTS avoided $O(N^2)$ attention because it leans its whole interaction on fragile per-pair weights, but at these node counts (207–325) the quadratic cost is actually affordable, and the fragility, I will argue, was never attention's fault — it was attention pointed at the wrong axis.

The crux is the axis. The reflex inherited from language is "a token per position in the sequence," and here the sequence is time, so a token becomes the slice of all $N$ nodes at one instant, with attention over the $T$ temporal tokens. Look at what is inside one such token: whatever every sensor read at the same wall-clock instant — and on a propagating-congestion network those readings are not even synchronized in the sense that matters, because a shockwave hits an upstream detector minutes before a downstream one, so "the same timestamp" lumps together different phases of the same event. The token is a time-misaligned node mixture, its receptive field is a single tick, and there is barely any temporal information in it — the temporal content lives *across* tokens. Worse, layer norm in this layout normalizes across the feature dimension of a token — the node mixture at a fixed time — blending nodes at different phases, injecting interaction noise rather than removing nuisance. And the attention map over $T$ temporal tokens tells me which *instants* resemble which, not which *nodes* drive which — the cross-node structure I actually need. There is a cleaner way to see the mismatch: attention is permutation-invariant in its tokens, which is why language bolts on positional encodings; but time *has* order, so a permutation-invariant operator on the temporal axis is a structural mismatch I would have to patch with positional encodings just to undo. Every symptom traces to one root: tokenizing along time and mixing nodes inside each token.

So I invert. The thing with order I must not permute is time; the thing without inherent order, where permutation invariance is *correct*, is the set of nodes — sensor 5 and sensor 200 have no canonical ordering. Make the token the whole twelve-step series of one node, embed *that* into a $D$-vector, and run attention across the $N$ node tokens. Now everything lands in the right place. The attention map is $N \times N$ — it relates node to node, precisely the cross-node correlation TimesNet lacked and PEMS-BAY craves, and now it is what attention literally computes. Permutation invariance over nodes is fine, so I need no positional encoding on this axis. And the temporal modeling moves into the per-token map: each token encodes one node's whole series, and the feed-forward network, shared across tokens, transforms that representation — exactly the regime where the linear/MLP forecasters were already strong. The inversion thus gives a clean division of labor: attention handles cross-node correlation, the FFN handles the per-node temporal representation. The linear floor was not beating attention because attention is useless; it was beating attention pointed at the wrong axis, with nothing doing the cross-node job. Flip the axes and each operator does what it is good at — and unlike SOFTS I keep the *explicit per-pair* correlation the dense grid needs, while unlike StemGNN I get it without any spectral machinery.

Two components flip their effect under the inversion, and that is the whole bet — I redesign nothing, I only point the existing pieces at the right axes. Layer norm, poison in the temporal layout, becomes the right medicine here: applied per node token, it normalizes each node's own representation across its feature dimension — within a single series, never blending nodes — bringing every node to a common scale and directly attacking the inter-node magnitude discrepancy (a flow sensor and a speed sensor, or two speed sensors at different baselines, put on equal footing). Same module, opposite effect, because the axis it normalizes over flipped. And the attention scores become interpretable: with node tokens already normalized, the query–key inner product

$$\text{Attn} = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

behaves like a learned correlation between nodes, the $N \times N$ map a learned multivariate correlation matrix, and the $1/\sqrt{d_k}$ scaling keeps the logits where the softmax is not saturated — kept exactly as is.

In this harness's edit surface the model is the lean encoder-only inverted Transformer. The embedding is a single $\text{Linear}(\text{input\_len}, \text{hidden})$ mapping each node's twelve steps to a token; the order of time is stored in that linear's weight columns (a distinct column per input step), so no positional encoding is needed, and the only axis attention sees — the node axis — correctly has none. The encoder is $\text{num\_layers}$ blocks of $\text{LayerNorm}(x + \text{Attn}(x))$ then $\text{LayerNorm}(x + \text{FFN}(x))$ — attention over the $N$ tokens, FFN per token. A linear head maps each $D$-token to its twelve-step horizon, sliced back to $N$ outputs. RevIN brackets the whole thing for drift, the same fix SOFTS used. The attention is plain `MultiHeadAttention` — full $O(N^2)$, affordable at these $N$ — with no efficient-attention drop-in and no covariate tokens (`inputs_timestamps` is unused). The wide $\text{hidden\_size}=512$ stack wants $\text{lr}=5\times10^{-4}$ through `CONFIG_OVERRIDES`, with weight decay at default. The delta from the two leaders is precise: keep TimesNet's lesson that strong per-node temporal modeling matters (the shared FFN over series tokens), keep SOFTS's lesson that explicit cross-node modeling in the robust inverted view matters (now genuine $N \times N$ attention, not a global core), and drop StemGNN's spectral fragility. I expect this to win the dense PEMS-BAY grid and edge METR-LA; the open question is PEMS04, where a flat per-token FFN with no 2D period structure may under-model flow's sharp periodic shape and trail TimesNet.

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from basicts.configs import BasicTSModelConfig


@dataclass
class CustomConfig(BasicTSModelConfig):
    input_len: int = field(default=12)
    output_len: int = field(default=12)
    num_features: int = field(default=207)
    hidden_size: int = field(default=512)
    n_heads: int = field(default=8)
    num_layers: int = field(default=3)
    dropout: float = field(default=0.1)


class RevIN(nn.Module):
    """Reversible Instance Normalization."""
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x, mode):
        if mode == "norm":
            self.mean = x.mean(dim=1, keepdim=True).detach()
            self.stdev = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + self.eps).detach()
            return (x - self.mean) / self.stdev
        else:  # denorm
            return x * self.stdev + self.mean


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, n_heads, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = hidden_size // n_heads
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

    def forward(self, x):
        B, N, D = x.shape
        q = self.q_proj(x).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) / self.scale
        attn = self.dropout(F.softmax(attn, dim=-1))
        out = (attn @ v).transpose(1, 2).contiguous().view(B, N, D)
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, n_heads, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(hidden_size, n_heads, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)

    def forward(self, x):
        x = self.norm1(x + self.attn(x))
        x = self.norm2(x + self.ffn(x))
        return x


class Custom(nn.Module):
    """iTransformer: Inverted Transformer baseline.

    Treats each node's time series as a token (inverted view).
    Self-attention captures cross-variate dependencies.
    """

    def __init__(self, config: CustomConfig):
        super().__init__()
        self.num_features = config.num_features
        self.revin = RevIN()

        # Embed each node's input_len time series -> hidden_size
        self.embed = nn.Linear(config.input_len, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)

        # Transformer encoder over nodes
        self.layers = nn.ModuleList([
            TransformerBlock(config.hidden_size, config.n_heads, config.dropout)
            for _ in range(config.num_layers)
        ])
        self.norm = nn.LayerNorm(config.hidden_size)

        # Project hidden_size -> output_len per node
        self.head = nn.Linear(config.hidden_size, config.output_len)

    def forward(self, inputs, inputs_timestamps):
        # inputs: [B, T, N]
        x = self.revin(inputs, "norm")

        # Invert: [B, N, T] -> embed -> [B, N, D]
        h = self.dropout(self.embed(x.transpose(1, 2)))

        for layer in self.layers:
            h = layer(h)
        h = self.norm(h)

        # Project: [B, N, D] -> [B, N, T'] -> [B, T', N]
        pred = self.head(h).transpose(1, 2)[:, :, :self.num_features]
        pred = self.revin(pred, "denorm")
        return pred


# CONFIG_OVERRIDES: override training hyperparameters for your method.
# Allowed keys: lr, weight_decay.
CONFIG_OVERRIDES = {'lr': 0.0005}
```
