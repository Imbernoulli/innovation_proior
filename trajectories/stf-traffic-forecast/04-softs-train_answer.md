The spectral-graph rung told me two things at once. StemGNN's learned-Laplacian coupling delivered the largest single jump on the ladder exactly where I expected it — PEMS04 flow MAE fell to $24.2040$, the dataset where cross-node propagation dominates — confirming that channel coupling is real signal worth keeping. But on the speed grids the same machinery was fragile: METR-LA MAE actually *regressed* to $4.0045$ (worse than the channel-independent TimeMixer's $3.9497$, with MAPE worsening too), and PEMS-BAY barely moved to $2.0334$. The lesson is sharp: keep cross-node modeling, but the *way* StemGNN does it — learn a graph, do eigendecomposition-free polynomial filtering, fuse two spectral domains — is expensive, numerically touchy, and on smooth twelve-step speed series it pays for the apparatus without recovering enough signal to come out ahead. I want cross-node information back, but through a mechanism that does not re-import the fragility and that survives $N$ in the hundreds.

I propose SOFTS — a series-core fusion model built on the STAR (STar Aggregate-Redistribute) module. Two design moves carry it, and they compound. First, invert the representation. A per-timestamp token (one instant's slice across all nodes) bakes channel entanglement into the token itself — it jams together sensors sitting at different phases of a propagating congestion wave — so any cross-node operation happens by accident, structurally, before the model decides anything. I flip the unit: a token is one node's whole twelve-step lookback, mapped by a single $\text{Linear}(\text{input\_len} \to D)$ into one $D$-vector standing for that node. Now there are $N$ tokens, one robust self-contained description per series, and every cross-node operation becomes an explicit, separable step over the *set* of node tokens. RevIN brackets the whole thing — subtract each node's mean over the lookback, divide by its std, restore both on the prediction — so the tokens carry shape, not the drifting per-node level that causes most of the train/test mismatch.

The second and load-bearing move is *how* the tokens talk. The obvious choice is self-attention across the $N$ tokens, but that re-imports both of StemGNN's problems in new clothing: it is $O(N^2)$ in time and memory (the binding constraint as $N$ grows), and worse, its entire content is the *per-pair* correlation weights — precisely the quantity that overfits under non-stationarity, because fitting cross-channel structure means fitting the joint distribution of the channels, and that joint distribution is what shifts between train and test. If a sensor goes anomalous, every node that attends to it ingests its garbage at full, now-meaningless weight. So I ask what "interaction" must actually mean: each node should benefit from the others, but it does not need to know *which specific* other node helped. That admits a different topology entirely — route everything through a hub instead of a mesh. Every node contributes to one shared summary, and every node reads back from that one summary. Star-shaped, not all-pairs: $N$ aggregate-in plus $N$ redistribute-out, linear cost. And the hub is not merely a cost trick — a single summary aggregated over *all* nodes is a *statistic*, and a statistic over hundreds of channels is far more stable than any one channel or any one pair. An anomalous sensor is one contributor out of hundreds, drowned out, rather than a correlation partner consumed whole. The hub fixes the $O(N^2)$ cost and the fragile pairwise dependence in one move.

This is the STAR layer. Aggregate the $N$ node tokens into one *core* with a shared MLP, $\text{combined} = \text{FFN}_1(x)$ mapping each token to a core dimension; pool the $N$ rows into one; repeat the core back to all $N$ nodes; concatenate it onto each token and fuse with $\text{FFN}_2$ plus a residual so the layer *refines* each node with global context rather than overwriting it. The aggregation must be permutation-invariant over nodes — relabeling sensor 5 as sensor 200 cannot change the network's summary — so it takes the set-function form $\rho(\sum_x \phi(x))$ with a *shared* $\phi$ summed at equal weight, not a permutation-variant per-coordinate weighting. This is not only symmetry: a per-channel weight spends capacity on the channel *index*, and the index is meaningless under distribution shift, so forcing the summary to depend on what each node *contains* rather than which slot it sits in is the robust choice. The outer $\rho$ is absorbed into the fusion MLP, so I drop it explicitly.

The pool itself deserves care, because the core is one shared object every node reads from — if it overfits, everyone overfits together. Mean pooling is maximally stable but washes a strong informative node into the crowd; max pooling preserves the salient signal but bets a core coordinate on the one node that might be the anomalous one. I want something between, that also regularizes, and stochastic pooling does exactly this. For each core feature, softmax the node activations *across the channel axis* into a distribution $r_{n} = \text{softmax}_n(\text{combined})$; in training, *sample* one node per feature from it,

$$\text{core}[f] = \text{combined}[\,n^\ast,\,f\,], \qquad n^\ast \sim r_{\cdot,f},$$

so high-activation nodes are picked often (the max tendency) but low ones occasionally too (the mean tendency), injecting noise into the shared core every step. At test time I take the expectation under the same distribution, a deterministic magnitude-weighted average $\text{core}[f] = \sum_n r_{n,f}\,\text{combined}[n,f]$. Train and test are consistent — the test value is the mean of the training sampler — and the result is a soft aggregate over all nodes with weight concentrated on the salient ones but spread so no single dead sensor can hijack the core. Softmax over the channel axis is the right axis: I normalize across the very things I am pooling.

One layer is then: $x \leftarrow \text{LayerNorm}(x + \text{STAR}(x))$, followed by an ordinary per-node feed-forward sublayer $x \leftarrow \text{LayerNorm}(x + \text{FFN}(x))$ — standard transformer-block hygiene, but with STAR replacing attention. Stack two. Every step is linear in $N$: $\text{FFN}_1$ is $O(N d^2)$, the pool $O(N d)$, $\text{FFN}_2$ is $O(N d^2)$, the head $O(N d H)$ — no $N \times N$ matrix is ever formed, the scaling StemGNN's Laplacian could not offer. Wrapping it: RevIN-normalize, embed each node's lookback to a token, run two STAR blocks, a final LayerNorm, a linear head $D \to 12$ per node, transpose, denormalize. The wide $\text{hidden\_size}=512$ inverted stack is unstable at the harness default learning rate, so I set $\text{lr}=5\times10^{-4}$ through `CONFIG_OVERRIDES` and leave weight decay at default. The model uses no timestamps. I expect this to reclaim both speed datasets that StemGNN's spectral machinery gave up; the open risk is PEMS04, where a hub that summarizes all nodes into one global core may *under*-model the directed upstream→downstream propagation that a real graph captures.

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
    core_size: int = field(default=128)
    num_layers: int = field(default=2)
    dropout: float = field(default=0.05)


class RevIN(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x, mode):
        if mode == "norm":
            self.mean = x.mean(dim=1, keepdim=True).detach()
            self.stdev = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + self.eps).detach()
            return (x - self.mean) / self.stdev
        else:
            return x * self.stdev + self.mean


class MLP(nn.Module):
    def __init__(self, in_dim, mid_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, mid_dim)
        self.fc2 = nn.Linear(mid_dim, out_dim)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class STAR(nn.Module):
    """STar Aggregate-Redistribute module.

    Aggregates cross-variate info into a core representation via
    stochastic pooling (training) or weighted mean (inference),
    then redistributes back to each variate.
    """
    def __init__(self, hidden_size, core_size):
        super().__init__()
        self.ffn1 = MLP(hidden_size, hidden_size, core_size)
        self.ffn2 = MLP(hidden_size + core_size, hidden_size, hidden_size)

    def forward(self, x):
        B, N, D = x.shape
        combined = self.ffn1(x)  # [B, N, core_size]

        if self.training:
            # Stochastic pooling
            ratio = F.softmax(combined, dim=1)  # [B, N, core_size]
            ratio = ratio.transpose(1, 2).reshape(-1, N)
            indices = torch.multinomial(ratio, 1)
            indices = indices.view(B, -1, 1).transpose(1, 2)  # [B, 1, core_size]
            core = torch.gather(combined, 1, indices)  # [B, 1, core_size]
            core = core.repeat(1, N, 1)
        else:
            # Weighted mean
            weight = F.softmax(combined, dim=1)
            core = (combined * weight).sum(dim=1, keepdim=True).repeat(1, N, 1)

        return self.ffn2(torch.cat([x, core], dim=-1))


class SOFTSBlock(nn.Module):
    def __init__(self, hidden_size, core_size, dropout):
        super().__init__()
        self.star = STAR(hidden_size, core_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)

    def forward(self, x):
        x = self.norm1(x + self.star(x))
        x = self.norm2(x + self.ffn(x))
        return x


class Custom(nn.Module):
    """SOFTS: Series-Core Fusion baseline.

    Inverted architecture (nodes as tokens), using STAR modules
    instead of self-attention for O(N) cross-variate communication.
    """

    def __init__(self, config: CustomConfig):
        super().__init__()
        self.revin = RevIN()

        # Sequence embedding: [B, T, N] -> transpose -> [B, N, T] -> [B, N, D]
        self.embed = nn.Linear(config.input_len, config.hidden_size)
        self.embed_drop = nn.Dropout(config.dropout)

        self.layers = nn.ModuleList([
            SOFTSBlock(config.hidden_size, config.core_size, config.dropout)
            for _ in range(config.num_layers)
        ])
        self.norm = nn.LayerNorm(config.hidden_size)

        # Output: [B, N, D] -> [B, N, T'] -> [B, T', N]
        self.head = nn.Linear(config.hidden_size, config.output_len)

    def forward(self, inputs, inputs_timestamps):
        x = self.revin(inputs, "norm")
        N = x.size(-1)

        h = self.embed_drop(self.embed(x.transpose(1, 2)))
        for layer in self.layers:
            h = layer(h)
        h = self.norm(h)

        pred = self.head(h).transpose(1, 2)[:, :, :N]
        pred = self.revin(pred, "denorm")
        return pred


# CONFIG_OVERRIDES: override training hyperparameters for your method.
# Allowed keys: lr, weight_decay.
CONFIG_OVERRIDES = {'lr': 0.0005}
```
