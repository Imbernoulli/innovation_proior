# Principal Neighbourhood Aggregation (PNA)

## Problem

A message-passing GNN reduces each node's neighbor multiset to one vector with a *single* aggregator,
chosen by intuition. The expressivity guarantee behind "use sum" (sum is injective on multisets → WL-power)
is proved only for a *countable* feature universe; the vectors an aggregator actually sees in a deep
network are *continuous* (ℝ^d). The question: in the continuous setting, how much can one aggregator
retain, and if one is not enough, how many — and which — are needed, robustly across node degree.

## Key idea

**No single continuous aggregator suffices.** A continuous permutation-invariant map from a size-n
multiset of reals to one vector cannot be injective on all such multisets — a Borsuk-Ulam-style
dimension-counting argument gives a lower bound of *n* aggregators to discriminate size-n multisets over
ℝ. So the right object is *several complementary aggregators*, not the best single one.

**Two orthogonal axes.**

1. *Aggregators with complementary losses:* **mean** (distribution; drops counts), **max** (upper
   support), **min** (lower support), **std** = `sqrt(ReLU(E[x²]−E[x]²)+ε)` (spread; the others are all
   location statistics and miss it). Center + both envelopes + dispersion.
2. *Degree scalers:* sum is degree-coupled and explodes across depth; mean is degree-blind. Make degree a
   learnable knob with a **logarithmic** scaler `S(d, α) = (log(d+1)/δ)^α`, α ∈ {−1, 0, +1}
   (attenuate / identity / amplify), δ = mean of `log(d+1)` over the training set. The bare *linear*
   scaler `S(d)=d` turns a mean into a sum, and a degree-linear injective scaler composed with a
   constructed element map makes the mean injective on bounded countable multisets (this is the
   injective-scaler result I can establish) — but linear scaling is what blows up across depth. The **logarithmic** scaler is the
   bounded-magnitude *replacement*: `log(d+1)≠d`, so it does **not** reproduce a sum, but it reinjects a
   controlled, monotone degree dependence that recovers the count-sensitivity a mean discards without the
   sum's exponential growth.

**Full operator (tensor product).**

  ⊕ = [ I, S(d,+1), S(d,−1) ]ᵀ ⊗ [ μ, σ, max, min ]   — 12 channels,

concatenated with the node's own state and pushed through the update MLP U. Each channel is lossy (it must
be), but the twelve together retain far more of the multiset than any one, and the log scalers make the
operator degree-stable enough to *extrapolate* to larger-degree graphs than seen in training.

## Why it works

The channels separate exactly the cases a single aggregator collapses: shared mean / differing spread →
the σ channels split them; a multiset vs. its k-fold inflation → the amplification channel (degree-scaled
mean) splits them; differing upper envelope → max splits them. The update MLP recombines the surviving
information per task. Gains are largest on structure-heavy graphs (chemistry) and smallest on near-regular
grids (vision) — the expected control.

## Implementation

Canonical PNA aggregation: scatter messages to destination nodes, compute the four aggregators by
scatter-reduce, build per-node log-degree scalers, take the outer product, concatenate (12·F). The std is
clamped non-negative before the root to avoid NaNs at zero spread.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import degree, scatter


class PNAAggregation(nn.Module):
    """4 aggregators {mean, max, min, std} x 3 log-degree scalers {I, +1, -1}."""

    def __init__(self, in_dim, avg_deg_log):
        super().__init__()
        # delta: average of log(d+1) over the training set (fixed normalizer).
        self.register_buffer("delta", torch.tensor(float(avg_deg_log)))
        self.eps = 1e-5
        self.out_dim = in_dim * 12

    def _aggregate(self, messages, index, num_nodes):
        mean = scatter(messages, index, dim=0, dim_size=num_nodes, reduce="mean")
        mx = scatter(messages, index, dim=0, dim_size=num_nodes, reduce="max")
        mn = scatter(messages, index, dim=0, dim_size=num_nodes, reduce="min")
        mean_sq = scatter(messages * messages, index, dim=0,
                          dim_size=num_nodes, reduce="mean")
        std = torch.sqrt(F.relu(mean_sq - mean * mean) + self.eps)
        return torch.cat([mean, mx, mn, std], dim=-1)            # [N, 4F]

    def forward(self, messages, index, node_degree, num_nodes):
        agg = self._aggregate(messages, index, num_nodes)        # [N, 4F]
        log_deg = torch.log(node_degree + 1.0)                   # [N]
        ratio = (log_deg / self.delta.clamp(min=self.eps)).clamp(min=self.eps)
        amp = ratio.unsqueeze(-1)                                # S(d, +1)
        att = ratio.pow(-1.0).unsqueeze(-1)                      # S(d, -1)
        return torch.cat([agg, agg * amp, agg * att], dim=-1)    # [N, 12F]
```

Training details: the 12·F channels are concatenated with the node state into the update MLP, which mixes
them back to the layer width; parameter-matched comparisons (enlarge baselines) isolate the gain from the
operator rather than from extra capacity; the logarithmic scalers are what let the model hold its advantage
when extrapolating to graphs with larger degrees than training.
