# Learned weighted sum (softmax-based weighted feature fusion), distilled

Learned weighted sum aggregates `V` aligned feature sources into one by a *learned, normalized weighted
average*: attach one learnable scalar weight per source, push the weights through a softmax over the
source axis so they form a probability distribution (nonnegative, summing to one), and take the
corresponding convex combination of the source maps. This is the softmax-based member of
EfficientDet/BiFPN's weighted feature-fusion family, adapted to the local `VariableAggregator` scaffold.
The final EfficientDet BiFPN uses the ReLU fast-normalized sibling for latency; the softmax form here
matches the `attn`/softmax fusion variant and the task's `learned_weighted_sum` edit.

## Problem it solves

At each spatial location you hold `V` heterogeneous feature maps — different pyramid levels /
resolutions / input variables, already resized to a common shape — and must collapse them into one map.
The default (element-wise sum or mean) forces every source to count equally, but heterogeneous sources
demonstrably contribute unequally. The goal is a fusion rule that (1) lets sources contribute
differently, (2) costs almost nothing in parameters/FLOPs, (3) trains end-to-end by plain backprop with
no projection step, and (4) keeps the fused output on the same scale as its inputs.

## Key idea

Combine by a weighted sum `O = sum_i w_i I_i`, but learn the weights and constrain them to the simplex so
only the *relative* split is free, not the overall scale:

- Per-source **scalar** weights are enough — the quantity being decided ("how much does source `i`
  matter overall") is one number per source, so `V` parameters total, negligible cost. (A per-channel
  weight — a length-`D` vector per source, with the softmax taken over the source axis independently per
  channel — is the more expressive variant when channel-level discrimination is needed.)
- **Normalize onto the simplex** (`w_i >= 0`, `sum_i w_i = 1`). Then `O` is a convex combination of the
  inputs: same scale as a single input, no arbitrary global gain (the failure mode of unbounded
  weights), and it reduces to the plain mean at the uniform point.
- **Softmax** maps unconstrained raw parameters `a in R^V` onto that simplex: `w_i = e^{a_i} / sum_j
  e^{a_j}`. It is smooth and differentiable everywhere, so ordinary SGD/Adam trains the raw `a` with no
  constrained-optimization step; the result reads as a probability/importance distribution over sources.
  Softmax is shift-invariant, so any constant initialization of `a` (zeros or ones) gives the uniform
  `1/V` start — i.e. training begins from the plain-mean baseline and moves off it only as the data
  rewards.

## Final form

Raw per-source weights `a = (a_1, ..., a_V)`; normalized weights and fused output:

```
w_i = exp(a_i) / sum_{j=1}^{V} exp(a_j)        # softmax over the source axis -> simplex
O   = sum_{i=1}^{V} w_i * I_i                   # convex combination of the V source maps
```

with `a` initialized to a constant (e.g. zeros), giving `w_i = 1/V` (plain mean) at init.

**Efficiency sibling — fast normalized fusion.** Replace softmax with ReLU-and-normalize to drop the
exponential and the slower softmax reduction (softmax is measurably slower on GPU). This variant must
use a positive initialization, such as ones; reusing the zero initialization above would produce all-zero
ReLU weights and can leave the fusion dead at initialization:

```
w_i = relu(a_i) / (eps + sum_{j=1}^{V} relu(a_j)),   eps = 1e-4
O   = sum_{i=1}^{V} w_i * I_i
```

The normalized weights still lie in `[0,1]` and (up to `eps`) sum to one; in practice this learns very
nearly the same per-source split as softmax. It trades softmax's strict smoothness (a ReLU can pin a
weight at exactly zero, where it gets no gradient) for speed.

## Working code

Filling the aggregator slot: one learnable length-`V` parameter vector, a softmax over the source axis,
a broadcast multiply, and a sum that collapses the source axis.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class VariableAggregator(nn.Module):
    """Learned, softmax-normalized weighted sum over V source feature maps.

    Input  x: [B, V, L, D]   (V source maps per spatial location)
    Output:   [B, L, D]      (one fused map per location)
    """

    def __init__(self, embed_dim, num_heads, num_vars):
        super().__init__()
        self.embed_dim = embed_dim          # D (unused: weights are per-source scalars)
        self.num_heads = num_heads          # unused: no attention
        self.num_vars = num_vars            # V
        # One raw scalar weight per source; zeros -> uniform 1/V (plain mean) at init.
        self.var_weights = nn.Parameter(torch.zeros(num_vars), requires_grad=True)

    def forward(self, x):
        # x: [B, V, L, D]
        w = F.softmax(self.var_weights, dim=0)      # [V] -> simplex (nonneg, sums to 1)
        w = w.view(1, self.num_vars, 1, 1)          # [1, V, 1, 1] broadcast over B, L, D
        out = (x * w).sum(dim=1)                     # convex combination over sources -> [B, L, D]
        return out
```

Fast-normalized variant (requires changing the parameter initialization to ones):

```python
        # In __init__, use this instead of the zero init above:
        self.var_weights = nn.Parameter(torch.ones(num_vars), requires_grad=True)

        # In forward:
        w = F.relu(self.var_weights)
        w = w / (w.sum(dim=0) + 1e-4)               # ReLU + normalize; no exp, no softmax reduction
        w = w.view(1, self.num_vars, 1, 1)
```

## Why this and not the alternatives

- **vs. uniform mean / element-wise sum:** the mean is the uniform point of this same convex-combination
  family; learned weighted sum keeps the family but lets the network leave the uniform point, so it can
  express the unequal per-source contribution that equal-weight fusion cannot.
- **vs. unbounded learnable weights (`O = sum_i w_i I_i`, `w_i` free):** the map is homogeneous in `w`,
  so the overall scale is an unpinned degree of freedom — it drifts, rescaling the fused output by an
  arbitrary gain that knocks the downstream network off its operating point and can grow without bound
  (instability). Normalizing onto the simplex removes exactly that nuisance scale while keeping all the
  relative-importance expressiveness.
- **vs. cross-attention aggregation:** cross-attention recomputes a source mixing *per location* and
  carries query/key/value projection matrices plus a per-location attention computation. Learned
  weighted sum uses a single global per-source distribution (one `V`-vector, shared across all locations
  and examples) — `V` parameters, no projections, no attention matrix. It captures the global "how much
  does each source matter" signal at a tiny fraction of the cost, at the price of not modeling
  location-dependent mixing.
- **vs. EfficientDet's deployed fast-normalized BiFPN fusion:** the deployed detector avoids softmax
  latency with ReLU-normalized positive weights. The softmax variant is smoother and exactly sums to one,
  while fast-normalized fusion is the implementation choice when GPU latency is the binding constraint.
