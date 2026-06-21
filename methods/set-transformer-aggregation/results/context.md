# Context: pooling a set of per-variable tokens at each spatial location (weather aggregation)

## Research question

A weather forecasting model has tokenized each meteorological variable independently: at every
spatial patch it produces one `D`-dimensional token per variable, giving a tensor
`x: [B, V, L, D]` (`B` examples, `V` variables, `L` spatial patches, `D` channels). Before a
shared Vision-Transformer backbone runs, these per-variable tokens must be collapsed into a
*single* token per spatial location — an aggregator `[B, V, L, D] → [B, L, D]`, reducing over
the variable axis `V` independently at each location. The reduction must be permutation-invariant
in the variables (no canonical order over temperature, geopotential, humidity, …), must accept
any `V` (read from the input at runtime), and must keep the fused token on the single-token scale
the rest of the pipeline expects. The question is what reduction to use for this slot.

## Background

The aggregation slot has a lineage of set-to-vector reductions:

- **Uniform mean** over the `V` variable tokens. Parameter-free, permutation-invariant,
  scale-preserving (a convex combination, so on the single-token scale), and the maximum-entropy
  default when no variable is privileged.
- **Learned weighted sum** — one learnable scalar per variable, softmax-normalized over the
  variable axis, weighted sum. A convex combination (keeps the scale), strictly more expressive
  than the mean, recovers it at the uniform point.
- **Learnable-query cross-attention** (the ClimaX default; Nguyen et al. 2023). A single
  trainable query cross-attends over the `V` variable tokens at each location:
  `α_v = softmax_v((W^Q q)·(W^K x_v)/√d_k)`, output `W^O · Concat_h(Σ_v α_v W^V x_v)`. The
  weights are *content-dependent* and the values are re-projected (`W^V`); multiple heads keep
  several "which variables matter" patterns distinct.

Three background frames are load-bearing.

First, **attention** (Vaswani et al. 2017): `Attention(Q,K,V) = softmax(QKᵀ/√d_k) V` maps a query
and a set of key-value pairs to a weighted average of the values, the softmax running over the
*set of keys* — so the output is invariant to key order and defined for any number of keys, the
property that makes attention the natural primitive for set-structured input. Multihead attention
runs `h` of these over learned `d_k = D/h` projections and concatenates, so different heads pick
up different relations; `1/√d_k` keeps the logits unit-variance so the softmax does not saturate.

Second, **Deep Sets** (Zaheer et al. 2017): a permutation-invariant function of a set can be
written `ρ(Σ_i φ(x_i))` — encode each element independently with `φ`, sum (a symmetric reduction),
decode with `ρ`. This is universal but encodes each element *in isolation*: the elements never
interact during the encoding. The uniform mean and the learned weighted sum are exactly this
skeleton with `φ` the identity/scalar-weight and the reduction a fixed average.

Third, the **Set Transformer** (Lee, Lee, Kim, Kosiorek, Choi, Teh, "Set Transformer: A Framework
for Attention-based Permutation-Invariant Neural Networks", ICML 2019; arXiv:1810.00825). It makes
the set-to-vector map *attentive* with three blocks, all permutation-respecting (no positional
encoding, no dropout):

- **MAB (Multihead Attention Block)**, a Transformer encoder block without positional encoding:
  for query set `X` and key/value set `Y`,
  `H = LayerNorm(X + Multihead(X, Y, Y; ω))`,  `MAB(X,Y) = LayerNorm(H + rFF(H))`,
  with `rFF` a row-wise feed-forward (applied identically to each element) and `ω = softmax(·/√d)`.
  Equivariant in `X`, invariant in `Y`.
- **SAB (Set Attention Block)** `SAB(X) := MAB(X, X)`: self-attention *within* the set, so the
  elements attend to each other and the block encodes pairwise interactions (stack for higher
  order). Permutation-equivariant.
- **PMA (Pooling by Multihead Attention)** `PMA_k(Z) := MAB(S, rFF(Z))`, where `S ∈ R^{k×d}` are
  `k` learnable *seed* vectors: a learnable, content-dependent attention pooling that returns `k`
  summary vectors.

The overall Set Transformer is `Decoder(Encoder(X))` with `Encoder = SAB∘SAB` (or `ISAB∘ISAB` for
large sets) and `Decoder(Z) = rFF(SAB(PMA_k(Z)))`; for `k=1` the trailing `SAB` is unnecessary
(it only models correlations *among the k outputs*). The paper proves the construction is a
universal approximator of permutation-invariant functions, and its toy max-regression experiment
shows the attentive SAB+PMA matches the max-pooling oracle where mean/sum pooling fail.

## Baselines

These are the reductions measured against in this aggregation slot.

**Uniform mean.** `(1/V) Σ_v x_v`. Parameter-free symmetric reduction; the Deep-Sets skeleton with
identity `φ` and a fixed uniform average.

**Learned weighted sum.** `Σ_v softmax(a)_v · x_v`, one learnable scalar per variable. A global,
fixed, content-independent convex combination.

**Single-query cross-attention (ClimaX default).** One learnable query, multihead cross-attention
over the `V` tokens. Content-dependent weights and re-projected values.

**Deep Sets pooling.** `ρ(Σ_v φ(x_v))` with learned `φ, ρ`. Universal but element-wise encoding —
no interaction among elements during encoding.

## Evaluation settings

- **Data.** ERA5 reanalysis (Hersbach et al. 2020) regridded to 5.625° (a 32×64 grid), the
  WeatherBench resolution; the model is fine-tuned from pretrained ClimaX weights.
- **Targets / protocol.** Map the current atmospheric state to a future one at a given lead;
  evaluate held-out forecasts — 500 hPa geopotential at 3-day lead, 850 hPa temperature at 5-day,
  10 m wind speed at 7-day.
- **Metric — latitude-weighted RMSE (lower is better).** Equal-degree cells cover more area near
  the equator, so each row's squared error is weighted by `L(i) = cos(lat(i)) / ((1/H) Σ_{i'}
  cos(lat(i')))` (weights average to one); the score is the square root of the latitude-weighted
  MSE, averaged over forecasts. The same weighting defines the training loss.
- **Harness.** Each variable/level is standardized to zero mean, unit variance and de-normalized
  before the metric; standard PyTorch / AdamW pipeline with the ClimaX backbone fixed.

## Code framework

A ViT backbone over spatial patch tokens and the rest of the pipeline (data, normalization,
optimizer, the latitude-weighted loss) exist and are fixed. The only empty slot is the module that
turns the *set of per-variable tokens at each spatial location* into *one token per location*.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class VariableAggregator(nn.Module):
    """Reduce a set of per-variable tokens at each spatial location to one token.

    Input:  x : [B, V, L, D]  — V variable tokens at each of L spatial patches, each a D-vector.
    Output:     [B, L, D]     — one token per spatial location for the sequence backbone.

    V is read from the input at runtime and may differ between inputs; the module must accept
    any V. Standard PyTorch primitives (nn.Linear, nn.LayerNorm, nn.MultiheadAttention, F.softmax)
    are available.
    """

    def __init__(self, embed_dim, num_heads, num_vars):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_vars = num_vars
        # TODO: the per-location set-reduction to design.

    def forward(self, x):
        b, v, l, d = x.shape
        # TODO: produce one backbone token per spatial location, returning [B, L, D].
        raise NotImplementedError
```

The encoder supplies one `D`-vector per variable per spatial patch; this module is the
per-location set reduction.
