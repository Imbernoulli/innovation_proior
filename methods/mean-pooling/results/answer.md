# Mean Pooling

Mean pooling collapses a variable-size, unordered set of D-dimensional member vectors
(transformer token outputs, or per-variable patch embeddings) into a single D-dimensional
vector per item by averaging the members. It is permutation-invariant, accepts any set size,
keeps the raw output from being dominated by set size, and adds no learnable parameters — the
transparent default that more elaborate aggregators (attention/cross-attention pooling, a
learned weighted sum) must beat.

## Problem it solves

An encoder emits a set `{h_1, ..., h_n}`, `h_i ∈ R^D`, with n varying per item, but the
downstream consumer (cosine similarity, a classifier, the next backbone block) needs one
fixed-size vector per item. The reduction `g({h_1,...,h_n}) -> R^D` must be (i) invariant to
member order, (ii) defined for any n, (iii) not dominated by n, and — for a clean baseline —
(iv) parameter-free.

## Key idea

Order-invariance forces a symmetric reduction; a parameter-free, size-stable one is the
uniform mean.

- **Why a symmetric reduction.** Concatenation is order- and size-dependent; a recurrent
  reader imposes a spurious order. Only symmetric combiners (sum, max, mean, weighted average)
  respect the set's lack of order.
- **Why mean, not sum.** For roughly-independent members with per-coordinate mean μ and
  variance σ², the sum has expectation nμ and standard deviation √n·σ, so length leaks into
  the raw representation. Dividing by n gives the mean, whose expectation is μ independent of
  n and whose variance is σ²/n.
- **Why mean, not a single token.** A designated summary token is a bottleneck and is trained
  for the pretraining objective, so off-the-shelf it is a poor representation; the general
  set-pooling case (e.g. per-variable tokens) has no such token at all. Averaging uses every
  member.
- **Why uniform weights.** Among all weighted averages `Σ w_i h_i`, `Σ w_i = 1`, the uniform
  weighting `w_i = 1/n` is the only permutation-symmetric choice and the maximum-entropy
  weighting when no member is privileged. It is also the zero-query / uniform-softmax limit of
  attention pooling — so the mean is exactly the value a learned attentive aggregator collapses
  to with no learned signal, i.e. the floor it must clear.
- **Mean vs max.** Max keeps the per-dimension extreme member (good when an outlier carries the
  signal; non-smooth, discards the rest). Mean uses every member equally and is smooth — the
  safer default when the set's information is spread across comparably-informative members.

## Final form

For a set of member vectors `h_1, ..., h_n` (real members only):

```
g({h_i}) = (1/n) Σ_{i=1}^{n} h_i
```

Masked form (ragged sets padded to length N, mask m_i ∈ {0,1}, 1 = real member):

```
g = ( Σ_i m_i · h_i ) / clamp( Σ_i m_i , min=1e-9 )
```

- The numerator zeroes pad slots; the denominator is the true count n = Σ_i m_i, not the
  padded N. The clamp at 1e-9 only fires on an empty set (0/0 → 0), invisible otherwise.
- Reduces to the plain mean when every slot is real (m all ones).

Variance-stabilizing sibling (mean-sqrt-len): divide by `√(Σ_i m_i)` instead of `Σ_i m_i`.
The √n normalizer keeps the aggregate standard deviation about σ across set sizes. Plain ÷n
keeps the expectation fixed at μ and shrinks variance like σ²/n (standard deviation σ/√n).
Plain mean (÷n) is the default.

## Working code

Masked mean over a set, faithful to the canonical pooling implementation
(`sum(token · mask) / clamp(sum(mask), min=1e-9)`):

```python
import torch
import torch.nn as nn


def mean_pool(token_embeddings, attention_mask, sqrt_len=False):
    """Mean pooling of a set of member vectors with a real/pad mask.
      token_embeddings: [B, N, D]   attention_mask: [B, N]  (1 real, 0 pad)
    Returns [B, D]."""
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
    sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
    if sqrt_len:
        return sum_embeddings / torch.sqrt(sum_mask)
    return sum_embeddings / sum_mask


class Pooling(nn.Module):
    """Parameter-free mean pooling layer (mean / mean-sqrt-len)."""

    def __init__(self, embed_dim, sqrt_len=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.sqrt_len = sqrt_len

    def forward(self, token_embeddings, attention_mask):
        return mean_pool(token_embeddings, attention_mask, sqrt_len=self.sqrt_len)
```

Fixed-size-set special case (variable aggregation: every set is full, no padding, so the
masked mean is just the average over the variable axis):

```python
import torch.nn as nn


class VariableAggregator(nn.Module):
    """Mean pooling over the V variable tokens at each spatial location.
    No additional learnable parameters."""

    def __init__(self, embed_dim, num_heads, num_vars):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads      # unused
        self.num_vars = num_vars        # unused

    def forward(self, x):
        # x: [B, V, L, D]  ->  uniform average over the variable axis V
        return x.mean(dim=1)            # [B, L, D]
```

## Relation to alternatives

- **Sum pooling** = mean without the ÷n; raw output magnitude scales with set size.
- **Attention / cross-attention pooling, learned weighted sum** = `Σ_i α_i h_i` with learned,
  content-dependent weights; more expressive, but adds parameters. Mean is their uniform /
  zero-query special case and the parameter-free lower bound.
- **Max pooling** = per-dimension maximum; keeps the extreme member, discards the rest,
  non-smooth. Better when the signal is concentrated in an outlier; mean is better when it is
  spread across the members.
- **Single-token (CLS) reduction** = use one designated vector; a bottleneck trained for the
  pretraining objective, and unavailable for sets without such a token.
