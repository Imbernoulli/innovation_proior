## Research question

A decoder-only Transformer generates one token at a time, and at each step every layer's
attention must look back at all preceding tokens. To avoid recomputing the per-token keys and
values from scratch on every step, implementations store them in a *KV cache* and reload it
each step. For standard multi-head attention the cache holds
`2 * n_h * d_h * l` scalars per token across `l` layers (keys and values, every head), and it grows
linearly with sequence length and batch size. Past a certain context length the cache, not the
arithmetic, sets the ceiling on the maximum batch size and sequence length a given amount of
accelerator memory can serve, and each generation step is bottlenecked by the *memory bandwidth*
needed to stream the whole K and V tensors back in.

The question is: how can a causal self-attention structure reduce the number of bytes cached per
token while remaining compatible with rotary position encoding, which modern decoders rely on?

## Background

By this time the decoder-only Transformer is the standard generative architecture. Each block
is an attention module plus a feed-forward network, with rotary position embeddings carrying
position. The attention module is where the inference cost concentrates.

The memory-bandwidth analysis of incremental decoding (Shazeer 2019) is the load-bearing fact.
For batched multi-head attention during *training*, the ratio of memory accessed to arithmetic
performed is roughly `O(1/d_h + 1/(b·n))` — comfortably compute-bound. But during *incremental*
generation, where queries from different positions cannot be processed in parallel, that ratio
becomes roughly `Θ(n/d + 1/b)`: the `n/d` term comes from reloading, at every one of the `n`
steps, the K and V tensors that encode the whole history. As context length `n` approaches the
model dimension `d`, or batch size `b` is small, this ratio approaches 1 and memory bandwidth
dominates. The conclusion is concrete: to make generation fast you must shrink the K and V tensors
that get reloaded each step — equivalently, shrink the KV cache.

The second concept the structure must respect is rotary position embedding (RoPE, Su et al.
2021). RoPE injects position multiplicatively: it rotates each query and key by an angle
proportional to its absolute position,
`q_m -> R^d_{Θ,m} W_q x_m`, `k_n -> R^d_{Θ,n} W_k x_n`,
with a block-diagonal rotation matrix `R^d_{Θ,m}` whose 2×2 blocks rotate by `m·θ_i`,
`θ_i = 10000^{-2(i-1)/d}`. The crucial algebraic property is that the attention score then
depends only on the *relative* position through the product of rotations,
`q_m^T k_n = x_m^T W_q^T R^d_{Θ,n-m} W_k x_n`, because `R^{dT}_{Θ,m} R^d_{Θ,n} = R^d_{Θ,n-m}`
and the rotation is orthogonal. This relative-position-by-rotation behavior is exactly why RoPE
is used; it is also a *position-dependent matrix that sits in the middle of the query-key
product*, which any cache-restructuring will have to contend with.

## Baselines

These are the prior attention structures a new design would be measured against and react to.

**Multi-Head Attention (MHA), Vaswani et al. 2017.** Project the input `h_t ∈ R^d` to queries,
keys, values with `W^Q, W^K, W^V ∈ R^{n_h d_h × d}`, slice into `n_h` heads, and for each head
`o_{t,i} = sum_{j≤t} softmax_j(q_{t,i}^T k_{j,i} / sqrt(d_h)) v_{j,i}`, then `u_t = W^O[o_{t,1};
...;o_{t,n_h}]`. Every query head has its own independent key and value head, caching
`2 n_h d_h l` elements per token.

**Multi-Query Attention (MQA), Shazeer 2019.** Keep `n_h` separate query heads but share a
*single* key head and a *single* value head across all of them: in einsum terms, drop the head
index from `K` and `V`. The cache falls to `2 d_h l` per token — a factor `n_h` smaller — and
this is precisely the `n/d -> n/(d·h)` reduction the bandwidth analysis calls for.

**Grouped-Query Attention (GQA), Ainslie et al. 2023.** Interpolate between the two: partition
the `n_h` query heads into `n_g` groups, each group sharing one key head and one value head.
`n_g = 1` recovers MQA; `n_g = n_h` recovers MHA. The cache is `2 n_g d_h l`, tunable by `n_g`.
The group heads are obtained by mean-pooling the original heads when converting a trained MHA
checkpoint.

## Evaluation settings

The natural yardstick is a fixed small-scale GPT-style pretraining loop with downstream
language-model evaluation — the same protocol other attention-architecture studies use.

- **Pretraining substrate:** a nanoGPT-style decoder, here at the 345M scale
  (24 layers, 16 heads, hidden 1024), trained on a tokenized web-text mixture
  (ClimbMix training split, ~58GB) for a Chinchilla-optimal token budget (~7.1B tokens, 13535
  steps), 2-GPU data-parallel, learning rate 3e-4.
- **Primary metric:** validation cross-entropy loss at 345M (lower is better).
- **Efficiency metric:** KV bytes cached per token, derived from the realized attention
  structure — the memory axis the whole design targets (lower is better).
- **Held-out language-model loss:** average cross-entropy on WikiText-2, WikiText-103, and
  LAMBADA at the final checkpoint.
- **Downstream zero-shot accuracy** via the standard lm-eval harness: ARC-Easy, HellaSwag,
  PIQA, Winogrande.
- **Protocol:** identical training recipe across attention structures; the only thing that
  varies between runs is the attention block, so any difference in loss or cached bytes is
  attributable to the structure.

## Code framework

The attention structure plugs into a fixed nanoGPT pretraining loop. Everything outside the
attention block is settled and shared across all structures: the token/position embeddings, the
`LayerNorm`, the residual `Block` wiring, the MLP, the `GPTConfig`, the training loop, the loss.
Only the internals of the attention block are open. That slot receives a tensor `x` of shape
`[batch, time, channels]` and must return a tensor of the same shape; no assumption is made yet
about how it constructs attention state, handles position, or accounts for cached bytes.

```python
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass


class LayerNorm(nn.Module):
    """LayerNorm with optional bias (fixed; shared by every block)."""
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    """Causal self-attention block. The body is the open architecture slot."""

    def __init__(self, config, layer_idx=0):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.layer_idx = layer_idx
        # TODO: the attention architecture.

    def forward(self, x):
        bsz, seq_len, _ = x.size()
        # TODO: compute the causal attention output.
        raise NotImplementedError


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config, layer_idx=layer_idx)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 24
    n_head: int = 16
    n_embd: int = 1024
    dropout: float = 0.0
    bias: bool = False
```

The KV-bytes-per-token metric is read off whatever the attention block decides to keep during
generation; the loss comes from the shared training loop. The whole design lives inside
`CausalSelfAttention` and any helper functions it later needs.
