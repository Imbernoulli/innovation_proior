# Context

## Research question

A decoder-only language model around the 7B scale can be attractive only if it is also cheap to serve. Autoregressive serving spends cost in two coupled places: every new token streams cached keys and values from prior tokens, and attention over a long history grows with sequence length. The research question is how to keep the quality benefits of a modern Transformer decoder while making decode-time memory traffic, cache size, and long-prompt attention cost small enough for high-throughput deployment.

The design has to preserve useful long-range dependence. A naive efficiency change that simply discards old context may make inference cheaper but fails the language-modeling problem. The interesting target is a compact decoder that keeps a route for old information to affect later predictions while bounding the most expensive inference state.

## Background

**Decoder Transformer substrate.** A modern decoder stack uses causal self-attention, residual connections, pre-normalization, a position-wise feed-forward network, and learned token embeddings. Contemporary 7B-scale decoders commonly use RMSNorm, rotary position embeddings on queries and keys, and a gated feed-forward such as SwiGLU.

**Attention math.** For a head dimension $d$, causal attention at position $i$ forms scores $q_i k_j^\top / \sqrt{d}$ for allowed positions $j \le i$, applies a softmax over those scores, and returns the weighted sum of the corresponding values. Full causal attention allows every prior position, so a length-$n$ prefill has quadratic score work and decode step $i$ reads a cache proportional to $i$.

**KV cache bottleneck.** During incremental decoding, the current query is transient, but keys and values from prior tokens persist. Per layer and per sequence, a dense multi-head cache stores one key and one value vector per attention head per cached position. This makes memory traffic scale with both the number of stored key/value heads and the number of cached positions.

**Sparse/local attention lineage.** Prior sparse-attention systems restrict the set of visible keys per query to reduce the full $n \times n$ attention matrix. A fixed local band gives linear work in sequence length for fixed band width, but each layer directly sees only nearby states. Whether depth can compensate for that local restriction is the central architectural question.

**Efficient kernels.** Memory-efficient attention kernels can apply causal and local masks without materializing the full score matrix. Such kernels make a sparse attention pattern an actual implementation strategy rather than just an asymptotic argument.

## Baselines

**Dense causal multi-head attention.** This baseline keeps separate query, key, and value projections for each head and attends to the full causal history. It gives direct access to all previous positions but cache memory and decode bandwidth grow with sequence length and with the number of heads.

**Shared-key/value-head attention.** Existing variants keep multiple query heads while sharing fewer key/value heads. The extreme form shares a single key/value head across query heads; intermediate forms use more than one shared key/value head but fewer than the query-head count. These variants reduce per-position cache width, but by themselves they do not bound cache length.

**Fixed local attention.** A local mask gives each query access only to a recent band of keys. It reduces attention cost for long sequences, but one layer alone no longer has direct access to arbitrary old tokens.

**Prompt prefill with a growing cache.** Prompt tokens are known in advance and can be processed in parallel, unlike later generation. For very long prompts, a dense prefill still forms large attention blocks and leaves a cache whose length grows with the prompt.

## Evaluation settings

The natural quality yardsticks are held-out language modeling and standard open LLM evaluations: commonsense reasoning, world knowledge, reading comprehension, mathematics, code generation, and aggregate multitask benchmarks. The efficiency yardsticks are decode latency, throughput at batch, cache memory per sequence, cache bandwidth per generated token, and prefill cost for long prompts.

The comparison should separate quality from serving cost. A useful method should maintain competitive task performance while reducing the per-token KV cache width, bounding the number of cached positions used by attention, and allowing long prompts to be ingested without forming dense long-context attention.

## Code framework

The available implementation substrate is PyTorch plus a memory-efficient attention primitive that accepts a causal or local attention mask. The scaffold below fixes only the ordinary decoder pieces; the attention rule and cache policy are left open.

```python
import torch
import torch.nn as nn
from dataclasses import dataclass


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    head_dim: int = 128
    hidden_dim: int = 14336
    n_heads: int = 32
    vocab_size: int = 32000
    norm_eps: float = 1e-5


class Attention(nn.Module):
    # TODO: the attention rule and cache policy to design.
    def __init__(self, args):
        super().__init__()

    def forward(self, x, freqs_cis, cache=None, mask=None):
        raise NotImplementedError


class FeedForward(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.w1 = nn.Linear(args.dim, args.hidden_dim, bias=False)
        self.w2 = nn.Linear(args.hidden_dim, args.dim, bias=False)
        self.w3 = nn.Linear(args.dim, args.hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(nn.functional.silu(self.w1(x)) * self.w3(x))


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self._norm(x.float()).type_as(x) * self.weight


class TransformerBlock(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.attention = Attention(args)
        self.feed_forward = FeedForward(args)
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x, freqs_cis, cache=None, mask=None):
        h = x + self.attention(self.attention_norm(x), freqs_cis, cache, mask)
        return h + self.feed_forward(self.ffn_norm(h))
```
