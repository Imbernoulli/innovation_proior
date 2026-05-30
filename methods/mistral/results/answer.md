# Mistral 7B

## Problem

Make a compact (~7B) decoder language model that is genuinely cheap to *serve*: cut the two costs of autoregressive Transformer inference — the key/value cache that is streamed every decode step (limits throughput/batch) and the attention that is $O(n^2)$ compute and $O(n)$ cache in sequence length $n$ — without losing the ability to use long-range context.

## Key idea

A LLaMA-style decoder (pre-norm RMSNorm blocks, SwiGLU FFN, RoPE) with four inference mechanisms:

- **Grouped-query attention (GQA).** Keep all $h=32$ query heads but use only $n_{kv}=8$ key/value heads, each shared by $h/n_{kv}=4$ query heads. The cache stores 8 heads (4× smaller and 4× less decode bandwidth); the 8 heads are repeated to 32 *after* loading, just before the score matmul, so quality stays near full multi-head while only 8 heads are cached.
- **Sliding-window attention (SWA).** Each token at layer $k$ attends only to positions $[i-W, i]$ of the previous layer ($W=4096$), making attention $O(n\,W)$ instead of $O(n^2)$. Long-range information is preserved because local attention *stacks*: by induction, after $k$ layers a position's receptive field is $\approx kW$. With 32 layers and $W=4096$, the last layer reaches back $\approx 131$K tokens.
- **Rolling buffer cache.** A fixed window means any key/value at position $\le i-W$ is never attended to again, so the cache is capped at size $W$. Position $i$ is stored at slot $i \bmod W$; once $i\ge W$ this overwrites position $i-W$, which has just left every window. At sequence length 32k with $W=4096$ this is an 8× cache-memory reduction with no quality impact. The ring is "unrotated" into chronological order before attention.
- **Pre-fill and chunking.** The prompt is known in advance, so pre-fill the cache in one parallel pass; chunk long prompts at size $W$. Each chunk attends causally to itself, within-window to the cache, and not at all to tokens older than $W$.

Config: `dim=4096`, `n_layers=32`, `head_dim=128`, `hidden_dim=14336`, `n_heads=32`, `n_kv_heads=8`, `window=4096`, `vocab=32000`.

## Code

```python
import torch
import torch.nn as nn
from typing import Optional
from dataclasses import dataclass


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    head_dim: int = 128
    hidden_dim: int = 14336
    n_heads: int = 32
    n_kv_heads: int = 8
    sliding_window: int = 4096
    vocab_size: int = 32000
    norm_eps: float = 1e-5


def repeat_kv(keys, values, repeats, dim):
    keys = torch.repeat_interleave(keys, repeats=repeats, dim=dim)
    values = torch.repeat_interleave(values, repeats=repeats, dim=dim)
    return keys, values


class Attention(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.n_heads = args.n_heads
        self.n_kv_heads = args.n_kv_heads
        self.head_dim = args.head_dim
        self.repeats = self.n_heads // self.n_kv_heads
        self.sliding_window = args.sliding_window
        self.scale = self.head_dim ** -0.5
        self.wq = nn.Linear(args.dim, args.n_heads * args.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * args.head_dim, args.dim, bias=False)

    def forward(self, x, freqs_cis, cache):
        seqlen_sum, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(seqlen_sum, self.n_heads, self.head_dim)
        xk = xk.view(seqlen_sum, self.n_kv_heads, self.head_dim)
        xv = xv.view(seqlen_sum, self.n_kv_heads, self.head_dim)
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        if cache is None:
            key, val = xk, xv
        elif cache.prefill:
            key, val = cache.interleave_kv(xk, xv)
            cache.update(xk, xv)
        else:
            cache.update(xk, xv)
            key, val = cache.key, cache.value
            key = key.view(seqlen_sum * cache.sliding_window, self.n_kv_heads, self.head_dim)
            val = val.view(seqlen_sum * cache.sliding_window, self.n_kv_heads, self.head_dim)

        key, val = repeat_kv(key, val, self.repeats, dim=1)
        xq, key, val = xq[None, ...], key[None, ...], val[None, ...]
        output = memory_efficient_attention(xq, key, val, None if cache is None else cache.mask)
        return self.wo(output.view_as(x))


def unrotate(cache, seqlen):
    position = seqlen % cache.shape[0]
    if seqlen < cache.shape[0]:
        return cache[:seqlen]
    elif position == 0:
        return cache
    else:
        return torch.cat([cache[position:], cache[:position]], dim=0)


class RotatingBufferCache:
    def __init__(self, n_layers, max_batch_size, sliding_window, n_kv_heads, head_dim):
        self.sliding_window = sliding_window
        shape = (n_layers, max_batch_size, sliding_window, n_kv_heads, head_dim)
        self.cache_k = torch.empty(shape)
        self.cache_v = torch.empty(shape)
        self.kv_seqlens = None

    def cache_positions(self, positions):
        return positions % self.sliding_window          # slot i mod W; overwrites i - W

    def to_cache_mask(self, seqlen):
        return torch.tensor([x >= seqlen - self.sliding_window for x in range(seqlen)])


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
    def forward(self, x, freqs_cis, cache):
        h = x + self.attention(self.attention_norm(x), freqs_cis, cache)
        return h + self.feed_forward(self.ffn_norm(h))


class Transformer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
        self.layers = nn.ModuleList(TransformerBlock(args) for _ in range(args.n_layers))
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)
```
