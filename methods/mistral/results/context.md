# Context

## Research question

The dominant way to make a language model better has been to make it bigger, but every extra parameter raises the cost and latency of *serving* the model, which is the barrier to actually deploying it in real applications. The problem is to get high performance at a small, fixed, efficient inference cost — to design a compact model (around 7B parameters) whose decoding is cheap enough for real-time use and whose memory footprint at inference is small enough to allow large batches and long inputs. Concretely, a solution has to attack the two things that make autoregressive Transformer inference expensive: the key/value cache that must be streamed from memory on every decoding step (which caps batch size and throughput), and the attention operation whose compute is quadratic and whose memory is linear in the sequence length (which makes long contexts costly in both latency and cache size). It must do this without giving up the ability to use information from far back in a long sequence.

## Background

**The decoder Transformer substrate.** A modern decoder-only language model is a stack of pre-normalized residual blocks, each with a multi-head self-attention sub-layer and a position-wise feed-forward sub-layer, using RMSNorm for normalization, a SwiGLU feed-forward, and rotary position embeddings (RoPE) applied to queries and keys at every layer. This is the standard substrate a 7B-scale model is built on.

**Where autoregressive inference spends its cost.** Generating one token requires reading the model weights and the cached keys and values for every past position, then doing a small amount of arithmetic to emit one distribution. The cache reads dominate as the sequence grows: at step $i$ the keys and values have shape $[\text{heads}, i, \text{head\_dim}]$ per sequence, so the per-step memory traffic grows with both the number of heads and the length of the history. This makes decoding memory-bandwidth-bound, and the cache size — not the arithmetic — is what limits how many sequences can be batched and how long they can be.

**Multi-head attention and its cache cost.** Standard multi-head attention learns a separate key, value, and query projection per head; the cache therefore stores one key and one value vector *per head per position*. Note the asymmetry in what accumulates: the query brought to a decode step is the current token's, tiny and never cached, while keys and values pile up — so the cache cost is set by how many distinct key/value heads are stored. Schemes that store fewer key/value heads than query heads have been studied (Shazeer, 2019; Ainslie et al., 2023); see Baselines.

**Attention cost in sequence length.** Vanilla self-attention forms the full $n\times n$ score matrix: $O(n^2)$ operations and, with a growing cache, $O(n)$ stored key/value vectors per token. For long inputs this is the dominant cost. Sparse and local attention patterns restrict each query to a subset of keys to cut this — for instance attending only to a fixed-width band of recent positions (Child et al., 2019; Beltagy et al., 2020) — at the price of each layer no longer directly seeing the whole past.

**Efficient attention kernels.** Kernels such as FlashAttention (Dao et al., 2022) and xFormers (2022) compute attention without materializing the full score matrix and support masked/local patterns, so a restricted attention pattern can be turned into an actual wall-clock speedup.

## Baselines

**Multi-head attention (Vaswani et al., 2017).** Each of $h$ heads has its own query, key, value projections; the output projection mixes the concatenated head outputs. Core idea and math: $\text{softmax}(QK^\top/\sqrt{d})V$ per head over the full causal history. The gap: the cache holds $h$ key and $h$ value vectors per position, making decode-time memory traffic and cache size large — the throughput bottleneck.

**Multi-query attention / grouped-query attention (Shazeer, 2019; Ainslie et al., 2023).** Reduce the number of key/value heads (to one, or to $G$ groups) while keeping $h$ query heads, cutting the cache by $h$ or $h/G$. Core idea: query heads in a group share one key/value head. The gap a pure choice still leaves: it shrinks the cache by a constant factor but does nothing about the *growth* of the cache and attention cost with sequence length — a long enough sequence is still quadratic in compute and linear-and-unbounded in cache.

**Full (dense causal) attention with a growing KV cache.** The standard serving setup stores every past key/value and attends over all of them. Core idea: exact attention to the entire history. The gap: the cache grows without bound with sequence length, so long-context serving is memory-limited, and each step pays to stream an ever-larger cache.

**Local / sliding-window sparse attention (Child et al., 2019; Beltagy et al., 2020).** Restrict each query to a fixed window of recent keys, making attention linear in length. Core idea: a banded attention mask. The gap: naively, restricting the window means a query never looks more than a window back, which appears to throw away long-range information that language depends on.

## Evaluation settings

The natural yardsticks are the standard zero-/few-shot LLM suites — commonsense reasoning, world knowledge and reading comprehension, mathematics (GSM8K, MATH), and code (HumanEval, MBPP), plus aggregate multitask benchmarks (MMLU) — alongside held-out language-modeling perplexity, all compared against open models of similar and larger size. The efficiency-relevant metrics are decode throughput and latency, the key/value cache memory footprint as a function of sequence length, and the maximum batch size and context length that fit in a given memory budget. These benchmarks and metrics predate the model and are the yardstick it would be measured on.

## Code framework

The primitives that already exist: PyTorch modules, a memory-efficient attention kernel that accepts an arbitrary attention mask, an RMSNorm, a SwiGLU feed-forward, and rotary position embeddings. The decoder is the usual stack of pre-norm residual blocks; what needs filling in is the self-attention and its key/value cache.

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
    n_kv_heads: int = 8
    vocab_size: int = 32000
    norm_eps: float = 1e-5


class Attention(nn.Module):
    # TODO: the self-attention sub-layer and its key/value cache.
    def __init__(self, args):
        super().__init__()
        pass
    def forward(self, x, freqs_cis, cache):
        pass


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
```
