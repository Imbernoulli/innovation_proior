## Research question

A decoder-only transformer caches, for every layer and every key-value head, the Key and Value
tensors of all tokens it has seen — the KV cache. During long-context inference this cache is
the dominant cost on two axes at once. Memory: it grows linearly with sequence length and, for
a multi-billion-parameter model on a tens-of-thousands-of-tokens prompt, reaches the size of
the model weights themselves. Compute: every decode step attends over the entire cached
sequence, so per-step latency grows linearly in the cache length and the prefill attention is
quadratic in the prompt length. Reasoning-style models that emit long "thinking" traces make
both worse.

The precise goal is a controller that, after prefill, ranks the cached prefill KV entries by
importance and keeps only a small fixed-budget subset (e.g. 20% of tokens) so that decoding can
proceed from the reduced cache without losing long-context task quality.

## Background

**The autoregressive substrate and the KV cache.** In a decoder-only transformer, tokens are
embedded into `X ∈ R^{n×d}` and each layer maps `X` to query/key/value states per head `i` via
linear projections `Q_i = X W_i^Q`, `K_i = X W_i^K`, `V_i = X W_i^V` with head dimension `d_h`,
then forms the output `Y = Concat_i(A_i V_i) W^O` where `A_i = softmax(Q_i K_i^T / sqrt(d_h))`.
During decode the model appends one token at a time: its `k_i, v_i` are concatenated onto the
cached `K_i, V_i`, and the single new query attends over the whole cache. Caching `K_i, V_i`
is what makes decode fast, and is also exactly what grows without bound.

**Grouped-query attention.** Modern models (GQA, Ainslie et al. 2023) share one KV head across
several query heads, so `num_key_value_heads < num_attention_heads` and the cache — and any
eviction decision — is made per KV head, not per query head. An importance score therefore has
shape `(batch, num_kv_heads, seq_len)`.

**FlashAttention as a hard constraint.** The de-facto attention kernel (Dao 2023) computes the
softmax-weighted output without ever forming the `n×n` matrix `A_i`. Any compression method
that needs the per-token attention weights cannot run on top of it, which in a production stack
makes it impractical.

**Two diagnostic findings about the KV space.** Work on KV *quantization* mapped out the
structure of the cached tensors:

- *Token-wise locality* (CacheGen, Liu et al. 2024, Insight 1): within the same layer and
  channel, tokens closer together in position have more similar K/V values than tokens far
  apart. Concretely, the distribution of the delta between consecutive tokens' K (or V) values
  is tightly concentrated near zero — adjacent KV vectors barely move. This is a direct
  consequence of the autoregressive nature: the next representation does not jump abruptly from
  the previous one.
- *Key/value distributional asymmetry* (KIVI, Liu et al. 2024): the key cache has a few fixed
  channels whose magnitudes are very large — persistent per-channel outliers — so keys should
  be quantized per channel (grouping along the channel dimension) to confine the error; the
  value cache has no such obvious channel outlier pattern and is best quantized per token.
  Keys carry structure along the channel axis; values along the token axis.

**The attention-sink phenomenon.** StreamingLLM (Xiao et al. 2023) observed that softmax forces
the attention weights of a query to sum to one, so when the query has no strong match it dumps
the leftover mass onto the first few tokens, which — being visible to every later position — get
trained into "sinks" that absorb large attention regardless of semantic content. Keeping the KV
of ~4 initial sink tokens, plus a recent sliding window, keeps the attention distribution close
to normal and recovers fluent generation over arbitrarily long streams.

## Baselines

**StreamingLLM / window attention with sinks (Xiao et al. 2023; LM-Infinite, Han et al. 2024).**
Keep the first `S` sink tokens and the most recent `L` tokens in a sliding window; evict
everything else. O(TL), constant memory, query-free, FlashAttention-compatible.

**H2O — Heavy-Hitter Oracle (Zhang et al. 2023).** Accumulated attention scores across the
sequence follow a power law, so a small set of "heavy hitter" tokens receives most of the
attention; H2O keeps recent tokens plus the running heavy hitters, scoring each token by the
sum of attention it has received, and is near-optimal under a submodularity assumption.

**SnapKV (Li et al. 2024).** An "observation window" at the end of the prompt reveals which
prefix positions each head attends to; SnapKV clusters and pools those positions and keeps
them together with the observation window.

**L2-norm key selection (Devoto et al. 2024).** Empirically a *low* L2 norm of a key embedding
correlates with a *high* attention score during decoding, so keep the keys with the lowest L2
norm (and their values), skipping the first couple of layers. FlashAttention-compatible.

**Quantization (KIVI, Liu et al. 2024; CacheGen, Liu et al. 2024; PyramidInfer).** Represent
the KV cache at reduced precision (e.g. 2-bit), with per-channel keys and per-token values to
control error. Large memory savings with little quality loss.

## Evaluation settings

- **Models.** Open instruction-tuned LLMs that use GQA, so eviction is per KV head:
  Llama-3.1-8B-Instruct and Qwen2.5-7B-Instruct; a smaller instruction-tuned model can be used
  for quick harness-level checks. Compression is applied per layer, after prefill, on the cache.
- **Benchmarks.** RULER-16K (Hsieh et al. 2024): thirteen synthetic long-context subtasks —
  single/multi-key and multi-value needle-in-a-haystack, multi-query, variable tracking,
  common- and frequent-word extraction, and QA. LongBench (Bai et al. 2024): single- and
  multi-document QA, summarization, few-shot, synthetic, and code, scored by the task-native
  metric. A 64-digit passkey-retrieval needle-in-a-haystack with Paul Graham essays as filler,
  reported as both partial- and exact-match. GSM8K final-answer accuracy in the shared harness.
- **Protocol.** Compression is run *without* the question appended (compress-then-query) so the
  comparison is not biased by query-aware methods; the natural prior-art comparison set is
  SnapKV, StreamingLLM, and H2O. Compression budgets are swept from mild to aggressive, and the
  standard measurements are retained fraction, task score, and wall-clock runtime.

## Code framework

The controller plugs into a KV-cache compression harness that already exists. The harness owns
the model, the dataset, the decode loop, and the cache; it registers a forward hook on each
attention layer that fires once, during prefill, after the layer has written its keys and
values into the cache. The hook hands the policy the layer's cached `keys` and `values`
(shape `(batch, num_kv_heads, seq_len, head_dim)`), a target budget, and asks for two things:
a per-token importance score, and the selection of which tokens to keep. The base class already
implements the generic "score then keep the top-`n_kept`" loop; everything method-specific
lives in the scoring rule, which is the one empty slot.

```python
import torch
from torch import nn
from dataclasses import dataclass


@dataclass
class ScorerPress:
    """Generic score-based KV eviction. The harness calls compress() during prefill;
    a subclass supplies score(). keys/values are (batch, num_kv_heads, seq_len, head_dim);
    the returned score is (batch, num_kv_heads, seq_len), higher = keep."""

    compression_ratio: float = 0.0

    def score(self, module, hidden_states, keys, values, attentions, kwargs) -> torch.Tensor:
        # TODO: the per-token importance rule we will design.
        #       Given only the cached keys and values (no attention weights, no query),
        #       return a score per token; higher means keep.
        raise NotImplementedError

    def compress(self, module, hidden_states, keys, values, attentions, kwargs):
        if self.compression_ratio == 0:
            return keys, values
        scores = self.score(module, hidden_states, keys, values, attentions, kwargs)
        k_len = keys.shape[2]
        n_kept = int(k_len * (1 - self.compression_ratio))
        indices = scores.topk(n_kept, dim=-1).indices                 # global top-K over the cache
        indices = indices.unsqueeze(-1).expand(-1, -1, -1, module.head_dim)
        keys = keys.gather(2, indices).contiguous()                   # gather the kept KV
        values = values.gather(2, indices).contiguous()
        return keys, values
```

The harness applies one global top-`n_kept` over the whole cached sequence. The slot to fill is
`score(...)` — a per-token rule that uses only the keys and values (the attention matrix is not
available under the fused kernel) and is independent of the query.
