## Research question

A decoder-only transformer, during autoregressive generation, stores a key and a value vector
for every token it has processed, so that each new query can attend over the whole history
without recomputing past projections. This Key-Value (KV) cache makes generation cheap
per step, and its memory grows linearly with sequence length, so for long contexts it becomes
the dominant cost: a 70B model with a one-million-token context needs on the order of 320 GB
just for the cache, far beyond a single GPU. Reasoning models that emit thousands of
intermediate tokens and agentic systems that ingest entire codebases or documents run in this
regime. Architectural alternatives (fewer KV heads, latent attention, state-space models,
sliding windows) change or retrain the model.

The setting here is: given a *pre-trained, unmodified* transformer, drop a subset of the cached
`(k_i, v_i)` pairs so that only a small fixed-budget fraction is retained, while losing as
little generation quality as possible — with no training and no architectural change. This asks
for some measure of how much each cached pair matters to the model's output, computed from
information available *at compression time*. The compression is applied both when the whole
prompt is compressed once before generation and when the cache is trimmed repeatedly during
long generation.

## Background

**The residual stream and a per-pair importance measure.** In a transformer the hidden state
`h_t` is a "residual stream" that each block updates by addition. Writing one attention head,
the post-attention hidden state is

```
h_t^out = h_t + sum_{i=1}^t a_ti W_o v_i = h_t + sum_{i=1}^t Δh_ti,
```

where `a_ti` is the attention weight from the query at position `t` to the key at position `i`,
`v_i` is the cached value, and `W_o` is the output projection. Each cached pair `(k_i, v_i)`
thus contributes exactly one additive term `Δh_ti = a_ti W_o v_i` to the stream. The size of
that contribution, `||Δh_ti|| = a_ti · ||W_o v_i||`, measures how much pair `i` moves the
output at step `t`: it factors into how strongly the query attends to the key (`a_ti`) and how
large the resulting value update is (`||W_o v_i||`). The value-norm factor is computable from
the cache at any time. The attention factor is

```
a_ti = z_ti / sum_{j<=t} z_tj,   z_ti = exp(q_t^T k_i / sqrt(d)),
```

with `q_t` the query at position `t`.

**Distributional properties of LLM activations.** Studies of activation statistics in modern
LLMs report that the hidden states feeding the attention and MLP blocks are zero-mean,
unimodal, and well approximated by a Gaussian, `h ~ N(mu, Sigma)` (the intermediate
activations inside those blocks are instead Laplacian-like). This regularity has been used
for magnitude-based activation sparsity, and it holds even for models with QK-normalization.

**Rotary position embedding (RoPE).** Positions enter through a per-position orthonormal
rotation applied to queries and keys: `q_i = R_i W_Q h_i`, `k_i = R_i W_K h_i`, with
`R_i in R^{d x d}` the RoPE rotation at position `i` (in the standard implementation,
`R_i x = x ⊙ cos_i + rotate_half(x) ⊙ sin_i`). The rotation makes `q_t^T k_i` depend
on the relative offset `t - i`, so a query's logit against a key carries that query position's
rotation.

**The attention-sink phenomenon.** The first few tokens of a sequence receive disproportionate
attention across essentially all layers and heads, regardless of their semantic content — one
reading is that softmax must place its mass somewhere and parks the excess on early,
always-visible tokens. These positions also carry unusually large ("massive") activations.
Both facts mean the first handful of tokens behave differently from the rest of the cache.

**The deployment constraint.** Production attention kernels (Flash Attention and its
successors) compute the softmax-weighted sum on the fly and never materialize the full
`t x t` attention matrix. The quantities available at deployment are the cached keys, values,
and hidden states, along with the module's projections and RoPE.

## Baselines

**Attention-score eviction — H2O (Zhang et al. 2024), SnapKV (Li et al. 2025), TOVA (Oren et
al. 2024).** These rank a cached pair by the attention it has received. H2O keeps "heavy
hitters" by accumulated attention mass over the observed steps; SnapKV pools the attention
that an observation window of recent (typically user-question) tokens pays to the context and
keeps the most-attended keys; TOVA ranks by the most recent query's attention. Core math: all
read entries of the realized attention matrix `a_ti` and keep the largest. The signal comes
from queries already observed, and SnapKV's observation window assumes a question follows the
context.

**Position heuristics — StreamingLLM (Xiao et al. 2023), H2O's recency component.**
StreamingLLM keeps a small set of initial "sink" tokens plus a sliding window of the most
recent tokens, discarding the middle; this stabilizes streaming to millions of tokens. It is
purely positional, scoring by position rather than content.

**Norm / embedding heuristics — KNorm (Devoto et al. 2024), KeyDiff, Q-Filters.** KNorm keeps
the keys with the smallest L2 norm; KeyDiff uses distances between key embeddings; Q-Filters
projects keys onto an SVD direction. These need only the cached keys, so they are Flash-
Attention compatible and cheap, scoring keys by a geometric proxy.

**Head-adaptive budgeting — AdaKV (Feng et al. 2024), PyramidKV.** Rather than a new scoring
rule, these reallocate a fixed total budget across heads/layers, since heads differ in how much
compression they tolerate. They operate on top of whatever per-pair score they are given.

## Evaluation settings

The natural long-context yardsticks, all pre-existing:

- **LongBench** — long-context tasks across single- and multi-document QA, summarization,
  few-shot learning, synthetic tasks, and code completion; per-task native scores.
- **RULER** — synthetic long-context probes at fixed context lengths (e.g. 4k, 16k): needle
  retrieval, variable tracking (multi-hop), and common/frequent-word extraction (aggregation);
  average score reported against the compression ratio.
- **Needle-in-a-Haystack** — a target fact ("needle") planted at varying depths in a long
  distractor context (up to ~125k tokens); retrieval success as a function of needle depth and
  context length.
- **Reasoning decoding** — AIME-25 and MATH-500, run on reasoning models that emit long
  chain-of-thought, where the cache grows during generation; the cache is allowed to reach a
  size and then trimmed, reported at `n x` compression.
- Models span instruction-tuned long-context families (Llama-3.1-8B, Qwen3-8B, Gemma3-12B for
  prefilling; distilled reasoning models for decoding). Protocol sweeps the compression ratio
  and, for prefilling, compresses the context with no assumed downstream question. Memory is
  measured as peak GPU usage vs. sequence length.

## Code framework

The substrate is a forward hook attached to each attention layer of a Hugging Face
transformer: after a layer's forward pass the hook receives that layer's hidden states and the
freshly written cached keys and values, decides which pairs to keep, and writes the trimmed
cache back before the next layer runs. The model, the cache object, RoPE, and the eviction
plumbing (a top-`k` keep-the-highest-score selection) already exist; what does not exist is the
rule that turns a layer's `(hidden_states, keys, values)` into a per-pair importance score.
That rule is the single empty slot.

```python
from typing import Any

import torch
from torch import nn


class CompressionPolicy:
    """Per-layer KV-cache compression run from a forward hook on an attention layer.
    Owns one thing: a rule that scores each cached KV pair so the eviction step can
    keep the highest-scoring `n_kept` and drop the rest. Must read only what is on
    hand at compression time -- the layer's hidden states, cached keys/values, and
    the module's projections/RoPE -- never a materialized attention matrix."""

    compression_ratio: float = 0.0

    def score(self, module: nn.Module, hidden_states: torch.Tensor,
              keys: torch.Tensor, values: torch.Tensor,
              attentions: torch.Tensor | None, kwargs: dict[str, Any]) -> torch.Tensor:
        # hidden_states: (batch, seq, hidden)  -- input to this attention layer
        # keys, values:  (batch, n_kv_heads, seq, head_dim)  -- the cache to be pruned
        # returns:       (batch, n_kv_heads, seq)  -- higher = keep
        # TODO: the per-pair importance rule we will design.
        raise NotImplementedError

    @torch.no_grad()
    def compress(self, module, hidden_states, keys, values, attentions=None, kwargs=None):
        if self.compression_ratio == 0:
            return keys, values
        scores = self.score(module, hidden_states, keys, values, attentions, kwargs or {})
        k_len = keys.shape[2]
        n_kept = int(k_len * (1 - self.compression_ratio))
        idx = scores.topk(n_kept, dim=-1).indices          # keep highest-scoring pairs
        idx = idx.unsqueeze(-1).expand(-1, -1, -1, module.head_dim)
        keys = keys.gather(2, idx).contiguous()
        values = values.gather(2, idx).contiguous()
        return keys, values
```

The eviction loop is fixed; the body of `score` is exactly what the method must supply.
