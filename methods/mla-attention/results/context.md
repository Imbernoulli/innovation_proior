# Context: efficient attention for autoregressive decoding

## Research question

A decoder-only Transformer generates text one token at a time. To produce token *t*, each attention layer must attend over the keys and values of all preceding tokens. To avoid recomputing those from scratch at every step, the standard trick is to keep a **KV cache**: after a token is processed, its per-layer keys and values are stored, and each new query reads the whole cache.

This cache is the problem. For multi-head attention it holds `2·n_h·d_h·l` scalars *per token* (keys and values, across all `n_h` heads of dimension `d_h`, in all `l` layers). At long context and large batch it grows until it no longer fits in accelerator memory, and — more insidiously — decoding becomes **memory-bandwidth bound**: every generation step streams the entire cache through the chip to do a relatively tiny amount of arithmetic (one new query dotted against the stored keys, then a weighted sum of the stored values). The cache size directly caps the maximum batch size and sequence length, and the bandwidth cost dominates per-token latency.

The goal: shrink the per-token KV cache by a large factor — ideally an order of magnitude — **without** giving up the modelling quality of full multi-head attention. A solution has to attack the bytes-cached-per-token quantity itself, and it has to remain compatible with whatever positional-encoding scheme the model uses for long context.

## Background

**Multi-head attention and the decode bottleneck.** In a Transformer attention layer, the input `h_t ∈ R^d` of token *t* is mapped to queries, keys, and values by `W^Q, W^K, W^V`, each sliced into `n_h` heads of width `d_h`. Head *i* computes `o_{t,i} = Σ_{j≤t} softmax_j( q_{t,i}·k_{j,i} / √d_h ) v_{j,i}`, and the head outputs are concatenated and mixed by `W^O`. During training the whole sequence is processed in parallel, so this is compute-bound and efficient. During **incremental decoding** parallelism over the sequence is gone: each step appends one token, and the cost is dominated by repeatedly loading the large stored key and value tensors from memory (Shazeer, 2019). The arithmetic-to-memory ratio is low, so the operation is limited by bandwidth, not flops. This is why the KV cache, not the parameter count, governs decode throughput.

**Rotary position embedding (RoPE).** Modern long-context decoders encode position with RoPE (Su et al., 2021). RoPE multiplies the query and key of a token at position *m* by a block-diagonal rotation `R_m` (rotating coordinate pairs by angles `m·θ_i`, with `θ_i = base^{-2i/d}`) *before* the dot product. The defining property is `(R_m q)^T (R_n k) = q^T R_{n-m} k`: the attention score depends only on the **relative** offset `n−m`. The mechanism is load-bearing for the present problem in a subtle way — the rotation sits *between* the query and the key, and it is position-dependent, so it does **not** commute with arbitrary left-multiplication applied to `q` or `k`. Any restructuring of the key/value computation has to respect that the rotation cannot be moved out of the way.

**Diagnostic finding — sharing keys and values across heads costs quality.** A controlled comparison of three 7B dense models, identical except for the attention mechanism and with parameters realigned to ~7B by adjusting depth, trained on the same 1.33T tokens, measures the quality price of cache reduction on hard benchmarks:

| Benchmark (metric) | MQA (1 KV head) | GQA (8 groups) | MHA |
|---|---|---|---|
| BBH (3-shot, EM) | 33.2 | 35.6 | **37.0** |
| MMLU (5-shot, acc) | 37.9 | 41.2 | **45.2** |
| C-Eval (5-shot, acc) | 30.0 | 37.7 | **42.9** |
| CMMLU (5-shot, acc) | 34.6 | 38.4 | **43.5** |

Full multi-head attention is uniformly best; collapsing keys and values onto fewer heads (GQA) is worse, and collapsing onto one (MQA) is worst. So the cheap routes to a small cache trade away accuracy in a way that shows up clearly on demanding evaluations.

## Baselines

**Multi-Query Attention — MQA (Shazeer, 2019).** Keep `n_h` separate query heads but let *all* of them share a *single* key head and a *single* value head. The KV cache shrinks from `2·n_h·d_h·l` to `2·d_h·l` per token — a factor of `n_h`. This is the most aggressive cut available by head-sharing and it directly relieves the bandwidth bottleneck. The gap it leaves: with only one key/value subspace, every head attends through the same low-rank lens; reported quality degrades, mildly on easy tasks and more sharply on harder ones.

**Grouped-Query Attention — GQA (Ainslie et al., 2023).** Interpolate between MHA and MQA: partition the `n_h` query heads into `G` groups and give each group its own shared key/value head. `G = n_h` recovers MHA; `G = 1` recovers MQA. The cache is `2·n_g·d_h·l` per token (`n_g = G`), so `G` tunes the quality/memory trade-off. GQA also comes with an uptraining recipe — convert an existing MHA checkpoint by mean-pooling each group's key/value heads, then continue training with ~5% of the original pre-training compute. The gap it leaves: it is still strictly on the same Pareto frontier as MQA — to approach MHA quality you need `G` large, which means the cache stays large; you cannot have both small cache *and* MHA-level quality. Every head in a group still shares one key/value subspace.

Both baselines reduce the cache by **throwing away key/value subspaces** (forcing heads to share). That is the move whose quality cost the diagnostic above quantifies.

## Evaluation settings

The natural yardstick is a like-for-like comparison at fixed parameter and token budgets, varying only the attention mechanism. The protocol that exists at this time:

- **Quality vs. cache trade-off** measured on hard few-shot benchmarks: BBH (3-shot, exact match), MMLU (5-shot accuracy), and the Chinese suites C-Eval and CMMLU (5-shot accuracy). These stress reasoning and broad knowledge, where cheap attention variants visibly lose ground.
- **Controlled ablation design:** models sharing one architecture except for the attention block, with parameter counts realigned (by adjusting the number of layers) so the comparison isolates the attention mechanism, trained on a fixed token budget (e.g. 1.33T tokens for the 7B-scale study).
- **Efficiency metric:** KV cache per token, counted in number of cached elements per layer (storage-precision-agnostic), since that is the quantity that governs both the memory ceiling and the decode bandwidth.
- **Long-context behaviour:** retrieval-style probes (needle-in-a-haystack) over contexts up to ~128K, to confirm the positional scheme still works once attention is restructured.

## Code framework

The scaffold is a standard decoder attention block: linear projections, per-head scaled-dot-product attention, an output projection, RoPE applied to queries and keys, and an append-only KV cache for decoding. The key/value projection path is the empty slot.

```python
import torch
import torch.nn as nn

def apply_rope(x, cos, sin, position_ids):
    # rotate coordinate pairs by position-dependent angles (Su et al., 2021)
    pass

class DecoderAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim

        # Projections: queries / keys / values / output.
        # TODO: define the query path
        # TODO: define the key/value path
        self.o_proj = nn.Linear(self.num_heads * self.head_dim,
                                self.hidden_size, bias=False)

    def project_query(self, hidden_states):
        # TODO: produce per-head queries (with RoPE)
        pass

    def project_key_value(self, hidden_states):
        # TODO: produce per-head keys and values for this token.
        pass

    def forward(self, hidden_states, position_ids, past_key_value=None, attention_mask=None):
        bsz, q_len, _ = hidden_states.size()

        q = self.project_query(hidden_states)          # (b, n_h, q_len, head_dim)
        k, v = self.project_key_value(hidden_states)   # (b, n_h, q_len, *)

        if past_key_value is not None:
            k, v = past_key_value.update(k, v)         # append to the KV cache

        scale = self.head_dim ** -0.5
        scores = torch.matmul(q, k.transpose(2, 3)) * scale
        if attention_mask is not None:
            scores = scores + attention_mask
        probs = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)

        out = torch.matmul(probs, v)                   # (b, n_h, q_len, head_dim)
        out = out.transpose(1, 2).reshape(bsz, q_len, -1)
        return self.o_proj(out)
```
