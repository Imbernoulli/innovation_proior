## Research question

Long-context LLM inference stores a key-value tensor for every prefill token in every attention layer. The cache grows linearly with prompt length; on a 30k-token document it rivals the model weights, and every decode step attends over the whole cache. How can the prefill KV cache be compressed to a small fixed-budget subset — retaining roughly 20% of prefill tokens — while preserving long-context task quality?

## Prior art / Background / Baselines

- **Scaled dot-product attention.** A layer computes `softmax(QK^T/sqrt(d_k))V` over all positions in one parallel matmul during training, but generation is sequential.
- **Incremental KV cache.** Because the causal mask freezes each `(k_j, v_j)` once position `j` is produced, each decode step only projects the new token, appends its KV, and attends the new query over the stored history.
- **Sliding-window / recent-only cache.** Keep only the last `L` tokens and evict the oldest.
- **Quantization of the cache (e.g., KIVI, CacheGen).** Store every token's KV at lower precision.
- **Attention-score eviction (e.g., H2O, SnapKV).** Score tokens by the attention mass they receive and keep the heavy hitters.

The substrate keeps the full incremental cache during prefill and then asks the controller to discard tokens under a fixed budget, using only the hook's inputs (cached keys, values, hidden states) and never a realized attention matrix.

## Fixed substrate / Code framework

A full-attention replay harness is frozen and must not be touched. It owns the model (`Qwen/Qwen2.5-3B-Instruct`), the datasets and prompt templates, the fixed cache budget (`SELECTION_KV_COMPRESSION_RATIO=0.8`, i.e. retain ~20% of prefill tokens), the greedy decode loop, and the scoring invocation. The model runs once over the prompt as a standard full-attention prefill; a per-layer forward hook (`PrefillSelectionCompressor.forward_hook`) then fires on every attention layer with the freshly filled cache and invokes the editable policy to compress that layer's prefill KV in place, after which generation proceeds from the compressed cache. The harness force-overrides the compression ratio at the call site, measures the actually retained fraction from the policy's output (`n_kept / k_len` per layer, averaged), and wires `module.rotary_emb` onto each attention module so a policy that re-rotates kept keys has the rotary table available.

## Editable interface

Exactly one region is editable — the `SelectionPolicy` class in `transformers-kv-lab/custom_selection_eval.py` (lines 40-101). Each submission is a fill of this same three-method contract:

- `retention_plan(layer_id, request_meta, cache_meta)` — return a dict; the policy's private channel from `retention_plan` to `score_tokens`. The harness enforces `disable_compression` (skip scoring, report retained = 1.0) and force-overrides `compression_ratio` to its own value; everything else (`sink_tokens`, `lag_size`, `n_future_positions`, ...) is advisory, read only by the policy's own `score_tokens`.
- `score_tokens(module, hidden_states, keys, values, kwargs, plan)` — return a per-token score tensor of shape `(batch, num_kv_heads, k_len)`, higher = keep, or `None` to keep everything. No attention tensor is passed — scoring may read only keys, values, hidden states, and the module's weights.
- `select_cache(module, keys, values, scores, n_kept)` — return `(selected_keys, selected_values)`; the harness has already computed `n_kept = int(k_len * (1 - compression_ratio))` and uses the policy's output as the new cache. A class attribute `rerotate_selected_keys` tells the harness whether decode positions must follow the re-rotated cache.

The starting point is the scaffold default shown below: a no-compression anchor that disables compression and keeps the full cache.

```python
# EDITABLE region of custom_selection_eval.py (lines 40-101) — default fill: full-attention anchor
class SelectionPolicy:
    """Naive full-attention anchor: keep every prefill KV token."""

    method_name = "full_attention"
    rerotate_selected_keys = False

    def retention_plan(self, layer_id, request_meta, cache_meta):
        return {
            "method": self.method_name,
            "disable_compression": True,            # harness skips scoring, reports retained = 1.0
        }

    def score_tokens(self, module, hidden_states, keys, values, kwargs, plan):
        return None                                  # no scoring: keep everything

    def select_cache(self, module, keys, values, scores, n_kept):
        return keys, values                          # no eviction
```

## Evaluation settings

Five public text workloads, one seed (42), three reported signals per workload — `final_score` (benchmark-native task accuracy, 0-100), `mean_retained_fraction` (average retained prefill KV fraction after the policy runs), and `runtime_seconds`:

| Label | Source | Final score |
|---|---|---|
| `longbench_hotpotqa` | LongBench-E `hotpotqa_e` | LongBench QA F1 (0-100) |
| `longbench_passage_retrieval` | LongBench-E `passage_retrieval_en_e` | LongBench retrieval (0-100) |
| `longbench_repobench` | LongBench-E `repobench-p_e` | LongBench code similarity (0-100) |
| `longbench_v2` | LongBench v2 `train` split | multiple-choice exact accuracy (0-100) |
| `gsm8k` | `openai/gsm8k` main test split | exact final-answer accuracy (0-100) |

The leaderboard combines, per workload, three normalized terms — accuracy, runtime, cache reduction — with weights `accuracy:time:reduction = 6:2:2`, and takes the geometric mean across the five workloads. A soft upper-bound penalty fires when `mean_retained_fraction > 0.25`, so the no-compression anchor stays visible as a reference but is not a valid budgeted submission: it pays the budget penalty in full.
