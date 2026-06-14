## Research question

Long-context LLM inference keeps a Key-Value tensor for every token in every attention layer; that
cache grows linearly with the prompt and on a 30k-token document it rivals the model weights, while
every decode step attends over the whole of it. The single thing being designed here is a **KV
token-retention controller**: after a standard full-attention prefill, score the cached prefill KV
entries, keep a small fixed-budget subset (the canonical budget retains ~20% of prefill tokens), and
decode from that subset without losing long-context task quality. Everything else — the model, the
datasets, the prompt templates, the decode loop, the budget enforcement — is fixed. The only degree of
freedom is the per-token scoring rule and the metadata that drives it.

## Prior art before the first rung (KV-cache lineage)

The scaffold the first rung fills is itself the resolution of how to *run* a masked decoder's
generation loop at all; the methods below are the lineage every later retention rule reacts to.

- **Scaled dot-product attention (Vaswani et al. 2017).** A layer reads `softmax(QK^T/sqrt(d_k))V`
  over all positions; in training this is one batched matmul over the whole sequence. But generation is
  `p(y)=prod_t p(y_t|y_{<t},x)`, sequential by construction, so a single parallel pass is unavailable.
  Gap: the parallelism that makes training fast is gone at decode time.
- **The incremental KV cache (the full-attention loop; Shazeer 2019 names the cost).** Because the
  causal mask makes `k_j,v_j` *frozen* the instant position `j` is produced — no later token can revise
  them — each step need only project the one new token and append its `(k,v)`, attending the lone new
  query over the stored history. Arithmetic drops from the naive re-run's `Theta(b n^2 d^2)` to
  `Theta(b n d^2)`. Gap: it evicts nothing, so the cache and the per-step bandwidth to reload it grow
  *linearly* in sequence length — the memory wall this task exists to attack.
- **Sliding-window / recent-only cache.** Cap attention to the last `L` tokens, evict the oldest.
  Constant memory and latency. Gap: quality collapses at a cliff the moment the *initial* tokens leave
  the cache, and any dependency older than the window is silently dropped — it cannot keep a fact in
  the middle of a long context.
- **Quantization of the cache (KIVI, CacheGen).** Store every token's KV at 2 bits instead of 16.
  Shrinks memory a lot, but keeps *every* token, so the attention compute — quadratic prefill,
  linear-in-cache decode — is unchanged. Gap: it does not reduce the number of attended tokens, which
  is what a budgeted controller must do.
- **Attention-score eviction (H2O, SnapKV).** Score a token by the attention mass it has received and
  keep the heavy hitters. Gap for *this* harness: it needs the materialized `t x t` attention matrix,
  which the production kernel (SDPA / FlashAttention) never builds and which the scoring hook below is
  not handed; and it is query-dependent, so the kept set changes with the question. Disqualified here
  by construction.

The substrate keeps the full incremental cache during prefill and then asks the controller to throw
tokens away under a fixed budget — using only what the hook exposes (cached keys, values, hidden
states), never a realized attention matrix.

## The fixed substrate

A full-attention replay harness is frozen and must not be touched. It owns the model
(`Qwen/Qwen2.5-3B-Instruct`, aligned with `llm-kv-adaptive-quantization`), the datasets and their
prompt templates, the fixed cache budget (`SELECTION_KV_COMPRESSION_RATIO=0.8`, i.e. retain ~20% of
prefill tokens), the greedy decode loop, and the scoring. The model is run once over the prompt as a
standard full-attention prefill; a per-layer forward hook
(`PrefillSelectionCompressor.forward_hook`) then fires on every attention layer with the freshly
filled cache and invokes the editable policy to compress that layer's prefill KV in place, after which
generation proceeds from the compressed cache. The harness force-overrides the compression ratio at
the call site (a policy cannot lie about the budget), measures the actually-retained fraction from the
policy's output (`n_kept / k_len` per layer, averaged), and wires `module.rotary_emb` onto each
attention module so a policy that re-rotates kept keys has the rotary table available.

## The editable interface

Exactly one region is editable — the `SelectionPolicy` class in
`transformers-kv-lab/custom_selection_eval.py` (lines 40-101). Every rung on the ladder is a fill of
this same three-method contract:

- `retention_plan(layer_id, request_meta, cache_meta)` — return a dict; the policy's private channel
  from `retention_plan` to `score_tokens`. The harness *enforces* `disable_compression` (skip scoring,
  report retained = 1.0) and *force-overrides* `compression_ratio` to its own value; everything else
  (`sink_tokens`, `lag_size`, `n_future_positions`, ...) is advisory, read only by the policy's own
  `score_tokens`.
- `score_tokens(module, hidden_states, keys, values, kwargs, plan)` — return a per-token score tensor
  of shape `(batch, num_kv_heads, k_len)`, higher = keep, or `None` to keep everything. **No attention
  tensor is passed** — scoring may read only keys, values, hidden states, and the module's weights.
- `select_cache(module, keys, values, scores, n_kept)` — return `(selected_keys, selected_values)`;
  the harness has already computed `n_kept = int(k_len * (1 - compression_ratio))` and will take the
  policy's output as the new cache. A class attribute `rerotate_selected_keys` tells the harness whether
  decode positions must follow the re-rotated cache.

The starting point is the scaffold default. The shipped template happens to carry a sink+window fill;
the **weakest budgeted submission** is the no-compression anchor — disable compression and keep the
full cache — which is the first rung below. Each later method replaces exactly this class and nothing
else.

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

Five public text workloads, one seed (42), three reported signals per workload — `final_score`
(benchmark-native task accuracy, 0-100), `mean_retained_fraction` (average retained prefill KV
fraction after the policy runs), and `runtime_seconds`:

| Label | Source | Final score |
|---|---|---|
| `longbench_hotpotqa` | LongBench-E `hotpotqa_e` | LongBench QA F1 (0-100) |
| `longbench_passage_retrieval` | LongBench-E `passage_retrieval_en_e` | LongBench retrieval (0-100) |
| `longbench_repobench` | LongBench-E `repobench-p_e` | LongBench code similarity (0-100) |
| `longbench_v2` | LongBench v2 `train` split | multiple-choice exact accuracy (0-100) |
| `gsm8k` | `openai/gsm8k` main test split | exact final-answer accuracy (0-100) |

The leaderboard combines, per workload, three normalized terms — accuracy, runtime, cache reduction —
with weights `accuracy:time:reduction = 6:2:2`, and takes the geometric mean across the five
workloads. A soft upper-bound penalty fires when `mean_retained_fraction > 0.25`, so the
no-compression anchor stays *visible* as a reference but is not a valid budgeted submission: it pays
the budget penalty in full.
