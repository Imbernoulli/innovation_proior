**Problem.** Long-context inference keeps a per-layer KV tensor for every token; the cache grows
linearly and the decode step is bandwidth-bound on reloading it. The task is to compress that cache to
a fixed budget (~20% retained) with minimal quality loss. This first rung does the opposite: it keeps
everything, to pin the quality ceiling and the budget penalty every real compressor must clear.

**Key idea.** In a causal layer, `k_j, v_j` are frozen the instant position `j` is produced — no later
token can revise them — so keeping the entire frozen set makes every decode step attend over the
genuine complete history. The decoded output follows the same computation as re-running the model on the
full prefix. This is exact, full-context, no-eviction: the accuracy ceiling, by construction
unbeatable, paid for with a cache and per-step bandwidth that grow with the sequence.

**Why this is the floor.** It refuses the task's central constraint. The leaderboard penalizes
`mean_retained_fraction > 0.25`; the full cache retains 1.0, four times over tolerance, so the budget
penalty fires hard and the reduction term sits at its floor. Highest raw accuracy, worst *budgeted*
score — a visible reference anchor, not a valid budgeted submission.

**Step-1 edit.** The plan declares `disable_compression: True`, the one non-budget plan field the
harness actually enforces: it skips `score_tokens`/`select_cache` and records retained = 1.0.
`score_tokens` returns `None` (also read as keep-everything) and `select_cache` returns keys/values
unchanged. `rerotate_selected_keys = False` because nothing is re-positioned, so the prefill RoPE
phases stay correct and decode continues from the true sequence length.

**Hyperparameters.** None — there is no scoring rule, no sink count, no budget knob the policy reads.
The only operative declaration is `disable_compression = True`; the harness owns the rest.

**What to watch.** The per-workload accuracy is the target each later rung is measured against; gsm8k
under the full cache is the reasoning-headroom number (a compressor that shreds the chain-of-thought
prefix craters there first); the runtime row is the slowest decode and calibrates the runtime term.

```python
# EDITABLE region of custom_selection_eval.py (lines 40-101) — step 1: full-attention anchor
class SelectionPolicy:
    """Naive full-attention anchor: keep every prefill KV token."""

    method_name = "full_attention"
    rerotate_selected_keys = False

    def retention_plan(self, layer_id, request_meta, cache_meta):
        return {
            "method": self.method_name,
            "disable_compression": True,
        }

    def score_tokens(self, module, hidden_states, keys, values, kwargs, plan):
        return None

    def select_cache(self, module, keys, values, scores, n_kept):
        return keys, values
```
