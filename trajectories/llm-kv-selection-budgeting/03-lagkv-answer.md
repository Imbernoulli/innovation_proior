**Problem.** StreamingLLM cleared the budget but its *positional* keep dropped middle-of-context needles
(hotpotqa 25.6, passage retrieval 53.1, both below the anchor) and shredded the reasoning prefix
(gsm8k 1.7 vs 31.8). I need a content-aware score that decides *which* middle tokens to keep — using only
cached K and V (no attention matrix, no query), since the hook forbids both and H2O/SnapKV need the
attention weights this harness never materializes.

**Key idea.** The model is autoregressive, so adjacent KV vectors barely move (token-wise locality);
hence the *next* contiguous chunk is a faithful external reference for the current chunk's local
distribution. Normalize the current chunk's K and V channel-by-channel by the *next* chunk's per-channel
min/max (over the token axis) — this strips the persistent per-channel key outliers (KIVI) and re-expresses
each token in the next chunk's frame. The surviving channel-wise std of a normalized token measures how
*incoherent* it is with what comes next: high std = a discontinuity carrying information local flow does
not predict = keep it. Softmax over the window separates outliers; sum the K and V scores (the two caches
organize information differently). This is "lag-relative" scoring.

**Why it works.** It keeps the StreamingLLM skeleton (sinks always kept; the final reference-less chunk
plus remainder is the always-kept recent window) but replaces the blind middle-eviction with a per-token
incoherence score, so informative middle tokens survive. Attention-free and query-free, so it runs under
SDPA/FlashAttention and is reusable across instructions.

**Harness fit.** The harness does one global `topk(n_kept)`. A naive global top-K would not respect
per-window quotas, so each partition's scores are replaced by their within-partition rank
(`argsort(argsort) / lag_size`, in [0,1)); under the aligned budget one global top-K then behaves as a
per-partition top-K, and every scored token sits strictly below the sinks/tail (set to 1.0). When
`q_len < n_sink + 2*lag_size` the lag rule is skipped and a recent-biased ramp is returned so any forced
top-K degrades to StreamingLLM. `compression_ratio` is read from the plan but force-overridden by the
harness.

**Hyperparameters.** `n_sink = 4` (attention-sink anchor); `lag_size = 128` (short enough that locality
holds, long enough that `rL` keeps multi-token facts intact); `cross_scoring = False` (use
within-window ranks for the per-partition guarantee).

**What to watch.** Retained ~0.20, runtime near step 2. Expect passage retrieval to beat 53.1 and hotpotqa
to recover above 25.6 (middle content kept); LongBench v2 ~29; repobench possibly a touch below
StreamingLLM (recent-window keep suits next-line completion). The honest worry: gsm8k may stay near 1.7,
since this scores *prompt* KV and reasoning tokens live in the model's own generation — if so, the next
rung needs a score tied to *future-query* attention. The falsifiable hinge: LagKV must beat StreamingLLM
on the geometric mean.

```python
# EDITABLE region of custom_selection_eval.py (lines 40-101) — step 3: LagKV
class SelectionPolicy:
    """LagKV: score tokens by lag-relative key/value variation."""

    method_name = "lagkv"
    rerotate_selected_keys = False

    def retention_plan(self, layer_id, request_meta, cache_meta):
        return {
            "method": self.method_name,
            "sink_tokens": 4,
            "lag_size": 128,
            "cross_scoring": False,
            "compression_ratio": cache_meta["compression_ratio"],
        }

    def score_tokens(self, module, hidden_states, keys, values, kwargs, plan):
        bsz, num_key_value_heads, q_len, dim = keys.shape
        n_sink = int(plan.get("sink_tokens", 4))
        lag_size = int(plan.get("lag_size", 128))
        if q_len < n_sink + 2 * lag_size:
            scores = torch.ones((bsz, num_key_value_heads, q_len), dtype=keys.dtype, device=keys.device)
            if q_len > n_sink:
                scores[:, :, n_sink:] = (
                    torch.arange(q_len - n_sink, device=keys.device) / (q_len - n_sink)
                ).to(keys.dtype)
            return scores
        end_idx = n_sink + ((q_len - n_sink) // lag_size) * lag_size
        tail_len = lag_size + q_len - end_idx

        def state_score(target):
            ref = target[:, :, 1:, :, :]
            value = target[:, :, :-1, :, :]
            min_ref = ref.min(dim=-2).values.unsqueeze(-2).expand_as(value)
            max_ref = ref.max(dim=-2).values.unsqueeze(-2).expand_as(value)
            return ((value - min_ref) / (max_ref - min_ref)).std(dim=-1).softmax(dim=-1)

        key_score = state_score(keys[:, :, n_sink:end_idx].view(bsz, num_key_value_heads, -1, lag_size, dim))
        value_score = state_score(values[:, :, n_sink:end_idx].view(bsz, num_key_value_heads, -1, lag_size, dim))
        scores = (key_score + value_score) / 2
        if not bool(plan.get("cross_scoring", False)):
            scores = scores.argsort(dim=-1).argsort(dim=-1) / lag_size
            scores = scores.to(keys.dtype)
        sink_scores = torch.ones((bsz, num_key_value_heads, n_sink), dtype=scores.dtype, device=scores.device)
        tail_scores = torch.ones((bsz, num_key_value_heads, tail_len), dtype=scores.dtype, device=scores.device)
        return torch.cat((sink_scores, scores.reshape(bsz, num_key_value_heads, -1), tail_scores), dim=-1)

    def select_cache(self, module, keys, values, scores, n_kept):
        indices = scores.topk(n_kept, dim=-1).indices
        gather_idx = indices.unsqueeze(-1).expand(-1, -1, -1, keys.shape[-1])
        selected_keys = keys.gather(2, gather_idx).contiguous()
        gather_idx = indices.unsqueeze(-1).expand(-1, -1, -1, values.shape[-1])
        selected_values = values.gather(2, gather_idx).contiguous()
        return selected_keys, selected_values
```
