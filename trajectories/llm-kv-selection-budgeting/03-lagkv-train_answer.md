StreamingLLM cleared the budget — retained $0.20$ across the board, runtimes alongside the anchor — but the accuracy gaps confirmed exactly the diagnosis I was afraid of. Hotpotqa fell from the anchor's 37.1 to 25.6 and passage retrieval from 62.4 to 53.1, the needle-in-a-haystack workloads where a purely positional rule drops the needle whenever it lives in the middle; LongBench v2 roughly held at 29.6; and the loud one, gsm8k, cratered from 31.8 to 1.7 — effectively zero. On a reasoning trace the load-bearing intermediate steps are scattered through the *middle* of the prompt, not at the end, so keeping four sinks plus the last 20% throws away the very content the answer depends on. The verdict is unambiguous: I need a *content-aware* score that decides *which* middle tokens to keep, not just where they sit.

I have to be careful, because the obvious content-aware methods are precisely the ones this harness forbids. The strongest line scores a token by the attention it has received — H2O accumulates per-token attention weight and keeps the heavy hitters; SnapKV looks at what an end-of-prompt observation window attends to. Both are query-dependent, and worse, both need the realized attention matrix $\text{softmax}(QK^\top/\sqrt{d})$, which this hook never hands me: `score_tokens` receives the module, hidden states, keys, and values, and the model runs under SDPA, which streams the softmax-weighted output and never materializes the $t \times t$ weights. That is a wall, not an inconvenience. Quantization (KIVI, CacheGen) is out for a different reason — it keeps *every* token at lower precision, so it never reduces the number of attended tokens and cannot hit a 20% budget at all. So the constraint sharpens: score the middle tokens for importance using *only the cached K and V tensors themselves*, with no attention weights and no query. The thin precedent, the key-norm heuristic (low key L2 norm correlates with high later attention), is the right spirit but too thin — one scalar from the key alone, ignoring the value and any sense of where a token sits relative to its neighbors. A context-free magnitude cannot notice whether a token is a local *discontinuity* in the flow of the cache, which is exactly the kind of informative middle token StreamingLLM blindly dropped.

I propose *LagKV*: a lag-relative score that reads importance off the structure of K and V among themselves. Two facts about how K and V are distributed make it possible. First, token-wise locality: because the model is autoregressive, adjacent tokens' K (and V) vectors barely differ — the next token's representation does not jump abruptly from the previous one's, so a short contiguous chunk of tokens is a tight cloud. Second, a K/V asymmetry: the key cache has a few *fixed* channels of persistently huge magnitude (the per-channel outliers that force per-channel key quantization), while the value cache has no such pattern and varies per token. That outlier fact is a trap: any naive magnitude-or-variance score over the raw key vector would be dominated by those same fixed giant channels for every token, so the score would mostly measure persistent channel norm leaking through, not importance. Before I can read importance off the spread of a token's vector, I must strip the channel-specific scale — normalize each channel so the giant and tiny channels sit on the same footing.

Locality supplies both the normalization and the right yardstick. Because adjacent tokens are nearly identical, a contiguous chunk's per-channel min and max faithfully describe each channel's local scale, so I can map each channel into roughly $[0,1]$ by subtracting the local min and dividing by the local range — and the persistent giant channels, divided by their own large local range, collapse onto everyone else's scale, leaving only each channel's *relative* position within the local cloud. But *which* window's statistics? Normalizing a chunk by its own min/max is circular: it asks which token is an outlier relative to its own chunk-mates, and worse, the extreme tokens that *achieve* the min and max get pinned to exactly 0 or 1 by construction. I want a reference external to the tokens I am scoring, and locality hands it over for free — the *next* chunk. Because the model is autoregressive, the chunk immediately after the current one is a faithful sample of the same local distribution; the current chunk's tokens are "supposed to" look like the next chunk's. So I normalize the current chunk by the *next* chunk's min and max, and the question I ask each token becomes sharp and query-free: how coherent is this token with what comes right after it? A token whose normalized vector sits comfortably inside the next chunk's envelope is predictable — local flow was going to produce it anyway, so dropping it loses little. A token still strange after normalizing by the next chunk's scale is a discontinuity carrying information local flow does not explain. That is the token to keep — exactly the middle token StreamingLLM could not distinguish.

Pinning the axes is where the method lives or dies. Partition the cache after the sinks into contiguous chunks of size $L$ (the lag). For chunk $p$, head $i$, and $Z$ standing for either K or V (a chunk of $L$ tokens by $d_h$ channels), take the next chunk $Z^{p+1}$ and compute, *per channel, over the token axis*, its min and max — collapsing the $L$ reference tokens to one min and one max per channel, which is the token-axis collapse that makes the normalization channel-wise. Then normalize the current chunk channel-by-channel: $(Z^p - \min)/(\max - \min)$. Now read the spread *over the channels* of each normalized token — its channel-wise standard deviation, one scalar per token. A coherent token has all channels in similar normalized positions, hence small channel-wise std; an incoherent, surprising token has channels scattered across the range, hence large std. That std is a well-defined importance signal *only because* I first removed the channel norms; otherwise it would just track the giant channels. I compute it for both K and V and add them, because the two caches carry complementary structure (K channel-organized, V token-organized), so a token can be a discontinuity in the key flow, the value flow, or both. Before adding, I softmax each std vector over the $L$ tokens of its chunk: this normalizes to a per-chunk distribution and, because exp is convex, sharpens the tail so genuinely high-std tokens pull away from the merely above-average — the right behavior when keeping a small fraction.

The bookkeeping has to be exact or I reference off the end of the cache. The first $n_{\text{sink}}$ tokens are never scored and never evicted — the StreamingLLM attention-sink anchor, still load-bearing for the denominator. After the sinks, the very last chunk has no chunk after it to reference, so it cannot be scored — and that is not a defect, it is precisely the recent sliding window I want anyway, since the most recent tokens matter most. So the always-kept tail is one full lag window plus the remainder when $(q_{\text{len}} - n_{\text{sink}})$ is not a multiple of $L$: the scored region runs from $n_{\text{sink}}$ up to $\text{end\_idx} = n_{\text{sink}} + \lfloor (q_{\text{len}} - n_{\text{sink}})/L \rfloor \cdot L$, with $\text{tail\_len} = L + (q_{\text{len}} - \text{end\_idx})$. If there is not even enough sequence for one chunk to score and one to reference ($q_{\text{len}} < n_{\text{sink}} + 2L$), I skip the lag rule, but rather than let a forced top-K keep arbitrary tokens I return a recent-biased ramp ($\text{arange}/(q_{\text{len}}-n_{\text{sink}})$ after the sinks) so any forced top-K degrades gracefully to StreamingLLM. I reshape the scored region into $(\text{num\_partitions}, L, d_h)$ and vectorize: the reference chunks are partitions $1{:}$ and the scored chunks are $:{-1}$, so chunk $p$ is scored by chunk $p{+}1$ in one shot, min/max over the token axis, std and softmax over the channel/token axes respectively.

Then the design choice that actually decides the implementation. I want top-K *per partition* — within each lag window keep $rL$ of $L$ — so every region of the context contributes its share and I do not accidentally keep a whole early window and evict a whole late one. But this harness does the dead-simple thing: one score vector over the entire cache and a single global $\text{topk}(n_{\text{kept}})$. Hand it the raw softmax-std scores and a window with systematically larger raw std could steal the whole budget. The fix is to make the scores carry the per-partition structure so one global top-K *is* a per-partition top-K: replace each partition's scores with their within-partition *rank*. Argsort the scores along the $L$ axis, argsort again — the double-argsort returns each token's rank among its chunk-mates, an integer in $\{0,\dots,L-1\}$ — and divide by $L$. Now every partition's scores are the same set $\{0, 1/L, \dots, (L-1)/L\}$ just permuted by importance, identically distributed across partitions, so the global top-K takes the same count of high-rank tokens from each window. (Skipping the rank step — `cross_scoring` — lets budget flow toward windows with larger raw outliers, a deliberate option, but it loses the uniform-per-partition guarantee, so it is off by default.) This also puts every scored token strictly in $[0,1)$, below the sinks and tail which I set to exactly $1.0$, so those are always kept above any scored token. The assembled vector is $n_{\text{sink}}$ ones, the flattened per-partition ranks, then $\text{tail\_len}$ ones, and a single global top-K keeps the sinks, the recent tail, and the top-ranked content tokens of every middle window — attention-free, query-free, computed entirely by comparing the cached K and V among themselves. The compression ratio I read from the plan for provenance; the harness force-overrides it.

I expect retained to stay $\sim 0.20$ and runtime near step 2, since the score is a few elementwise reductions over the cache. The bet is that keeping the *informative* middle tokens recovers what StreamingLLM's blind eviction lost, so passage retrieval should beat 53.1 by a clear margin and hotpotqa should recover above 25.6, with LongBench v2 near 29. The honest worry is gsm8k: my score is computed from the *prompt's* KV statistics, but gsm8k's important tokens live in the model's own generated reasoning, so a coherence-with-the-next-chunk score on the prompt may still not protect the chain-of-thought — if gsm8k stays near 1.7, that says even content-aware lag scoring on prompt KV is insufficient for reasoning and the next rung needs a score tied to what *future queries* will attend to. And the sharpest falsifiable line: LagKV must beat StreamingLLM on the geometric mean across the five workloads. Repobench is where I might lose a little — next-line code completion is well-served by StreamingLLM's recent-window keep, and spending budget on "incoherent" middle tokens may not help there — so the question is whether recovering retrieval outweighs that slip.

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
