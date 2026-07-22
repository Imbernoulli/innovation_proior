Let me start from what actually hurts. I am serving a long-context model and the KV cache is eating me alive on two fronts at once. Memory: every layer, every key-value head, stores the Key and Value vector of every token I have ever seen, and on a thirty-thousand-token prompt that cache is as big as the model weights. Compute: every single decode step attends over the whole cached sequence, so latency climbs linearly in the cache length, and prefill attention is quadratic in the prompt. I want to keep, after prefill, only a small fixed-budget slice of the cached tokens — say the most important 20% — and decode from that without the task quality falling apart. So the whole game is: out of the cached K and V tensors, manufacture a per-token importance score, rank the tokens, keep the top fraction, evict the rest.

What does everyone do today? The strongest line is to score a token by how much attention it has received. H2O is the cleanest version: accumulate, for each cached token, the attention weight it gets summed across the sequence, observe that these accumulated scores follow a power law so a few "heavy hitter" tokens dominate, keep those plus the recent ones, and there is even a near-optimality argument under submodularity. SnapKV is the same family with a twist: put a small "observation window" of tokens at the end of the prompt, look at what those query positions attend to in the prefix, cluster and pool the high-attention prefix positions, keep them. Both work well. But both have the same two diseases for my deployment. First, they need the attention matrix — the full `A_i = softmax(Q_i K_i^T / sqrt(d_h))`. My production kernel is FlashAttention, which is fast precisely because it *never* materializes that `n×n` matrix; it streams the softmax-weighted output and throws the weights away. So a score that is a function of `A_i` simply cannot be computed in the stack I actually run. That is not an inconvenience, it is a wall. Second, both are instruction-dependent. H2O accumulates attention *during decoding*, so the kept set depends on the queries I have already issued; SnapKV scores the prefix *against the question window at the end of the prompt*, so the kept set changes with the question. But I want to compress a context once and reuse it across turns and across different instructions — multi-turn chat, a document I will ask many things about. If the kept tokens depend on the question, I have to recompress for every query, and the compression is wrong the moment the question moves from the end of the prompt to somewhere else. So attention-weight scoring is doubly disqualified: not kernel-compatible, and not query-independent.

The other big direction is quantization — KIVI, CacheGen — store the same tokens at 2 bits instead of 16. That genuinely shrinks memory, a lot. But it keeps *every* historical token, so the number of attention operations is unchanged: the quadratic prefill and the linear-in-cache decode are exactly as expensive as before. It fixes memory and does nothing for compute. I need to drop tokens, not just shrink them.

And then there is the cheap, robust baseline: StreamingLLM. Keep the first few "sink" tokens and a recent sliding window, evict everything in between. This is beautiful for my constraints — it is query-free, it needs no attention weights, it runs fine under FlashAttention, and it generalizes to millions of tokens. The sink part is not a hack either: softmax forces a query's attention weights to sum to one, so when the query has no strong match it has to dump the residual mass somewhere, and it dumps it on the first tokens, which are visible to every later position and get trained into attention sinks that soak up weight regardless of meaning. Keeping ~4 of those keeps the attention distribution sane. So I will certainly keep sinks. But the eviction itself is *indiscriminate*: anything between the sink and the recent window is gone, no matter how much it carries. Put a needle in the middle of a long haystack and StreamingLLM drops it, and retrieval collapses. So this gives me the kernel-friendly, query-free skeleton — sink plus recent window — but not the part I actually need: a way to decide, among the tokens in the middle, which ones to keep.

So the real question sharpens to this. I am forbidden from using attention weights and from using the query. I am allowed to look only at the cached K and V tensors themselves. Out of *just those tensors*, can I tell which tokens are important enough to keep? The L2-norm method says maybe: it found, empirically, that a low L2 norm of a key embedding correlates with a high attention score later, so keep the low-key-norm tokens. That is the right spirit — attention-free, query-free, uses only the cached tensors. But it is thin. It is a single scalar per token, computed from the key alone, ignoring the value entirely; it was derived by correlating against attention loss rather than from any property of the sequence; and it is a *global, static* per-token statistic with no sense of where the token sits or how it relates to its neighbors. A context-free per-key scalar can rank isolated magnitudes, but it cannot notice whether a token is a local discontinuity in the flow of the cache. So "use only the tensors" is the right constraint, but the score has to extract more structure from K and V.

Let me go back to what I actually know about how K and V are distributed, because that is the only raw material I have. The quantization people mapped this out carefully, even if they used it for a different purpose. CacheGen's first insight: within the same layer and channel, tokens that are close together in position have much more similar K/V values than tokens far apart — the delta between consecutive tokens' K (or V) values is tightly piled up near zero. That is not a coincidence, it falls straight out of the model being autoregressive: the next token's representation does not jump abruptly away from the previous one's. Token-wise locality. And KIVI's insight: the key cache has a few *fixed* channels with very large magnitude — persistent per-channel outliers — which is why you quantize keys per channel; the value cache has no such channel outlier pattern and you quantize values per token. So keys carry structure along the channel axis (some channels are just huge, always), and values carry structure along the token axis.

Sit with the channel-outlier fact for a second, because it is a trap I have to avoid. If a handful of key channels are always enormous, then any naive magnitude- or variance-based score over the raw key vector will be dominated by those same fixed channels for *every* token — the score would mostly measure "which token happened to have a slightly bigger value in the one giant channel," which is not importance, it is just the persistent channel norm leaking through. The L2-norm method partly works *despite* this; I want something that works because I have removed it. So before I can read importance off the spread of a token's vector, I have to strip out the channel-specific scale and offset — normalize each channel so the giant channels and the tiny channels are on the same footing. The question is: normalize against *what* statistics? I have no labels, no global mean I trust, and I refuse to use the query.

Here token-wise locality pays off. If adjacent tokens have nearly identical K/V values, then a short contiguous chunk of tokens is a tight little cloud, and its per-channel min and max are a faithful local description of the scale of each channel right there. So I can normalize a token's key (or value) by min-max over a nearby window: for each channel, subtract the window min and divide by the window range. That maps each channel into roughly [0,1] *using statistics from the token's own neighborhood*, which is exactly the regime where locality says those statistics are valid. The persistent giant channels get divided by their own large local range and collapse onto the same scale as everyone else — the channel-specific norms are gone — and what is left in the normalized vector is each channel's *relative* position within the local cloud.

But which window's min and max? The obvious choice is the token's own chunk — normalize the chunk by its own min/max. Let me think about what that measures, though. If I min-max a chunk by its own statistics and then ask which token in it is unusual, I am asking which token is an outlier *relative to itself and its chunk-mates* — a purely self-referential question. Worse, the min and the max of the chunk are *achieved by tokens in the chunk*, so those extreme tokens get pinned to exactly 0 or 1 in some channel by construction, which biases the score toward whichever tokens happened to define the envelope. It is circular. I want a reference that is *external* to the tokens I am scoring, so that "unusual" means something — unusual compared to a yardstick the scored tokens did not get to set.

And locality hands me the external yardstick for free: the *next* chunk. Because the model is autoregressive and adjacent tokens barely move, the chunk immediately after the current one is, by locality, a faithful sample of the same local distribution — the current chunk's tokens are "supposed to" look like the next chunk's tokens. So normalize the current chunk's K and V using the min and max taken from the *next* chunk. Now the question I am asking each token is sharp and query-free: *how coherent is this token with what comes right after it?* A token whose normalized vector sits comfortably inside the next chunk's envelope is predictable — it is the kind of token the local flow was going to produce anyway, so dropping it loses little. A token that, even after normalizing by the next chunk's scale, is still strange — spread out, off-distribution — is one the next chunk does *not* explain, a discontinuity, a piece of information that does not follow from local flow. That is the token worth keeping. So the scoring is relative to the chunk one step ahead — chunk `p` measured against the statistics of chunk `p+1`, a fixed lag of one window — which is a clean, query-free way to phrase "is this token predictable from what immediately follows it."

Let me make this concrete and pin the axes, because the whole thing lives or dies on getting min/max/std over the right dimension. Partition the cache (after the sinks) into contiguous chunks of size `L`. For a partition `p`, head `i`, and `Z` standing for either `K` or `V` (each a chunk of shape `L × d_h`: `L` tokens, `d_h` channels), take the *next* partition `Z_i^{p+1}` and compute, for each channel, its min and max over the sequence (token) axis:

```
min_i^{p,Z} = min_seq( Z_i^{p+1} ),   max_i^{p,Z} = max_seq( Z_i^{p+1} ).
```

The min over `seq` is what makes this token-wise — I am collapsing the `L` tokens of the reference chunk down to one min and one max *per channel*. Then normalize the current chunk channel-wise with those reference statistics:

```
Zbar_i^p = ( Z_i^p - min_i^{p,Z} ) / ( max_i^{p,Z} - min_i^{p,Z} ).
```

Now `Zbar` is the current chunk re-expressed in the next chunk's coordinate frame: each channel scaled by the local range of *that* channel in the reference, so the persistent giant key channels are flattened and every channel is comparable. The channel-specific norms are eliminated and, crucially, the remaining channel-wise spread within a token is no longer just a proxy for which raw channels are huge. So now I can read importance off that spread.

What is the spread, and over which axis? For a single token, its normalized vector has `d_h` channel entries; the standard deviation *over the channels* of that vector is one number per token. A coherent token — one that looks like the local cloud — has all its channels sitting in a similar normalized position, so a *small* channel-wise std. An incoherent, surprising token has channels scattered across the normalized range, a *large* channel-wise std. So the per-token channel-wise std is exactly the importance signal I wanted, and it is well-defined only because I first removed the channel norms (otherwise the std would just track the giant channels). Compute it per token:

```
std_i^p = Std_channel( Zbar_i^p )      # one scalar per token in the partition
```

I should not take it on faith that this std actually points at the surprising tokens; let me build a tiny case and look. Make a reference chunk that is a tight local cloud — `L=4` tokens, `d=16` channels, every channel near 2.0 with spread 0.05, which is what locality says a chunk looks like. Make a current chunk of three tokens drawn from the same tight cloud plus one token that I deliberately knock off-distribution by adding `(+1,-1,+1,-1,...)` across its channels, i.e. a discontinuity. Normalize the current chunk by the reference's per-channel min/max and read the channel-wise std per token. The three coherent tokens come out at std ≈ 0.88, 0.76, 0.72; the off-distribution token comes out at std ≈ 14.0. So the std does separate them, and by a wide margin — the surprising token is the argmax, exactly the one I want to keep. (The large gap is itself informative: a token that does not fit the next chunk's narrow per-channel range gets divided by a small range and blows up, which is the behavior I want.)

I have one std vector of length `L` for the keys and one for the values. I should use both, because KIVI told me K and V carry *complementary* structure — K is channel-organized, V is token-organized — so a token can be a discontinuity in the key flow, the value flow, or both, and I want to catch all three. So I will produce a key score and a value score and add them.

Before adding, though, I want to turn each raw std vector into something with a consistent scale across partitions and that *separates* the outliers rather than just ranking them linearly. A softmax over the `L` tokens of the partition is the natural candidate: it normalizes the std vector into a per-partition distribution summing to one, and because exp is convex it should sharpen the tail. In the toy case above the four stds `(0.88, 0.76, 14.0, 0.72)` softmax to `(0.0, 0.0, 1.0, 0.0)` — the one genuinely high-std token swallows essentially all the mass and the three coherent tokens are pushed to zero, which is the behavior I want when I am about to keep only a small fraction. So:

```
score(Z_i^p) = Softmax_tokens( Std_channel( Zbar_i^p ) ),
score_i^p    = score(K_i^p) + score(V_i^p).
```

Sum the key-softmax and the value-softmax. Look at what the score depends on: `min`, `max`, `std`, `softmax` over the cached K and V only — no `A_i`, no query state anywhere in the expression. So it is query-free and attention-free by construction, computed entirely from comparisons among the KV tensors, which means it does not need the kernel to materialize the attention matrix and runs under FlashAttention. Dividing the sum by 2 in code keeps the range bounded; since both summands are non-negative and I am only going to rank, scaling by a positive constant leaves the order untouched.

Now the bookkeeping around the chunks, which I have to get exactly right or I will reference off the end of the cache. The sinks — the first `S` tokens — are never scored and never evicted; they are the StreamingLLM anchors. After the sinks I have `q_len - S` tokens to partition into chunks of `L`. The very last chunk has *no chunk after it* to use as a reference, so it cannot be scored by this rule — and that is not a defect, it is exactly the recent sliding window I wanted anyway: the most recent tokens matter most and should never be compressed. So the last full partition is always kept, joined by whatever remainder is left when `q_len - S` is not a multiple of `L` (that remainder also has no clean reference and is recent). So the never-compressed tail is "one full lag window plus the modulo." If there is not even enough sequence to have one chunk to score and one chunk to reference it — `q_len < S + 2L` — I skip compression entirely.

Let me write the index arithmetic the way I will implement it. The scored region runs from `S` up to the start of that always-kept tail:

```
end_idx  = S + ((q_len - S) // L) * L          # end of the last *reference-able* chunk boundary
tail_len = L + (q_len - end_idx)               # the last full window + the remainder
```

and I reshape the scored region `[S : end_idx]` into `(num_partitions, L, d_h)` so I can vectorize. Reshaped to `(b, h, num_partitions, L, d_h)`, the reference chunks are partitions `1 .. num_partitions-1` (`target[..., 1:, :, :]`) and the chunks being scored are `0 .. num_partitions-2` (`target[..., :-1, :, :]`) — partition `p` scored by partition `p+1`, the lag, done in one shot. min and max over the `L` axis (dim `-2`) of the reference, broadcast back over `L`, normalize, std over the channel axis (dim `-1`), softmax over the `L` tokens (dim `-1`). Average — equivalently, since I will rank, sum — the key and value scores.

Now I hit the question that actually decides the implementation, and I almost glossed it. What I want is top-K *per partition* and per head — within each lag window keep `rL` of the `L` tokens, so every region of the context contributes its share and I do not accidentally keep a whole early partition and evict a whole late one. But the compression harness I am plugging into does the dead-simple thing: it computes one score vector over the *entire* cached sequence and takes a single global `topk(n_kept)`. The softmax-std score I just built is not on a common scale across partitions — different layers and positions have different magnitudes, and the softmax only normalizes *within* a window, not across them. So I have to actually worry that a global top-K over my raw scores will let one partition's larger numbers eat another partition's budget.

Let me test that worry rather than guess. Three partitions of `L=6`, raw scores that are deliberately on different scales: partition 0 around `0.1–0.9`, partition 1 the same pattern shifted up by 5, partition 2 the same pattern blown up by 100 (values `20–90`). If I flatten these and take a global top-6, the kept count per partition comes out `[0, 0, 6]` — partition 2's larger numbers sort above everything and swallow the entire budget, leaving the other two windows with nothing kept. That is not a mild imbalance; it is total starvation of the smaller-scale windows. So a naive global top-K over the raw scores genuinely breaks the per-partition quota. That is a real wall, not a stylistic preference.

I could special-case the harness to loop over partitions, but that fights the framework. The other option is to make the scores themselves carry the per-partition structure so that one global top-K *becomes* a per-partition top-K — strip the cross-partition scale out of the numbers while keeping the within-partition order. Replace each partition's raw scores with their *within-partition rank* on a common scale: argsort within the `L` axis, argsort again — the double-argsort trick that returns, for each token, its rank position among its `L` chunk-mates, an integer in `{0, ..., L-1}` — and divide by `L`:

```
score = score.argsort(dim=-1).argsort(dim=-1) / L      # within-partition rank, in [0, 1)
```

Now I have to check whether this actually fixes the budget-stealing, because it is easy to convince myself and be wrong. Re-run the same three lopsided partitions through the double-argsort. The output rows become identical value sets — every partition is now exactly `{0, 1/6, 2/6, 3/6, 4/6, 5/6}`, just permuted so the highest-raw token in each partition gets `5/6`, the next `4/6`, and so on. The 100× scale of partition 2 is gone; only the *order* within each partition survived. Flatten and take the global top-6 again: this time the kept count per partition is `[2, 2, 2]`, and the two kept positions in every partition are precisely the two highest-raw tokens of that partition. So the global top-K now lands `rL = 2` in each window, which is exactly the per-partition top-K I wanted.

I want to know this is not luck on one example, so I sweep it: 2000 random configurations, `L=8`, five partitions, retained-per-window `keep_per ∈ {1,2,3,5,7}`, each time flattening the rank scores and taking a global top-`keep_per·5`. In every single case the per-partition kept count is exactly `keep_per`. The reason is now clear from the values: because every partition carries the identical multiset `{0, 1/L, ..., (L-1)/L}`, the global top-K threshold falls between the same two rank levels in every partition, so each contributes the same count. The within-partition rank values also sit in `[0,1)`, strictly below the `1.0` I will assign the sinks and the tail, so those are always kept first and never compete with scored tokens. The alternative — skip the rank step and let the raw softmax-std scores compete globally, "cross-partition scoring" — is now visibly the *opposite* policy: it is exactly the budget-stealing I measured above, which is occasionally what I want (let windows with larger raw outliers draw more budget), so I keep it as an option but leave it off by default.

So the full assembled score vector over the cache is: `S` ones for the sinks, then the flattened per-partition ranks for the scored region, then `tail_len` ones for the always-kept sliding window. There is one more case to handle gracefully: when `q_len < S + 2L` and I skip compression, but the harness might still force a top-K. I do not want it to keep arbitrary tokens — I want it to fall back to the sensible default, keep the most recent ones. So in that branch I hand back a ramp: ones for the sinks, and for the post-sink tokens an increasing `arange / (q_len - S)`, so later tokens score higher and any forced top-K keeps the recent tail. That is StreamingLLM behavior as the graceful degenerate case, which is the right thing to degrade to.

Let me also nail down the compression-ratio bookkeeping, because `r` (per-window retention) and the overall `C` are not the same number — the sinks and the whole sliding-window tail are kept at full, so the *effective* compression is gentler than `r` suggests, and a controller has to convert between them. For a sequence `L_s ≥ S + 2L`, the retained length is the sinks, plus `rL` from each of the *compressed* windows, plus the always-kept tail (one full window `L` plus the remainder):

```
L_R = S + rL ( floor((L_s - S)/L) - 1 ) + L + Mod(L_s - S, L),
C   = 1 - L_R / L_s.
```

The `floor((L_s - S)/L) - 1` is "number of windows after the sink, minus the last one that is never compressed." For `L_s < S + 2L`, `C = 0`. Let me sanity-check the formula against the layout it is supposed to summarize, because off-by-one errors in the sink/tail accounting are easy. Take `L_s = 1000`, `S = 4`, `L = 128`, `r = 0.25`: there are `floor(996/128) = 7` windows after the sink, the last one is never compressed, so 6 windows keep `rL = 32` each. Adding it up by hand — `4` sinks `+ 6·32` compressed `+ 128` last full window `+ (996 mod 128 = 100)` remainder `= 4 + 192 + 128 + 100 = 424`. The formula gives `4 + 0.25·128·6 + 128 + 100 = 424` too, and `C = 1 - 424/1000 = 0.576`. They agree, and I get the same agreement on a few other lengths I tried. One thing this surfaces: `rL` need not be an integer in general, so the code will round it, and the realized `C` can drift by a token or two from the formula — fine, but worth knowing the budget is approximate, not exact. So given a target overall `C` (the harness asks for a retained fraction, not a per-window retained count), I invert this for the per-window `r`. The practical constraint is simple: the kept tokens per window, `rL`, must be large enough for the contiguous span of information I may need to preserve. That pushes `L` away from both extremes. If `L` is too large, token-wise locality weakens because the next chunk is no longer a close reference; if `L` is too small, the min/max estimate is noisy and `rL` may be too few tokens to keep a multi-token fact intact. A small sink `S`, such as 4, is enough for the attention-sink anchor because the sink mass concentrates on the first few tokens.

Tracing back over the choices, each one now answers to a specific pressure I hit rather than to taste. The next-chunk reference is forced by the fact that an own-chunk min/max lets the scored tokens define the envelope used to judge themselves — the extreme tokens get pinned to 0 or 1 by construction — whereas locality makes the next chunk a yardstick the scored tokens did not set. Min-max runs along the token axis per channel, not across channels, because that is what cancels the persistent per-channel key outliers; without it the std I just verified as a signal would instead track which raw channel is biggest. The std runs over channels because I want one scalar per token after that normalization. The softmax runs over the window's tokens because the keep decision is local to the window, and I saw it concentrate mass on the one off-distribution token. Summing K and V uses both axes of structure that the cache exposes. And the rank-normalization is the fix for the budget-stealing I actually measured: it is the thing that makes one global top-K land `rL` per window, with every scored token below the `1.0` reserved for sinks and tail.

Let me write the scoring rule as the code I would actually ship, filling the one empty slot — the per-token `score` — in the harness's score-then-keep loop. The harness already does the global top-K and the gather; I only owe it a score tensor of shape `(batch, num_kv_heads, q_len)`, higher meaning keep.

```python
import torch
from torch import nn
from dataclasses import dataclass


@dataclass
class LagKVPress(ScorerPress):
    """Score each cached token by how INCOHERENT it is with the next lag window,
    using only the cached keys and values (no attention weights, no query)."""

    compression_ratio: float = 0.0
    n_sink: int = 4          # StreamingLLM attention sinks, always kept
    lag_size: int = 128      # L: partition / lag-window size (short enough that locality holds)
    cross_scoring: bool = False   # False -> per-partition top-K via rank-normalization

    def _get_states_score(self, target):
        # target: (b, h, num_partitions, L, d_h). Score chunk p by the stats of chunk p+1.
        ref = target[:, :, 1:, :, :]    # reference = the NEXT chunk (the "lag")
        v   = target[:, :, :-1, :, :]   # scored    = the current chunk
        # token-wise (over the L axis) min/max of the reference, per channel
        min_r = ref.min(dim=-2).values.unsqueeze(-2).expand(-1, -1, -1, self.lag_size, -1)
        max_r = ref.max(dim=-2).values.unsqueeze(-2).expand(-1, -1, -1, self.lag_size, -1)
        # normalize current chunk into the next chunk's per-channel frame, then read off
        # the channel-wise spread per token; softmax over the L tokens separates outliers
        return ((v - min_r) / (max_r - min_r)).std(dim=-1).softmax(dim=-1)

    def score(self, module, hidden_states, keys, values, attentions, kwargs):
        bsz, n_kv_heads, q_len, d = keys.shape

        # too short to have one chunk to score AND one chunk to reference: skip compression,
        # but bias toward the recent tail so any forced top-K degrades to StreamingLLM.
        if q_len < self.n_sink + 2 * self.lag_size:
            score = torch.ones((bsz, n_kv_heads, q_len), dtype=keys.dtype, device=keys.device)
            if q_len > self.n_sink:
                score[:, :, self.n_sink:] = (
                    torch.arange(q_len - self.n_sink, device=keys.device) / (q_len - self.n_sink)
                ).to(keys.dtype)
            return score

        # sinks fixed; last full window + remainder form the always-kept sliding tail
        end_idx  = self.n_sink + ((q_len - self.n_sink) // self.lag_size) * self.lag_size
        tail_len = self.lag_size + q_len - end_idx

        # reshape the scored region into partitions and score K and V against their next chunk
        key_score   = self._get_states_score(
            keys[:, :, self.n_sink:end_idx].view(bsz, n_kv_heads, -1, self.lag_size, d))
        value_score = self._get_states_score(
            values[:, :, self.n_sink:end_idx].view(bsz, n_kv_heads, -1, self.lag_size, d))
        score = (key_score + value_score) / 2     # same ordering as the summed K+V score

        if not self.cross_scoring:
            # within-partition rank in [0,1): under the aligned budget,
            # one global top-K becomes a per-partition top-K
            score = score.argsort(dim=-1).argsort(dim=-1) / self.lag_size
            score = score.to(keys.dtype)

        # sinks and the sliding tail score 1.0 -> always kept, above any scored token
        sink_score = torch.ones((bsz, n_kv_heads, self.n_sink), dtype=score.dtype, device=score.device)
        tail_score = torch.ones((bsz, n_kv_heads, tail_len), dtype=score.dtype, device=score.device)
        return torch.cat((sink_score, score.reshape(bsz, n_kv_heads, -1), tail_score), dim=-1)
```
