StreamingLLM answered the question I posed at step 2, and the answer is the one I was afraid of. It
cleared the budget — retained 0.20 across the board — and its runtimes came in on the shape I predicted.
The accuracy row read as *drops from the anchor* is the diagnosis. Hotpotqa fell 37.1 to 25.6, down
11.5 (31% of the ceiling gone). Passage retrieval 62.4 to 53.1, down 9.3 (15%). Repobench 47.6 to 43.2,
down 4.4 (9%). LongBench v2 ticked *up*, 29.0 to 29.6 — flat, exactly as its chance-floored headroom
demanded. And gsm8k collapsed 31.8 to 1.7, a 95% wipeout. Order those relative drops — gsm8k 95%,
hotpotqa 31%, passage 15%, repobench 9%, v2 ~0% — and it is a clean ranking of how *middle-distributed*
each workload's load-bearing tokens are. gsm8k's chain-of-thought scatters intermediate steps through
the whole prompt, so keeping only the last 20% throws away the content the answer depends on; 1.7 is the
model reasoning from a shredded prefix. Hotpotqa's multi-hop needles sit in the middle. Passage
retrieval's single needle is one span, so partial. Repobench's next-line completion is mostly local, so
the recent window was already well-suited. v2 cannot move. That gradient is the entire case for what
comes next.

The runtime row confirmed the prefill/decode model: against the anchor's 347/323/646/2589/7440,
StreamingLLM posted 361/331/672/2593/7110. The two cheap short-decode workloads and repobench ticked
*up* (14, 8, 26 seconds) because the per-layer top-K, gather, and re-rotation are paid up front on every
layer and not recouped over a handful of decode steps; v2 stayed flat because it is prefill-bound; gsm8k
dropped 330 seconds because it is decode-dominated and the 5x-smaller cache pays off on every step. So
compression has a nonzero cost and re-rotation is a visible part of it — a fact I can spend, because a
content score that keeps tokens at their *original* positions can skip the re-rotation entirely.

The verdict is unambiguous: I need a *content-aware* score that decides which middle tokens to keep, not
just where they sit. But the obvious content-aware methods are exactly the ones this harness forbids.
The strongest line scores a token by how much attention it has received — H2O accumulates attention
weight per token and keeps the heavy hitters, SnapKV keeps prefix positions an end-of-prompt observation
window attends to. Both are disqualified here for two reasons, the second fatal. First, they are
query-dependent — H2O's kept set depends on the queries already issued, SnapKV's on where the question
sits — but I want a compression that does not assume a particular question follows. Second, they need
the realized attention matrix `softmax(QK^T/sqrt(d))`, and this hook does not hand me one: `score_tokens`
gets the module, hidden states, keys, values — no attention tensor — and the model runs under SDPA, which
streams the softmax-weighted output and never materializes the `t x t` weights. So a score that is a
function of the attention matrix simply cannot be computed here. That is a wall. Quantization is out for
the reason I worked through at step 1 — it keeps every token, so it reports retained 1.0 and cannot hit
the token-count budget at all.

So the constraint sharpens: score the middle tokens for importance using *only the cached K and V
tensors themselves*, no attention weights and no query. There is a thin precedent — the key-norm
heuristic, where a low L2 norm of a key correlates with high later attention, so keep low-key-norm
tokens. Right spirit (attention-free, query-free, uses the cache) but too thin: one scalar per token
from the key alone, ignoring the value, with no sense of where the token sits or how it relates to its
neighbors. A context-free per-key magnitude cannot notice whether a token is a local *discontinuity* in
the flow of the cache — and a discontinuity is exactly the kind of informative middle token StreamingLLM
blindly dropped on gsm8k. So the score has to extract structure, not just magnitude.

What do I know about how K and V are distributed? Two facts. First, token-wise locality: within a layer
and channel, tokens close together have much more similar K/V than tokens far apart — the delta between
consecutive tokens' K (or V) is tightly piled near zero, which follows from the model being
autoregressive. Second, a K/V asymmetry: the key cache has a few *fixed* channels with persistently huge
magnitude (per-channel outliers — which is why keys are quantized per channel), while the value cache has
no such pattern and varies per token (values quantized per token). The channel-outlier fact is a trap. If
a handful of key channels are always enormous, any naive magnitude-or-variance score over the raw key
vector is dominated by those same fixed channels for every token. Suppose channel 5 is a persistent
giant, around `[1000, 1002, 1005, 1001]` across a local window while every other channel is order one:
any std over the raw key is dominated by channel 5's ~1000-scale entry, and because channel 5 barely
differs token to token, that std is nearly constant across tokens — it ranks almost nothing. So before I
can read importance off the spread of a token's vector, I must strip out the channel-specific scale.

Token-wise locality gives me the tool. If adjacent tokens have nearly identical K/V, a short contiguous
chunk is a tight cloud, and its per-channel min and max faithfully describe each channel's local scale.
So normalize a token's key (or value) per channel by min-max over a nearby window: subtract the window
min, divide by the window range. Channel 5 with local range `[1000, 1005]` maps a value of `1003` to
`(1003-1000)/5 = 0.6`, on the same `[0,1]` footing as every ordinary channel. What survives is each
channel's *relative* position within the local cloud, the persistent channel norms gone.

But which window's min and max? The token's own chunk is circular — asking which token is an outlier
relative to its chunk-mates, and worse, the min and max are *achieved by* tokens in the chunk, so those
tokens get pinned to 0 or 1 by construction. I want a reference *external* to the tokens I am scoring,
and locality hands it over for free: the *next* chunk. Because adjacent tokens barely move, the chunk
immediately after the current one is a faithful sample of the same local distribution — the current
chunk is "supposed to" look like the next. So normalize the current chunk's K and V by the min and max
of the *next* chunk. Now the question per token is sharp and query-free: how coherent is this token with
what comes right after it? A token sitting comfortably inside the next chunk's envelope is predictable —
local flow would have produced it anyway, dropping it loses little. A token still strange after
normalizing by the next chunk's scale is a discontinuity carrying information local flow does not
explain. Concretely, with a next-chunk envelope of channel 0 in `[2,6]` and channel 1 in `[10,14]`
(range 4 each), a token `[4,12]` normalizes to `[0.5,0.5]` with channel-wise std 0 (dead-center,
coherent, cheap to drop), while `[2.4,13.6]` normalizes to `[0.1,0.9]` with std 0.4 (opposite ends of
the envelope, an unusual shape worth keeping). So the channel-wise std of the next-chunk-normalized token
*is* the importance signal, well-defined only because I removed the channel norms first. I compute it for
both K and V and add them, because the two caches carry complementary structure (K channel-organized, V
token-organized) and a token can be a discontinuity in either flow. Before adding I softmax each std
vector over the `L` tokens of its chunk, which normalizes to a per-chunk distribution and sharpens the
tail so genuinely high-std tokens pull away — the right behavior when keeping a small fraction.

The bookkeeping has to be exact or I reference off the end of the cache. The sinks are never scored and
never evicted — the StreamingLLM anchor I keep, because the sink mechanism is real and it cost 30 gsm8k
points to learn how real. The very last chunk has *no chunk after it* to reference, so it cannot be
scored — and that is not a defect, it is exactly the recent sliding window I want: the most recent tokens
should never be compressed. So the always-kept tail is one full lag window plus the remainder when
`(q_len - n_sink)` is not a multiple of `L`; the scored region runs from `n_sink` to `end_idx = n_sink +
((q_len - n_sink) // L) * L`, and `tail_len = L + (q_len - end_idx)`. If there is not enough sequence for
one scored chunk plus one reference — `q_len < n_sink + 2L` — I skip the lag rule and hand back a ramp
(`arange / (q_len - n_sink)` after the sinks) so any forced top-K degrades to keeping the recent tail,
i.e. StreamingLLM behavior as the graceful degenerate case. I reshape the scored region into
`(num_partitions, L, d_h)` and vectorize: reference chunks are partitions `1..`, scored chunks are
`..num-2`, so chunk `p` is scored by chunk `p+1` in one shot.

Then the question that decides the implementation. I want top-K *per partition* — within each lag window
keep `rL` of `L`, so every region of the context contributes its share and I do not keep a whole early
window while evicting a whole late one. But this harness computes one score vector over the whole cache
and takes a single global `topk(n_kept)`. Hand it raw softmax-std scores and a window with systematically
larger raw std steals the whole budget. The fix is to make the scores carry the per-partition structure
so one global top-K *is* per-partition: replace each partition's scores with their within-partition
*rank* via a double argsort. For a partition `[0.3, 0.9, 0.1, 0.5]`, the first argsort gives the
ascending order at indices `[2,0,3,1]`, and argsort of that gives `[1,3,0,2]` — exactly each token's rank
(token 1, the largest, gets rank 3; token 2, the smallest, rank 0). Divide by `L = 4` and every
partition becomes some permutation of the identical set `{0, 0.25, 0.5, 0.75}`, so a global top-K with
budget for two per window takes `{0.5, 0.75}` from *every* partition — the global operation has become a
per-partition top-K, quotas enforced by construction. (Skipping the rank step — `cross_scoring` — lets
budget flow toward windows with larger raw outliers, a deliberate option, off by default.) The largest
rank is `(L-1)/L < 1`, so every scored token sits strictly in `[0,1)`, below the sinks and tail set to
`1.0`, which are therefore always kept.

That also settles why I do *not* re-rotate here, unlike step 2. StreamingLLM kept two far-apart blocks
and relabeled them to contiguous positions to bound the sink-to-window gap. LagKV keeps tokens
*distributed* across the whole context — sinks, a share of every window, the recent tail — tiling the
original context roughly uniformly, and every kept token keeps its *true* original position, so the
relative distances the model reads are the ones it saw at full attention (valid because this is a
one-shot within-context prefill, not a stream rolling past the trained length). Re-rotating the
distributed keep to contiguous positions would *lie* about the geometry: two tokens 130 apart told they
are one apart, corrupting the relative-position signal. So `rerotate_selected_keys = False`, which also
saves the per-layer re-rotation cost I watched StreamingLLM pay on the short-decode workloads.

The one hyperparameter with real freedom is the lag size, and the budget arithmetic pins it. At retain
0.2, a window of `L` keeps ~`int(0.2 L)` tokens. At `L = 128` that is ~25 per window — enough to preserve
a multi-token fact and give each neighborhood a real share. Push `L` down to 8 and I keep 1 token per
window (a multi-token fact cannot survive), and the 8-token reference chunk is far too few to estimate a
stable per-channel min/max — one stray token swings the envelope and poisons the normalization. Push `L`
up to 1024 and locality breaks the other way: tokens 1024 apart are no longer similar, so the "next chunk
is a faithful sample" premise fails, and the per-window quota becomes so coarse I lose the fine-grained
resolution the rank step buys. So `lag_size = 128` is the balance point. The compression ratio I read
from the plan for provenance, the harness enforcing its own value. The assembled score vector is then
`n_sink` ones, the flattened per-partition ranks, `tail_len` ones; a single global top-K keeps the
sinks, the recent tail, and the top-ranked content tokens of every middle window — attention-free,
query-free, computed entirely by comparing the cached K and V among themselves, which is exactly what
H2O/SnapKV could not respect. The full scaffold module is in the answer.

The falsifiable expectations, against StreamingLLM's row. Retained stays ~0.20 and runtime near step 2's
— I drop the re-rotation (should trim the short-decode overhead) but add elementwise reductions over the
cache (min, max, std, softmax, double argsort), so the net is roughly a wash. The bet is that keeping the
*informative* middle tokens recovers what StreamingLLM's blind eviction lost, so I expect the retrieval/QA
workloads to climb toward the anchor: passage retrieval to beat 53.1 by a clear margin (per-window
coverage is the whole game on a needle-in-30-paragraphs task), hotpotqa to recover above 25.6, v2 to
stay near 29. The honest worry is gsm8k: my score is computed from the *prompt's* KV statistics, but
gsm8k's important tokens live in the model's own reasoning, so a prompt-coherence rule may still not
protect the chain-of-thought — if it stays near 1.7 rather than recovering, that says even content-aware
lag scoring on prompt KV is insufficient for reasoning and the next rung needs a score tied to what
*future queries* will attend to. And the sharpest line: repobench's recent-window keep is already
well-suited to next-line completion (it dropped only 9% from the anchor), so a rank rule spending budget
on incoherent middle tokens might lose a little there — if the retrieval recovery does not outweigh that
on the geometric mean, the content-aware story is not paying for itself. The retrieval recovery against
53.1 is the number I watch first.
