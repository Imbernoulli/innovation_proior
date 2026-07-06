StreamingLLM answered the question I posed at step 2, and the answer is the one I was afraid of. It
cleared the budget — retained 0.20 across the board — and its runtimes came in on the shape I predicted.
Let me read the accuracy row as *drops from the anchor*, because the pattern of the drops is the
diagnosis, not any single number. Hotpotqa fell from 37.1 to 25.6, a drop of 11.5 — 31% of the ceiling
gone. Passage retrieval from 62.4 to 53.1, down 9.3, or 15%. Repobench from 47.6 to 43.2, down 4.4, 9%.
LongBench v2 actually ticked *up*, 29.0 to 29.6 — flat, exactly as its chance-floored headroom demanded
at step 1. And gsm8k collapsed from 31.8 to 1.7, a 95% wipeout. Now order those relative drops: gsm8k
95%, hotpotqa 31%, passage 15%, repobench 9%, v2 ~0%. That ordering is not noise; it is a clean ranking
of how *middle-distributed* each workload's load-bearing tokens are. gsm8k's chain-of-thought scatters
its intermediate steps through the whole prompt, so a rule that keeps only the last 20% throws away the
very content the final answer depends on — 1.7 is effectively zero, the model reasoning from a shredded
prefix. Hotpotqa's multi-hop needles sit in the middle. Passage retrieval's single needle can be
anywhere but is one span, so partial. Repobench's next-line completion is mostly local, so the recent
window was already well-suited and it barely moved. And v2 is the format that cannot move. The gradient
across those five drops is the entire case for what comes next.

The runtime row confirmed the prefill/decode model I built at step 2, and that confirmation is worth
spending because it tells me what the machinery cost. Against the anchor's 347/323/646/2589/7440,
StreamingLLM posted 361/331/672/2593/7110. The two cheap short-decode workloads and repobench ticked
*up* — 14, 8, and 26 seconds — because the per-layer top-K, gather, and re-rotation are paid up front on
every layer and are not recouped over a handful of decode steps. LongBench v2 stayed flat because it is
prefill-bound and the hook cannot touch the prefill. And gsm8k dropped 330 seconds because it is
decode-dominated and the 5x-smaller cache pays off on every one of its many decode steps. So the
compression itself has a nonzero cost, and re-rotation is a visible part of it — a fact I can spend at
this rung, because a content score that keeps tokens at their *original* positions can skip the
re-rotation entirely.

So the verdict from step 2 is unambiguous: I need a *content-aware* score — something that decides which
middle tokens to keep, not just where they sit. Now I have to be careful, because the obvious
content-aware methods are exactly the ones this harness forbids, and walking into that wall is how I
would waste a rung. The strongest content-aware line scores a token by how much attention it has
received — H2O accumulates attention weight per token and keeps the heavy hitters; SnapKV looks at what
an observation window at the end of the prompt attends to and keeps those prefix positions. Both work
well in their intended setting. But both are disqualified here for two reasons, and the second is fatal at the
level of this hook. First, they are query-dependent: H2O's kept set depends on the queries already
issued, SnapKV's on where the question sits in the prompt — but I want a compression that does not assume
a particular question follows the context. Second, and decisive: they need the realized attention matrix
`softmax(QK^T/sqrt(d))`, and this hook does not hand me one. `score_tokens` receives the module, hidden
states, keys, values — and no attention tensor; the model runs under SDPA, which streams the
softmax-weighted output and never materializes the `t x t` weights. So a score that is a function of the
attention matrix simply cannot be computed in this harness. That is not an inconvenience, it is a wall,
and it is the same wall the prior-art note flagged. Quantization (KIVI, CacheGen) is out for a different
reason I already worked through at step 1: it keeps *every* token at lower precision, so it never reduces
the number of attended tokens — the harness measures retained as a token count, so quantization reports
1.0 and cannot hit the budget by dropping tokens at all.

So the constraint sharpens to something concrete and demanding: score the middle tokens for importance
using *only the cached K and V tensors themselves*, with no attention weights and no query. Out of just
those tensors, can I tell which tokens are worth keeping? There is a thin precedent — the key-norm
heuristic found that a low L2 norm of a key correlates with high later attention, so keep low-key-norm
tokens. Right spirit (attention-free, query-free, uses the cache), but too thin: one scalar per token
from the key alone, ignoring the value, with no sense of where the token sits or how it relates to its
neighbors. A context-free per-key magnitude can rank isolated norms but cannot notice whether a token is
a local *discontinuity* in the flow of the cache — and a discontinuity is exactly the kind of informative
middle token StreamingLLM blindly dropped on gsm8k. So "use only the tensors" is the right constraint,
but the score has to extract structure, not just magnitude.

What do I actually know about how K and V are distributed? Two facts, both mapped out by the quantization
people for other purposes. First, token-wise locality: within a layer and channel, tokens close together
in position have much more similar K/V values than tokens far apart — the delta between consecutive
tokens' K (or V) is tightly piled near zero. That is not a coincidence; it falls out of the model being
autoregressive, the next token's representation does not jump abruptly from the previous one's. Second, a
K/V asymmetry: the key cache has a few *fixed* channels with persistently huge magnitude (per-channel
outliers — which is why you quantize keys per channel), while the value cache has no such channel pattern
and varies per token (quantize values per token). Sit with the channel-outlier fact, because it is a
trap. If a handful of key channels are always enormous, any naive magnitude-or-variance score over the
raw key vector is dominated by those same fixed channels for every token — the score would mostly measure
"which token had a slightly bigger value in the one giant channel," which is persistent channel norm
leaking through, not importance. Let me make that failure concrete so I know exactly what I must strip
out. Suppose channel 5 is a persistent giant: across a local window its values are around
`[1000, 1002, 1005, 1001]` while every other channel is order one. Any std taken over the raw key vector
is completely dominated by channel 5's ~1000-scale entry, and since channel 5 is a *persistent* outlier
its value barely differs token to token, so the raw std is nearly constant across tokens — it ranks
almost nothing. The key-norm heuristic partly works *despite* this; I want something that works *because*
I have removed it. So before I can read importance off the spread of a token's vector, I must strip out
the channel-specific scale — normalize each channel so the giant and tiny channels sit on the same
footing. Against what statistics?

Token-wise locality pays off here. If adjacent tokens have nearly identical K/V, a short contiguous chunk
of tokens is a tight cloud, and its per-channel min and max are a faithful local description of each
channel's scale right there. So normalize a token's key (or value) by min-max over a nearby window: per
channel, subtract the window min, divide by the window range. Take the giant channel 5 with local range
`[1000, 1005]`: a current-chunk token whose channel-5 value is `1003` maps to `(1003 - 1000)/5 = 0.6`,
landing on the same `[0,1]` footing as every ordinary channel, its 1000-scale magnitude divided away by
its own local range. What survives in the normalized vector is each channel's *relative* position within
the local cloud, with the persistent channel norms gone — exactly the thing I need before any spread
measure means anything.

But which window's min and max? The obvious choice — the token's own chunk — is circular. If I min-max a
chunk by its own statistics and ask which token is unusual, I am asking which token is an outlier
relative to its chunk-mates, a self-referential question; worse, the min and max are *achieved by* tokens
in the chunk, so those extreme tokens get pinned to exactly 0 or 1 by construction, biasing the score
toward whichever tokens happened to define the envelope. I want a reference *external* to the tokens I am
scoring, so "unusual" means something. And locality hands me the external yardstick for free: the *next*
chunk. Because the model is autoregressive and adjacent tokens barely move, the chunk immediately after
the current one is a faithful sample of the same local distribution — the current chunk's tokens are
"supposed to" look like the next chunk's. So normalize the current chunk's K and V using the min and max
taken from the *next* chunk. Now the question I ask each token is sharp and query-free: how coherent is
this token with what comes right after it? A token whose normalized vector sits comfortably inside the
next chunk's envelope is predictable — the local flow was going to produce it anyway, so dropping it
loses little. A token that is still strange after normalizing by the next chunk's scale is one the next
chunk does *not* explain: a discontinuity carrying information that does not follow from local flow.

Let me trace that on a toy so I can see the score actually separate a coherent token from a discontinuity.
Take two channels and a lag of four, and say the *next* chunk's per-channel envelope is channel 0 in
`[2, 6]` (range 4) and channel 1 in `[10, 14]` (range 4). Now score two current-chunk tokens. Token A is
`[4, 12]`: normalized it is `[(4-2)/4, (12-10)/4] = [0.5, 0.5]`, and the channel-wise std of `[0.5, 0.5]`
is `0`. It sits dead-center in both channels of the next chunk's envelope — perfectly coherent, cheap to
drop. Token B is `[2.4, 13.6]`: normalized it is `[(2.4-2)/4, (13.6-10)/4] = [0.1, 0.9]`, and the std of
`[0.1, 0.9]` is `0.4`. Its two channels sit at opposite ends of the local envelope — an unusual shape the
next chunk does not predict, a discontinuity worth keeping. So the channel-wise standard deviation of the
next-chunk-normalized token *is* the importance signal, and it is well-defined only because I removed the
channel norms first — otherwise, as I just showed, it would track channel 5 and nothing else. I compute
it for both K and V and add them, because the two caches carry complementary structure (K
channel-organized, V token-organized), so a token can be a discontinuity in the key flow, the value flow,
or both, and I want all three. Before adding, I softmax each std vector over the `L` tokens of its chunk:
that normalizes to a per-chunk distribution and, because exp is convex, sharpens the tail so the genuinely
high-std tokens pull away from the merely above-average — the right behavior when I am about to keep a
small fraction.

Now the bookkeeping, which I have to get exactly right or I reference off the end of the cache. The sinks
— the first `n_sink` tokens — are never scored and never evicted; that is the StreamingLLM anchor I am
keeping, because the attention-sink mechanism from step 2 is real and I still need the denominator held,
and it cost 30 accuracy points on gsm8k to learn how real it is. After the sinks, the very last chunk has
*no chunk after it* to reference, so it cannot be scored — and that is not a defect, it is exactly the
recent sliding window I want anyway: the most recent tokens matter most and should never be compressed. So
the always-kept tail is one full lag window plus the remainder when `(q_len - n_sink)` is not a multiple
of `L`. The scored region runs from `n_sink` up to `end_idx = n_sink + ((q_len - n_sink) // L) * L`, and
`tail_len = L + (q_len - end_idx)`. If there is not even enough sequence for one chunk to score and one to
reference — `q_len < n_sink + 2L` — I skip the lag rule entirely; but I do not want a forced top-K to keep
arbitrary tokens, so in that branch I hand back a ramp (`arange / (q_len - n_sink)` after the sinks) so any
forced top-K degrades to keeping the recent tail — StreamingLLM behavior as the graceful degenerate case,
which is the right thing to degrade to. I reshape the scored region into `(num_partitions, L, d_h)` and
vectorize: the reference chunks are partitions `1..` (`target[:, :, 1:]`), the scored chunks are
`..num-2` (`target[:, :, :-1]`), so chunk `p` is scored by chunk `p+1` in one shot, with the min/max over
dim `-2`, the std over dim `-1`, and the softmax over dim `-1`.

Then the question that actually decides the implementation, the one I almost glossed. I want top-K *per
partition* — within each lag window keep `rL` of `L`, so every region of the context contributes its
share and I do not accidentally keep a whole early window and evict a whole late one. But this harness
does the dead-simple thing: it computes one score vector over the entire cache and takes a single global
`topk(n_kept)`. If I hand it the raw softmax-std scores, a global top-K will not respect per-partition
quotas — a window with systematically larger raw std could steal the whole budget. The fix is to make the
scores themselves carry the per-partition structure so one global top-K *is* a per-partition top-K:
replace each partition's scores with their within-partition *rank*. Let me verify that the
double-argsort actually produces ranks, because the whole quota guarantee rests on it. Take a partition of
four with raw scores `[0.3, 0.9, 0.1, 0.5]`. The first `argsort` returns the indices that sort ascending:
the sorted order is `0.1, 0.3, 0.5, 0.9` at original indices `[2, 0, 3, 1]`. The second `argsort` of
`[2, 0, 3, 1]` returns `[1, 3, 0, 2]` — and that is exactly the rank of each original token: token 0 has
rank 1, token 1 (the largest, 0.9) has rank 3, token 2 (the smallest, 0.1) has rank 0, token 3 has rank 2.
Divide by `L = 4` and the partition becomes `[0.25, 0.75, 0.0, 0.5]`. The point is that *every* partition,
whatever its raw scores, ends up as some permutation of the identical set `{0, 0.25, 0.5, 0.75}`. So a
single global top-K that has budget for, say, two per window takes the ranks `{0.5, 0.75}` from *every*
partition — the global operation has become a per-partition top-K, quotas enforced by construction.
(Skipping the rank step — "cross_scoring" — lets budget flow toward windows with larger raw outliers, a
deliberate option, but it loses the uniform-per-partition guarantee, so it is off by default here.)
Crucially the largest rank is `(L-1)/L < 1`, so every scored token sits strictly in `[0, 1)`, below the
sinks and tail which I set to exactly `1.0` — those are therefore always kept above any scored token.

That also settles why I do *not* re-rotate here, unlike step 2, and it is a real distinction, not a
convenience. StreamingLLM kept two far-apart blocks — four sinks and one contiguous recent window — so it
relabeled them to contiguous cache positions to bound the sink-to-window gap. LagKV keeps tokens
*distributed* across the whole context: sinks, then a share of every 128-token window, then the recent
tail. Those kept positions tile the original context roughly uniformly, and every kept token keeps its
*true* original position, so the relative distances the model reads are exactly the ones it saw at full
attention — which are valid because this is a one-shot within-context prefill, not a stream rolling past
the trained length. If I re-rotated LagKV's distributed keep to contiguous positions I would be *lying*
about the geometry: two tokens that were 130 apart would be told they are one apart, corrupting the
relative-position signal that carries meaning. So `rerotate_selected_keys = False` is correct, and it also
saves the per-layer re-rotation cost I watched StreamingLLM pay on the short-decode workloads.

The one hyperparameter with real freedom is the lag size, and the budget arithmetic pins it. With retain
`0.2`, a window of `L` keeps about `int(0.2 L)` tokens. At `L = 128` that is ~25 tokens kept per window —
enough to preserve a fact that spans several tokens, and enough that each neighborhood gets a real,
non-trivial share of the budget. Push `L` down to 8 and I keep `int(0.2 * 8) = 1` token per window: a
multi-token fact cannot survive one kept slot, and worse, the *reference* chunk is only 8 tokens, far too
few to estimate a stable per-channel min/max — one stray token in the reference would swing the envelope
and poison the normalization. Push `L` up to 1024 and locality breaks the other way: tokens 1024 apart
are no longer similar, so the "next chunk is a faithful sample of the same local distribution" premise
that justified the whole normalization fails, and the per-window quota becomes so coarse (keep ~205 in one
lump) that I lose the fine-grained "keep the informative tokens in *this* neighborhood" resolution I built
the rank step to get. So `lag_size = 128` is the balance point: short enough that adjacent-within-window
tokens really are similar and the reference envelope is faithful, long enough that `rL` keeps multi-token
facts and every neighborhood contributes. The compression ratio I read from the plan, which the harness
force-overrides at the call site anyway, so I declare it for provenance but the harness enforces its own
value — the same contract as step 2. So the assembled score vector is: `n_sink` ones for the sinks, the
flattened per-partition ranks for the scored region, `tail_len` ones for the always-kept tail; a single
global top-K on that vector keeps the sinks, the recent tail, and the top-ranked content tokens of every
middle window. That is the whole policy: attention-free, query-free, computed entirely by comparing the
cached K and V among themselves — which is exactly what this hook permits and what H2O/SnapKV could not
respect. The full scaffold module is in the answer.

Let me close on the falsifiable expectations, against StreamingLLM's measured row and the anchor. Retained
stays at ~0.20 (the harness enforces it) and runtime should sit near step 2's — I dropped the re-rotation,
which should trim the small overhead the short-decode workloads paid, but I added a handful of elementwise
reductions (min, max, std, softmax, a double argsort) over the cache, so the net is roughly a wash and I
will read the row to see which effect won. The whole bet is that keeping the *informative* middle tokens
recovers the quality StreamingLLM's blind eviction lost. So I expect the LongBench retrieval/QA workloads
to climb back toward the anchor: passage retrieval should beat StreamingLLM's 53.1 by a clear margin (it
keeps content tokens scattered through the passage, and a needle-in-30-paragraphs task is exactly where
per-window coverage helps), and hotpotqa should recover above 25.6. LongBench v2 should stay near 29,
neither rule having much purchase on a chance-floored format. The honest worry is gsm8k: my score is
computed from the *prompt's* KV statistics, and gsm8k's important tokens are in the model's own reasoning,
so a prompt-statistics rule may still not protect the chain-of-thought — if gsm8k stays near StreamingLLM's
1.7 rather than recovering toward 31.8, that says even content-aware *lag* scoring on prompt KV is
insufficient for reasoning, and the next rung needs a score tied more directly to what *future queries*
will attend to. And the sharpest falsifiable line: if LagKV does *not* beat StreamingLLM on the geometric
mean across the five workloads — if recovering retrieval does not outweigh a code-similarity slippage on
repobench, where StreamingLLM's recent-window keep is actually well-suited to next-line completion (recall
its repobench dropped only 9% from the anchor, the smallest of the four), so a rank rule that spends budget
on incoherent middle tokens might lose a little there — then the content-aware story is not paying for
itself and I have mis-diagnosed. The retrieval recovery against 53.1 is the number I am watching first.
