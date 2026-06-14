StreamingLLM answered the question I posed at step 2, and the answer is the one I was afraid of. It
cleared the budget — retained 0.20 across the board — and its runtimes sat right alongside the anchor.
But the accuracy gaps confirm the diagnosis I bet on. Hotpotqa fell from the anchor's 37.1 to 25.6, and
passage retrieval from 62.4 to 53.1: exactly the needle-in-a-haystack workloads where the answer can sit
anywhere in a long passage, and a purely positional rule that keeps sinks plus the recent window drops
the needle whenever it lives in the middle. LongBench v2 roughly held (29.6 vs 29.0), as expected for a
head-tail-truncated multiple-choice format. And the loud one: gsm8k cratered from 31.8 to 1.7. That is
the signal I said would be the hinge. The model's chain-of-thought reasoning prefix is precisely what
StreamingLLM shreds — it keeps the last 20% of tokens and four sinks, and on a reasoning trace the
load-bearing intermediate steps are scattered through the *middle* of the prompt, not concentrated at
the end, so the policy throws away the very content the final answer depends on. 1.7 is effectively
zero; the positional rule is the wrong shape for any workload whose important tokens are not positional.
So the verdict from step 2 is unambiguous: I need a *content-aware* score — something that decides which
middle tokens to keep, not just where they sit.

Now I have to be careful, because the obvious content-aware methods are exactly the ones this harness
forbids, and walking into that wall is how I would waste a rung. The strongest content-aware line scores
a token by how much attention it has received — H2O accumulates attention weight per token and keeps the
heavy hitters; SnapKV looks at what an observation window at the end of the prompt attends to and keeps
those prefix positions. Both work well in their papers. But both are disqualified here for two reasons,
and the second is fatal at the level of this hook. First, they are query-dependent: H2O's kept set
depends on the queries already issued, SnapKV's on where the question sits in the prompt — but I want a
compression that does not assume a particular question follows the context. Second, and decisive: they
need the realized attention matrix `softmax(QK^T/sqrt(d))`, and this hook does not hand me one.
`score_tokens` receives the module, hidden states, keys, values — and no attention tensor; the model runs
under SDPA, which streams the softmax-weighted output and never materializes the `t x t` weights. So a
score that is a function of the attention matrix simply cannot be computed in this harness. That is not
an inconvenience, it is a wall, and it is the same wall the prior-art note flagged. Quantization (KIVI,
CacheGen) is out for a different reason: it keeps *every* token at lower precision, so it never reduces
the number of attended tokens — it cannot hit a 20% retained budget by dropping tokens at all.

So the constraint sharpens to something concrete and demanding: score the middle tokens for importance
using *only the cached K and V tensors themselves*, with no attention weights and no query. Out of just
those tensors, can I tell which tokens are worth keeping? There is a thin precedent — the key-norm
heuristic found that a low L2 norm of a key correlates with high later attention, so keep low-key-norm
tokens. Right spirit (attention-free, query-free, uses the cache), but too thin: one scalar per token
from the key alone, ignoring the value, with no sense of where the token sits or how it relates to its
neighbors. A context-free per-key magnitude can rank isolated norms but cannot notice whether a token is
a local *discontinuity* in the flow of the cache — and a discontinuity is exactly the kind of
informative middle token StreamingLLM blindly dropped on gsm8k. So "use only the tensors" is the right
constraint, but the score has to extract structure, not just magnitude.

What do I actually know about how K and V are distributed? Two facts, both mapped out by the
quantization people for other purposes. First, token-wise locality: within a layer and channel, tokens
close together in position have much more similar K/V values than tokens far apart — the delta between
consecutive tokens' K (or V) is tightly piled near zero. That is not a coincidence; it falls out of the
model being autoregressive, the next token's representation does not jump abruptly from the previous
one's. Second, a K/V asymmetry: the key cache has a few *fixed* channels with persistently huge
magnitude (per-channel outliers — which is why you quantize keys per channel), while the value cache has
no such channel pattern and varies per token (quantize values per token). Sit with the channel-outlier
fact, because it is a trap. If a handful of key channels are always enormous, any naive
magnitude-or-variance score over the raw key vector is dominated by those same fixed channels for every
token — the score would mostly measure "which token had a slightly bigger value in the one giant
channel," which is persistent channel norm leaking through, not importance. The key-norm heuristic partly
works *despite* this; I want something that works *because* I have removed it. So before I can read
importance off the spread of a token's vector, I must strip out the channel-specific scale — normalize
each channel so the giant and tiny channels sit on the same footing. Against what statistics?

Token-wise locality pays off here. If adjacent tokens have nearly identical K/V, a short contiguous chunk
of tokens is a tight cloud, and its per-channel min and max are a faithful local description of each
channel's scale right there. So normalize a token's key (or value) by min-max over a nearby window: per
channel, subtract the window min, divide by the window range. That maps each channel into roughly [0,1]
*using statistics from the token's own neighborhood* — exactly the regime locality says is valid — and
the persistent giant channels get divided by their own large local range and collapse onto everyone
else's scale. What survives in the normalized vector is each channel's *relative* position within the
local cloud, with the channel-specific norms gone.

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
chunk does *not* explain: a discontinuity carrying information that does not follow from local flow. That
is the token I must keep — and it is precisely the middle token StreamingLLM could not distinguish.

Let me pin the axes, because the whole thing lives or dies on getting min/max/std over the right
dimension. Partition the cache after the sinks into contiguous chunks of size `L` (the lag). For chunk
`p`, head `i`, and `Z` standing for either K or V (a chunk of `L` tokens by `d_h` channels), take the
next chunk `Z^{p+1}` and compute, *per channel, over the token axis*, its min and max — collapsing the
`L` reference tokens to one min and one max per channel; that token-axis collapse is what makes the
normalization channel-wise. Then normalize the current chunk channel-by-channel with those reference
statistics: `(Z^p - min) / (max - min)`. Now read the spread *over the channels* of each normalized
token — its channel-wise standard deviation, one scalar per token. A coherent token has all its channels
in similar normalized positions, so small channel-wise std; an incoherent, surprising token has channels
scattered across the range, large std. That std is well-defined as an importance signal only because I
first removed the channel norms — otherwise it would just track the giant channels. I compute it for
both K and V and add them, because the two caches carry complementary structure (K channel-organized, V
token-organized), so a token can be a discontinuity in the key flow, the value flow, or both, and I want
all three. Before adding, I softmax each std vector over the `L` tokens of its chunk: that normalizes to
a per-chunk distribution and, because exp is convex, sharpens the tail so the genuinely high-std tokens
pull away from the merely above-average — the right behavior when I am about to keep a small fraction.

Now the bookkeeping, which I have to get exactly right or I reference off the end of the cache. The
sinks — the first `n_sink` tokens — are never scored and never evicted; that is the StreamingLLM anchor I
am keeping, because the attention-sink mechanism from step 2 is real and I still need the denominator
held. After the sinks, the very last chunk has *no chunk after it* to reference, so it cannot be scored —
and that is not a defect, it is exactly the recent sliding window I want anyway: the most recent tokens
matter most and should never be compressed. So the always-kept tail is one full lag window plus the
remainder when `(q_len - n_sink)` is not a multiple of `L`. The scored region runs from `n_sink` up to
`end_idx = n_sink + ((q_len - n_sink) // L) * L`, and `tail_len = L + (q_len - end_idx)`. If there is not
even enough sequence for one chunk to score and one to reference — `q_len < n_sink + 2L` — I skip the
lag rule entirely; but I do not want a forced top-K to keep arbitrary tokens, so in that branch I hand
back a ramp (`arange / (q_len - n_sink)` after the sinks) so any forced top-K degrades to keeping the
recent tail — StreamingLLM behavior as the graceful degenerate case, which is the right thing to degrade
to. I reshape the scored region into `(num_partitions, L, d_h)` and vectorize: the reference chunks are
partitions `1..` (`target[:, :, 1:]`), the scored chunks are `..num-2` (`target[:, :, :-1]`), so chunk
`p` is scored by chunk `p+1` in one shot, with the min/max over dim `-2`, the std over dim `-1`, and the
softmax over dim `-1`.

Then the question that actually decides the implementation, the one I almost glossed. I want top-K *per
partition* — within each lag window keep `rL` of `L`, so every region of the context contributes its
share and I do not accidentally keep a whole early window and evict a whole late one. But this harness
does the dead-simple thing: it computes one score vector over the entire cache and takes a single global
`topk(n_kept)`. If I hand it the raw softmax-std scores, a global top-K will not respect per-partition
quotas — a window with systematically larger raw std could steal the whole budget. The fix is to make the
scores themselves carry the per-partition structure so one global top-K *is* a per-partition top-K:
replace each partition's scores with their within-partition *rank*. Argsort the scores within the `L`
axis, argsort again — the double-argsort that returns each token's rank among its chunk-mates, an integer
in `{0..L-1}` — and divide by `L`. Now every partition's scores are the same set `{0, 1/L, ..., (L-1)/L}`
just permuted by importance, identically distributed across partitions, so the global top-K takes the
same count of high-rank tokens from each window. (Skipping the rank step — "cross_scoring" — lets budget
flow toward windows with larger raw outliers, a deliberate option, but it loses the uniform-per-partition
guarantee, so it is off by default here.) Crucially this also puts every scored token strictly in [0,1),
below the sinks and tail which I set to exactly 1.0, so those are always kept above any scored token.

The compression ratio I read from the plan, which the harness force-overrides at the call site anyway, so
I declare it for provenance but the harness enforces its own value — the same contract as step 2. So the
assembled score vector is: `n_sink` ones for the sinks, the flattened per-partition ranks for the scored
region, `tail_len` ones for the always-kept tail; a single global top-K on that vector keeps the sinks,
the recent tail, and the top-ranked content tokens of every middle window. That is the whole policy:
attention-free, query-free, computed entirely by comparing the cached K and V among themselves — which is
exactly what this hook permits and what H2O/SnapKV could not respect. The full scaffold module is in the
answer.

Let me close on the falsifiable expectations, against StreamingLLM's measured row and the anchor.
Retained stays at ~0.20 (the harness enforces it) and runtime should sit near step 2's, since the score
is a few elementwise reductions over the cache, cheap. The whole bet is that keeping the *informative*
middle tokens recovers the quality StreamingLLM's blind eviction lost. So I expect the LongBench
retrieval/QA workloads to climb back toward the anchor: passage retrieval should beat StreamingLLM's
53.1 by a clear margin (it keeps content tokens scattered through the passage), and hotpotqa should
recover above 25.6. LongBench v2 should stay near 29, neither rule having much purchase there. The honest
worry is gsm8k: my score is computed from the *prompt's* KV statistics, and gsm8k's important tokens are
in the model's own reasoning, so a prompt-statistics rule may still not protect the chain-of-thought —
if gsm8k stays near StreamingLLM's 1.7 rather than recovering toward 31.8, that says even content-aware
*lag* scoring on prompt KV is insufficient for reasoning, and the next rung needs a score tied more
directly to what *future queries* will attend to. And the sharpest falsifiable line: if LagKV does *not*
beat StreamingLLM on the geometric mean across the five workloads — if recovering retrieval does not
outweigh any code-similarity slippage on repobench (where StreamingLLM's recent-window keep is actually
well-suited to next-line completion, so I might lose a little there) — then the content-aware story is
not paying for itself and I have mis-diagnosed. The retrieval recovery against 53.1 is the number I am
watching first.
