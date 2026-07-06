There is no measured result to react to yet, so the job of this first rung is narrow: pick the
floor, install it in the scaffold, and be precise about *why* it is the floor and *what it will fail
at*, so the number it produces tells me something. The scaffold compresses by scoring the prefill KV
and keeping a top-K subset; the question the whole task asks is which tokens are safe to drop. Before I
answer that, I want the run that drops *nothing* — the uncompressed full cache — because it pins two
things I cannot reason about otherwise: the quality ceiling of this exact model on these exact
workloads, and the budget penalty that any real compressor has to clear. Everything later is measured
against this row.

Let me also be honest about why I cannot skip this rung and just start at a compressor. Every later
score I report is a *relative* statement — "this method keeps 90% of full-context quality at 20% of the
cache" only means something if I know the 100% number on this exact model, these exact prompt
templates, this exact greedy decode loop, and this exact scoring code. The benchmark-native scores are
not transferable from a table I read somewhere: LongBench F1 depends on the harness's normalization,
repobench's code-similarity depends on which line of the generation it compares, gsm8k depends on the
boxed-answer extraction and numeric normalization the scaffold uses. A retrieval score of 62 in one
harness can be 55 in another purely from prompt formatting and answer parsing. So the only trustworthy
reference is one produced *by this harness*, and that is what running the uncompressed policy here
gives me. Without it I would be guessing at headroom — and a compressor that lands at, say, 30 on some
workload is either a triumph or a disaster depending on whether the ceiling above it is 33 or 62, and I
cannot tell which of those two stories I am in until this row exists.

Two properties of the harness shape how much I can trust any single gap I read, and I want them in mind
from the start. The decode is greedy — argmax at every step, no sampling — so the generation is a
deterministic function of the cache; there is no sampling noise to average away, which is convenient,
but it also means a compressor either flips a token or it does not, and near a decision boundary a
single evicted key can change one argmax and cascade the rest of the generation. And there is one seed,
42, so each reported mean is a single point estimate with no spread to tell me whether a two-point
accuracy difference between rungs is mechanism or luck-of-the-draw on which examples happened to tip.
That does not make the numbers untrustworthy — each workload aggregates over many examples — but it does
mean I should weight *large, structured* gaps (a whole workload collapsing, a clean per-workload
pattern) far above small ones, and not build a story on a single two-point wiggle. The anchor's job is
to make those gaps measurable; reading them responsibly is on me.

Let me be exact about what "keep everything" is, because it is not a no-op — it is a deliberate policy
on the same hook as every other rung. The harness runs a standard full-attention prefill: the model
sees the whole prompt at once, fills the per-layer KV cache with `k_1..k_p, v_1..v_p`, and the
per-layer forward hook fires with that filled cache. At that moment the controller gets to decide the
layer's retention. The reason the full cache is the principled reference, and not merely a lazy
default, comes straight from the causal structure of the decoder. In a masked self-attention layer the
key `k_j` and value `v_j` of position `j` are computed from token `j` and the layers beneath it, and
the mask means information flows only forward in position — no later position `i > j` can reach back and
change `k_j` or `v_j`. So once position `j` has been processed, its `(k_j, v_j)` is frozen for the rest
of the sequence.

I do not want to take that freezing on faith, because the whole claim that the full cache is *exact*
rests on it, so let me trace it on a tiny prefix. Take three prefill tokens `x_1, x_2, x_3` and a
fourth token `x_4` that decode is about to produce. The key at position `j` is `k_j = R_j W_k x_j` — a
projection of `x_j` alone, rotated by the position-`j` RoPE matrix `R_j`; it has no dependence on any
`x_i` with `i > j`. When position 2 was processed during prefill, the causal mask zeroed its attention
logits against positions 3 and 4, so whatever those later tokens are, they never entered the
computation that produced `k_2` or `v_2`. Now decode step 4 attends `q_4` over the stored
`{k_1, k_2, k_3}` plus its own freshly projected `k_4`. Compare that to the counterfactual where I
threw the cache away and re-ran a full-attention forward over `x_1..x_4`: the re-run would recompute
`k_2 = R_2 W_k x_2`, bit-for-bit the tensor I already had stored, because nothing downstream of
position 2 feeds that projection. So the stored-cache read and the full-recompute read are the *same*
read, not an approximation of it. That is the content of "exact": keeping the whole frozen set means
the decoded sequence follows the identical mathematical computation as re-running the model on the full
prefix at each step — only without the recomputation. No token is dropped, nothing is summarized, so
whatever the model can do with full context, it does here. That is the ceiling, and it is exact by
construction: there is no scoring rule clever enough to *exceed* it on quality, only rules that try to
approach it with a fraction of the cache.

Now the cost, because the ceiling is precisely the thing I am being asked to give up, and I want to put
real numbers on the wall rather than gesture at it. The full cache's size is
`b * h_kv * (k_dim + v_dim) * L * n` — linear in sequence length `n` and depth `L`. I can read the head
count straight off the interface: `score_tokens` must return a tensor shaped `(batch, num_kv_heads,
k_len)`, so the cache is stored per *KV* head, not per query head — this model uses grouped-query
attention. For `Qwen2.5-3B-Instruct` that is 2 KV heads of dimension 128 across 36 layers. So per token
per layer the cache holds `2 * 128` key elements plus `2 * 128` value elements, 512 numbers, which at
bf16's two bytes is 1 KB per token per layer, and across the 36-layer stack about 36 KB per token. A
30k-token LongBench prompt is then roughly a gigabyte of KV for a single request. The generic framing
where the cache "rivals the model weights" assumes full multi-head attention; here the grouped-query
layout has already bought an ~8x reduction over a 16-head cache (16 query heads share 2 KV heads), so
the absolute number is smaller than that framing suggests — call it a sixth of the ~6 GB of weights,
not a match for them. But the *structure* of the problem is identical: it grows linearly in `n`, and it
must be reloaded in full at every decode step.

And that reload is the real cost, not the storage. At each decode step, per layer, the arithmetic is a
query attending over the whole history: the `QK^T` product is `16` query heads times `n` keys times
`128` dims of multiply-accumulate, and the `AV` product another `16 * n * 128`, so on the order of
`4096 * n` floating-point operations. Against that I have to move the entire K and V from memory, about
`1024 * n` bytes. The arithmetic intensity is therefore around `4` FLOP per byte — and that number is
the whole story, because any modern accelerator's ratio of compute throughput to memory bandwidth sits
two orders of magnitude higher, on the order of `10^2` FLOP per byte at the ridge point. Four is far
below the ridge, so a long-context decode step is squarely *bandwidth-bound*: its wall-clock time is
set by how many cache bytes it drags out of HBM, not by the multiplies. That is exactly why chasing a
20% cache is worth the trouble — in the bandwidth-bound regime a fifth of the bytes is close to a fifth
of the decode-step time, which is precisely what the leaderboard's runtime term will reward. The honest
ledger of the full cache is a state and a per-step bandwidth that both grow with the sequence;
compression trades a slice of the quality ceiling for a bounded, fast-to-stream cache. This rung is the
point on that trade where the slice given up is zero and the cache is unbounded — the opposite extreme
from where any deployable controller must live.

One more thing the runtime accounting forces me to be precise about, because it bounds what *any*
compressor in this ladder can achieve. The total time per request splits into a prefill phase and a
decode phase. Prefill runs the whole prompt through full attention in one shot — that is `O(n^2)` in the
prompt length, quadratic, and it happens *before* the hook fires, because the harness explicitly does a
standard full-attention prefill and only then calls the policy to compress. Decode is the sequential
part: each generated token attends over the now-compressed cache, `O(n)` per step times the number of
generated tokens. The consequence is sharp and I need to carry it forward: compression touches *only the
decode cache*. It cannot shrink the prefill at all — the prefill is full attention no matter which policy
I install. So on a workload with a very long prompt and a short answer (a handful of decode steps), the
runtime is prefill-dominated and *no* compressor can move it much; the quadratic prefill swamps the few
cheaper decode steps. On a workload with a moderate prompt and a long generation (many decode steps), the
runtime is decode-dominated and compression's 5x-smaller cache pays off on every one of those steps. That
split predicts which workloads the runtime term can even reward a compressor on — the long-generation
ones — and it means I should read a compressor's runtime not as one win-or-loss but per workload, against
how decode-heavy each workload is. The anchor's runtime row is where I first get to see that split drawn
out in wall-clock seconds.

I should also be explicit about what makes *this* the right anchor rather than some "cheap but lossless"
trick, because it is tempting to think there is a free lunch, and there are three superficially
attractive alternatives that are not it. Windowing — cap attention to the last `L` positions — bounds
memory and bandwidth, but a query can no longer see anything outside the window, so any dependency
longer than `L` is silently dropped and the model's output diverges from its full-context output.
Memory compression — pool past positions into fewer key/value slots — has the same character: the query
attends over fewer, summarized slots instead of the true set, so the output again diverges. Both are
legitimate speed/quality trades, and indeed the later rungs *are* exactly such trades; but neither is
the model's actual full-prefix computation, so neither can serve as the quality reference. The third
temptation is the one worth killing carefully, because it looks like it should satisfy the budget for
free: quantization of the cache (KIVI, CacheGen), storing every token's KV at four bits instead of
sixteen. That saves 4x on memory and bandwidth without dropping a single token — sounds ideal. But look
at how this harness *defines* the budget: it measures the retained fraction as `n_kept / k_len` per
layer, averaged — a *token count*, not a byte count. A 4-bit cache still keeps every token, so it
reports retained = 1.0 and eats the identical budget penalty as the uncompressed cache, its 4x memory
win invisible to the leaderboard. So the entire quantization family is disqualified at the level of this
scoreboard, not on quality but on definition: the budget is a constraint on tokens *attended*, and the
only way to move off retained = 1.0 is to *drop tokens*. That sharpens what the anchor is. It is the
token-count extreme — keep every token — and every later rung must be a genuine eviction. Full KV,
exact, full context, no eviction: precisely the property that qualifies it as the ceiling and
disqualifies it as a budgeted submission.

Let me be clear-eyed about where this lands on the scoreboard, because that is the entire point of
running it. The leaderboard combines, per workload, accuracy at weight 6, runtime at weight 2, and
cache reduction at weight 2, then takes a geometric mean across the five workloads — and it applies a
soft upper-bound penalty the moment `mean_retained_fraction` exceeds 0.25. Let me reason through where
the full cache falls before I run it. Accuracy is 60% of each workload's weight, and the full cache
*maxes* it — nothing can beat the exact-context ceiling. But the reduction term, a fifth of the weight,
is a function of how much cache I dropped, and I dropped none, so it sits at its floor. And retained 1.0
is four times over the 0.25 tolerance, so the soft penalty fires at full strength. I do not actually
know the exact per-workload combiner — whether the three normalized terms are blended arithmetically or
geometrically before the cross-workload geometric mean — and that ambiguity is worth naming because it
changes the arithmetic: under a geometric blend a floored reduction term drives the whole workload
score toward zero, while under an arithmetic blend it caps the workload near `0.6 *` its accuracy term
before the penalty scales the result down further. Either reading gives the same verdict. The full cache
posts the highest raw accuracy anything in this ladder will ever reach and converts it into the worst
*budgeted* score, because the two budget-sensitive terms — reduction at its floor, penalty at full
strength — overwhelm the accuracy the constraint exists to protect. It is the weakest submission
precisely because it refuses the task's central constraint, while being the strongest on the one axis
(quality) that the constraint exists to protect. That tension is the thing the next rungs have to
resolve: get as close as possible to this accuracy while living at ~20% retained.

There is a second reason the anchor earns its keep beyond being the quality target, and it is about how
the leaderboard *normalizes*. The runtime and reduction terms are described only as "normalized," and
something has to set the scale they are normalized against; the full-attention run is the natural
reference for that, since it is the slowest decode and the largest cache, plausibly anchoring the top of
the runtime scale and the bottom of the reduction scale that every compressed submission is then read
against in relative terms. I will not pretend to know the exact normalization — the context says
"normalized" and stops there — but whatever it is, having the extreme row measured is what makes every
later relative number legible. The tolerance arithmetic, at least, is unambiguous: the penalty spares me
up to 0.25 retained, I am aiming the compressors at ~0.20, which leaves a thin 0.05 margin of comfort,
and the anchor at 1.0 sits 4x past the line — not near the boundary but deep in penalty territory, which
is exactly the visible-but-invalid role the task designed it for.

So how do I land "keep everything" on this specific hook? The contract has three methods —
`retention_plan`, `score_tokens`, `select_cache`. The cleanest way to express no compression is not to
return a score that happens to keep all tokens, because the harness still computes `n_kept = int(k_len *
(1 - ratio))` from its own force-overridden ratio and would top-K me down to ~20% regardless of what I
return — an all-ones score would not keep everything, it would keep an arbitrary top-K of ~20%. The way
to actually keep everything is to *tell the harness not to compress at all*. The plan field
`disable_compression` is the lever: when it is `True`, the harness skips `score_tokens` and
`select_cache` entirely, leaves the cache untouched, and records the retained fraction as exactly 1.0.
This is the one plan field besides the budget that the harness genuinely enforces — it is observable
from the post-selection cache state — so setting it is not advisory bookkeeping, it is the actual
no-compression switch. `score_tokens` then returns `None` (the harness reads `None` as "keep
everything," another path to the same retained = 1.0) and `select_cache` returns the keys and values
unchanged. I set `rerotate_selected_keys = False` because nothing is being re-positioned: the cache
keeps its original token positions, so the RoPE phases the prefill computed are already correct and the
decode positions continue from the true sequence length. There is no re-rotation, no gather, no top-K —
the policy's whole content is the declaration that this layer is not to be compressed.

It is worth noticing what the substrate went out of its way to expose, because it maps the design space
I am about to walk. The harness wires `module.rotary_emb` onto every attention module specifically so
that a policy which re-rotates kept keys has the rotary table on hand — that is a signpost that
position-remapping compressors are an anticipated move, not an exotic one, and I file it away for the
first rung that actually reorders the cache. And the `n_kept = int(k_len * (1 - ratio))` formula the
harness computes has a corner I should keep in view even though the anchor sidesteps it: on a short
prompt `int(k_len * 0.2)` is tiny — a 20-token prompt yields `n_kept = 4` — so a compressor that reserves
a few sink slots can find the whole budget consumed by sinks with nothing left for a window. The
anchor's `disable_compression` bypasses all of this, but the contract's arithmetic is now in front of me
for when it starts to bite on the compressors.

There is one workload I can predict the *shape* of a priori, before any number lands, and it will
matter for how I read the whole ladder. LongBench v2 is four-way multiple choice scored by exact
accuracy, so its random-guess floor is ~25, and a 3B instruct model on a deliberately hard long-context
multiple-choice set is unlikely to clear that floor by a wide margin. Narrow headroom between the ~25
chance floor and whatever ceiling the anchor posts means longbench_v2 is a *low-discrimination* workload:
every method, good or bad, will cluster in the same few points above chance, and its accuracy should
barely move as I change the retention rule. So I expect the discriminating signal in this ladder to come
from the open-ended workloads — hotpotqa's span F1, passage retrieval's needle accuracy, repobench's
code similarity, and gsm8k's exact-answer accuracy — where a compressor that drops the wrong tokens has
real room to fall. I will keep half an eye on longbench_v2 as a floor-consistency check, but I will not
let a flat longbench_v2 talk me into thinking a method is safe; the verdict lives in the four workloads
with headroom, and above all in gsm8k, the one place a positional rule can slide from "fine" to "zero."

What I will watch in the numbers, and what they have to set up for step 2. First, the per-workload
accuracy spread. I expect the retrieval and QA workloads (passage retrieval, hotpotqa) to be where full
context helps most, because the answer can sit anywhere in a long passage and the model needs to reach
the exact span; and the reasoning workload (gsm8k) to show whatever this 3B instruct model can do with
its full chain-of-thought intact. That gsm8k number is the one I care about most as a headroom figure,
because a compressor that shreds the reasoning prefix will crater there first — if the ceiling on gsm8k
is already low, I have little to lose there and the workload is uninformative; if it is healthy, gsm8k
becomes the sharpest test of whether a positional rule can survive on reasoning. Second, the runtime
row. The full cache is the *slowest* per-token decode because it streams the largest cache, so it
calibrates the runtime term every compressed method will improve on; and I expect the runtimes to rank
not by prompt length alone but by total decode *work*, roughly total generated tokens times cache size,
so the workload with the longest generations over long contexts should dominate the row. I have no prior
runtime to anchor that ordering, so it is a genuine prediction the table will confirm or refute rather
than something I already know. Third, and most important for the ladder: this row is the accuracy
target. When I drop to a sink-plus-window rule at step 2, the gap between its accuracy and this row's
accuracy — per workload — is exactly the quality I paid for the 5x cache reduction, and which workloads
lose the most will tell me whether a *positional* retention rule is even the right shape, or whether I
need a content-aware score. The floor is installed; the only way is to compress, and the first
compression I will try is the cheapest, most robust one — keep the sinks and the recent window — so that
its failures, read against this anchor, point at what a smarter score has to recover. The full scaffold
module for this rung is in the answer.
