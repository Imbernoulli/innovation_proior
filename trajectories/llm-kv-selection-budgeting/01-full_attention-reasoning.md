There is no measured result to react to yet, so the job of this first rung is narrow: pick the
floor, install it in the scaffold, and be precise about *why* it is the floor and *what it will fail
at*, so the number it produces tells me something. The scaffold compresses by scoring the prefill KV
and keeping a top-K subset; the question the whole task asks is which tokens are safe to drop. Before I
answer that, I want the run that drops *nothing* — the uncompressed full cache — because it pins two
things I cannot reason about otherwise: the quality ceiling of this exact model on these exact
workloads, and the budget penalty that any real compressor has to clear. Everything later is measured
against this row.

I cannot skip this rung and start at a compressor, because every later score I report is a *relative*
statement — "this method keeps 90% of full-context quality at 20% of the cache" only means something if
I know the 100% number on this exact model, these exact prompt templates, this exact greedy decode
loop, and this exact scoring code. The benchmark-native scores are not transferable from a table I read
somewhere: LongBench F1 depends on the harness's normalization, repobench's code-similarity on which
line of the generation it compares, gsm8k on the boxed-answer extraction and numeric normalization the
scaffold uses. A retrieval score of 62 in one harness can be 55 in another purely from prompt
formatting. So the only trustworthy reference is one produced *by this harness* — and a compressor that
lands at 30 on some workload is a triumph or a disaster depending on whether the ceiling above it is 33
or 62, which I cannot tell until this row exists.

Two properties of the harness decide how much I can trust any single gap. The decode is greedy — argmax
at every step, no sampling — so the generation is a deterministic function of the cache: no sampling
noise to average away, but it also means a compressor either flips a token or does not, and near a
decision boundary a single evicted key can change one argmax and cascade the rest of the generation.
And there is one seed, 42, so each reported mean is a single point estimate. Each workload still
aggregates over many examples, but the consequence is that I should weight *large, structured* gaps — a
whole workload collapsing, a clean per-workload pattern — far above a two-point wiggle, and not build a
story on a single small difference between rungs.

"Keep everything" is not a no-op; it is a deliberate policy on the same hook as every other rung. The
harness runs a standard full-attention prefill — the model sees the whole prompt at once, fills the
per-layer cache with `k_1..k_p, v_1..v_p`, and the per-layer forward hook fires with that filled cache —
and at that moment the controller decides the layer's retention. The full cache is the principled
reference, not a lazy default, because of causal structure: in a masked self-attention layer `k_j` and
`v_j` are projections of token `j` (and the layers beneath it), and the mask means no later position
`i > j` can reach back and change them, so once position `j` is processed its `(k_j, v_j)` is frozen for
the rest of the sequence. The stored `k_j = R_j W_k x_j` is bit-for-bit the tensor a from-scratch
full-attention re-run over the whole prefix would recompute, because nothing downstream of `j` feeds
that projection. So keeping the entire frozen set makes the read at every decode step the *same* read
as re-running the model on the full prefix — not an approximation of it, only cheaper. That is the
content of "exact": whatever the model can do with full context, it does here, and no scoring rule can
*exceed* this on quality, only approach it with a fraction of the cache.

Now the cost, because the ceiling is precisely what I am being asked to give up. The full cache is
`b * h_kv * (k_dim + v_dim) * L * n` — linear in sequence length `n` and depth `L`. I read the head
count off the interface: `score_tokens` returns a tensor shaped `(batch, num_kv_heads, k_len)`, so the
cache is stored per *KV* head — this model uses grouped-query attention, 2 KV heads of dimension 128
across 36 layers. Per token per layer that is `2*128` key plus `2*128` value elements, 512 numbers, ~1
KB at bf16, ~36 KB across the stack; a 30k-token prompt is then roughly a gigabyte of KV. The generic
"the cache rivals the model weights" framing assumes full multi-head attention — here grouped-query has
already bought an ~8x reduction (16 query heads share 2 KV heads), so the absolute number is a fraction
of the ~6 GB of weights, not a match for them. But the structure is identical: it grows linearly in `n`
and must be reloaded in full at every decode step. That reload, not the storage, is the real cost —
a decode step's arithmetic intensity is only a few FLOP per byte, two orders of magnitude below any
accelerator's compute/bandwidth ridge, so a long-context decode is squarely *bandwidth-bound*: its
wall-clock time is set by how many cache bytes it drags out of memory. That is exactly why chasing a
20% cache is worth it — in the bandwidth-bound regime a fifth of the bytes is close to a fifth of the
decode-step time, which is what the leaderboard's runtime term rewards.

But the runtime accounting also bounds what *any* compressor here can achieve, and I need to carry this
forward. Total time splits into prefill and decode. Prefill runs the whole prompt through full attention
in one shot — `O(n^2)`, quadratic — and it happens *before* the hook fires; the harness does a standard
full-attention prefill and only then calls the policy. Decode is the sequential part, `O(n)` per step
over the now-compressed cache. So compression touches *only the decode cache*: it cannot shrink prefill
at all. On a workload with a very long prompt and a short answer the runtime is prefill-dominated and no
compressor can move it much; on one with a moderate prompt and a long generation the 5x-smaller cache
pays off on every decode step. That split predicts which workloads the runtime term can even reward a
compressor on — the long-generation ones — so I should read runtime per workload, against how
decode-heavy each is.

I should also be clear about why this is the right anchor rather than a "cheap but lossless" trick.
Windowing and memory-pooling both bound the cache but make the query attend over fewer, summarized, or
truncated positions, so the output diverges from the full-context output — legitimate speed/quality
trades (the later rungs *are* such trades), but not the model's true full-prefix computation, so neither
can be the quality reference. The one worth killing carefully is quantization (KIVI, CacheGen): store
every token's KV at four bits, 4x memory saved, no token dropped — sounds ideal. But this harness
measures the retained fraction as `n_kept / k_len` per layer, a *token count*, not a byte count. A 4-bit
cache still keeps every token, reports retained = 1.0, and eats the identical budget penalty, its memory
win invisible to the leaderboard. So the whole quantization family is disqualified at the level of the
scoreboard, not on quality but on definition: the only way off retained = 1.0 is to *drop tokens*. That
sharpens what the anchor is — the token-count extreme, keep every token — and every later rung must be a
genuine eviction.

Where does this land on the scoreboard? Accuracy is 60% of each workload's weight and the full cache
maxes it — nothing beats the exact-context ceiling. But the reduction term, a fifth of the weight, is a
function of how much cache I dropped, which is none, so it sits at its floor; and retained 1.0 is four
times over the 0.25 tolerance, so the soft penalty fires at full strength. I do not know the exact
per-workload combiner — whether the three normalized terms blend arithmetically or geometrically before
the cross-workload geometric mean — but either reading gives the same verdict: the highest raw accuracy
anything in this ladder will reach, converted into the worst *budgeted* score, because the two
budget-sensitive terms overwhelm the accuracy the constraint exists to protect. That is the tension the
next rungs resolve: get as close as possible to this accuracy while living at ~20% retained. The
tolerance arithmetic is the one unambiguous part — the penalty spares me up to 0.25 retained, I aim the
compressors at ~0.20 (a thin 0.05 margin), and the anchor at 1.0 sits deep in penalty territory, the
visible-but-invalid role the task designed for it.

The substrate also signposts the design space. It wires `module.rotary_emb` onto every attention module
so a policy that re-rotates kept keys has the rotary table on hand — position-remapping compressors are
an anticipated move, which I file away for the first rung that reorders the cache. And `n_kept =
int(k_len * (1 - ratio))` has a corner the anchor sidesteps: on a short prompt `int(k_len * 0.2)` is
tiny, so a compressor reserving a few sink slots can find the whole budget consumed by sinks — the
contract's arithmetic is now in front of me for when it starts to bite.

One workload I can predict the *shape* of a priori. LongBench v2 is four-way multiple choice scored by
exact accuracy, so its random-guess floor is ~25, and a 3B instruct model on a deliberately hard
long-context multiple-choice set is unlikely to clear that floor by much. Narrow headroom means v2 is a
*low-discrimination* workload: every method clusters a few points above chance and its accuracy barely
moves as I change the retention rule. So the discriminating signal should come from the open-ended
workloads — hotpotqa's span F1, passage retrieval's needle accuracy, repobench's code similarity, and
gsm8k's exact-answer accuracy — where dropping the wrong tokens has real room to fall. I will read v2
as a floor-consistency check, not as evidence a method is safe.

So how do I land "keep everything" on this hook? The cleanest expression of no compression is not a
score that happens to keep all tokens — the harness still computes `n_kept = int(k_len * (1 - ratio))`
from its own force-overridden ratio and would top-K me down to ~20% regardless, so an all-ones score
keeps an arbitrary top-K, not everything. The way to actually keep everything is to tell the harness not
to compress: the plan field `disable_compression = True` makes it skip `score_tokens` and `select_cache`
entirely, leave the cache untouched, and record retained = 1.0. This is the one plan field besides the
budget the harness genuinely enforces — observable from the post-selection cache state — so it is the
actual no-compression switch. `score_tokens` returns `None` (also read as keep-everything) and
`select_cache` returns keys and values unchanged. I set `rerotate_selected_keys = False` because nothing
is re-positioned: the cache keeps its original token positions, the prefill RoPE phases stay correct,
and decode continues from the true sequence length.

What I watch, and what it sets up for step 2. The per-workload accuracy spread is the target each later
rung is measured against — I expect passage retrieval and hotpotqa (answer can sit anywhere in a long
passage) to be where full context helps most, and gsm8k to show whatever this 3B model does with its
chain-of-thought intact. That gsm8k number is the headroom figure I care about most: a compressor that
shreds the reasoning prefix craters there first, so if the ceiling is already low the workload is
uninformative, and if it is healthy gsm8k becomes the sharpest test of whether a positional rule can
survive on reasoning. The runtime row is the slowest per-token decode (largest cache to stream), so it
calibrates the runtime term; I expect the ordering to track total decode work — generated tokens times
cache size — more than prompt length alone, which is a genuine prediction with no prior to anchor it. And
the accuracy row is the reference: at step 2 I drop to a sink-plus-window rule, and the per-workload gap
to this row is exactly the quality I pay for the 5x reduction — which workloads lose the most tells me
whether a *positional* rule is even the right shape or whether I need a content-aware score. The floor
is installed; the first compression I try is the cheapest, most robust one — keep the sinks and the
recent window — so its failures, read against this anchor, point at what a smarter score must recover.
The full scaffold module for this rung is in the answer.
