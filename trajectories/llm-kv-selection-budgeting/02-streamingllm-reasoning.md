The anchor told me exactly what I am buying and what I am about to spend. Full attention posted
37.1 on hotpotqa, 62.4 on passage retrieval, 47.6 on repobench, 29.0 on LongBench v2, and 31.8 on
gsm8k — that last one matters, because gsm8k is the workload where the model writes a long
chain-of-thought and the answer depends on the whole reasoning prefix surviving. Those five numbers are
the ceiling, and they came at retained = 1.0, which is four times over the 0.25 budget tolerance, so as
a budgeted submission the anchor is worthless: the penalty zeroes most of what that accuracy earned.
The task is now sharp and unforgiving — get as close to those five accuracies as I can while throwing
away ~80% of every layer's prefill cache. The anchor also calibrated the runtime row (it is the slowest
decode, largest cache to stream), so any compressor that actually shrinks the cache should come in at or
below those times. So the first compression I reach for should be the cheapest, most robust thing that
could possibly work, run it, and read its gap to the anchor per workload — because *which* workloads
collapse will tell me whether a positional retention rule is even the right shape.

What is the cheapest rule that could plausibly keep quality? Keep the most recent tokens. Recent context
is what predicts the next token; cache the last `L`, evict the oldest as the window rolls. Constant
memory, constant latency, and it needs nothing but position — no attention read, no query, which is
exactly what this hook allows (it hands me keys, values, hidden states, and no attention matrix). This
is the obvious move and it ought to just work. But I know the failure mode of a bare recent-only window,
and it is not a gentle forgetting of old context — it is a cliff. The known perplexity curve for sliding
windows is flat and healthy while the cache still holds everything, and then the instant the window
fills and the *first* token of the sequence gets evicted, perplexity spikes and the model starts
producing nonsense. That is bizarre on its face: the evicted token is thousands of positions away from
what is being predicted, the window is full of perfectly good recent tokens, and yet dropping the
ancient first token detonates the model. So this is not a "lost useful long-range information" failure.
The collapse is tied specifically and abruptly to evicting the *initial* tokens. Something about the
first few positions is load-bearing in a way that has nothing to do with their being recent or relevant.
Before I commit to any rule, I have to understand that, because the diagnosis decides what to keep.

Look at what the model actually does with its attention. Across layers and heads, beyond the bottom
couple of layers, nearly every head dumps a huge fraction of its attention mass onto the first few token
positions — on long sequences the mass from the last token back to the very first often exceeds *half*
the total, in most layers. Half, onto whatever happened to start the text. Is that because the content
of those tokens matters, or because of their position? The two have completely different fixes. The
clean discriminating test is to replace the first few tokens with a meaningless filler token: if content
drove the attention, garbage should not attract it — but it still does, and reintroducing the fillers
after an eviction restores quality just as well as the originals would. So it is not semantics, it is
absolute position. The model has learned to use whatever sits in the first few slots as a fixed
destination for attention.

Why would it do that? Stare at the softmax. Within a head the weights are `exp(x_i)/sum_j exp(x_j)`, and
the defining property is that they sum to one — there is no "none of the above," no abstain. So a query
whose information is essentially self-contained, with no strong match anywhere, still has to produce a
distribution summing to one; it must dump the leftover mass *somewhere*, and the cheapest place is a
token whose value it can mix in harmlessly — one it can always find, that does not corrupt the residual
stream. The model manufactures a few such dumping grounds. Call them attention sinks: tokens that
collect large attention precisely because they are not semantically important, just to soak up the
surplus softmax forces it to allocate. And why the *initial* tokens as the sinks? Causal visibility
decides it — a query at position `t` can only attend to positions `<= t`, so the only positions visible
to *every* later query are the earliest ones. If the model wants a common dumping ground every position
can reach, the first few tokens are the only candidates. So initial-token sinks are forced by softmax's
sum-to-one plus causal masking, not an accident.

Now the cliff makes mechanical sense, and it tells me precisely what the recent-only window did wrong.
Write the attended positions as a sink set `S` plus the ordinary rest `R`. Before eviction a non-sink
token's weight is `exp(x_j)/(sum_{S} exp + sum_{R} exp)`; the sinks dominate that denominator. Evict
`S`, and the same logit now divides by only `sum_{R} exp` — every remaining weight is multiplied by a
large factor even though no remaining logit improved, and the attention output loses the sink-value
contribution and over-amplifies the ordinary values. Downstream layers receive a distributional shape
they never see in normal inference. The window had perfectly good recent tokens; it threw away the
denominator anchor. That is the cliff. So the sinks are valuable not for content but for the attention
mass they absorb and the denominator they hold — and the rescue writes itself: do not evict them. Pin a
small fixed set of the very first tokens, permanently, and slide the budget's remaining capacity over
the most recent tokens, dropping the middle. The cache becomes two pieces — sinks plus a recent window —
with no attention read and no query dependence. This is the StreamingLLM skeleton, and it fits this hook
perfectly because the hook forbids exactly the things this rule does not need.

How many initial tokens do I pin? If the model had been trained with one fixed token always at position
zero, one sink slot would suffice. But this is an off-the-shelf instruct model with no such dedicated
sink, so it spreads the sink role across the first *few* positions, and I would expect one or two pinned
tokens to be too few and the budget to saturate after a handful. Sweeping the sink count has exactly
that shape — one or two does not recover quality, four does, and beyond four is marginal. So `n_sink = 4`
is the right default here, and the recent window is whatever the budget leaves after the four sinks.
That is the entire retention decision: it is *positional and static*, determined before I look at a
single attention value, which is the whole appeal — it is the cheapest rule that respects the diagnosis.

There is one subtlety the naive "keep first four plus last L" rule gets silently wrong, and on this
model it matters a lot, because Qwen uses RoPE. When I keep tokens `[0,1,2,3]` and a recent window at
original positions like `[..., m-2, m-1, m]` while the middle is evicted, the intuitive thing is to leave
each kept token labeled with its original text position. But RoPE makes the query-key inner product
depend only on the *relative* offset: with `q_m = R_m W_q x_m` and `k_n = R_n W_k x_n` and `R` orthogonal
and additive in the index, `q_m^T k_n = x_m^T W_q^T R_{n-m} W_k x_n`, a function of `n - m` alone. So
what the model sees is the *gaps* between kept tokens. If I keep original positions `[0,1,2,3, ...,
m-1, m]` with a hole where the middle used to be, the relative distance between the pinned sinks and the
rolling window grows without bound as the stream advances — and those huge relative distances are
exactly the regime RoPE was never trained on, where it degrades. I would be re-importing the very
length-extrapolation failure I am trying to dodge, through the back door, and on long LongBench prompts
that is precisely where it would hurt.

The fix is to assign positions by index *within the cache*, not by original text position. If the cache
holds `n_kept` tokens, treat them as occupying contiguous positions `0, 1, ..., n_kept-1` regardless of
where they came from in the text. Then every relative distance the model sees is small and contiguous,
inside the trained range, no matter how far the stream has gone — the physical gap is simply erased from
the position bookkeeping. Concretely, for RoPE I re-rotate each kept key by the *difference* between its
new and old position. The keys in the cache are stored already rotated to their original positions: a
key for original position `p` is `R_p (W_k x)`, and I want it to behave as if it sat at new cache
position `p'`. Since rotations compose, `R_{p'} = R_{p'-p} R_p`, so I left-multiply the stored,
already-rotated key by `R_{p'-p}` — rotate it by `delta = p' - p`. I never need the un-rotated key. In
the efficient RoPE form, `R_m x = x * cos(m*theta) + rotate_half(x) * sin(m*theta)`, where
`rotate_half` swaps the two halves of the feature vector with a sign flip; to re-rotate by `delta` I form
the angles `delta * theta` from the module's rotary inv-freq table, take their cos and sin, and apply
the same formula. The new positions in cache order are `0..n_kept-1`, so after I sort the kept indices
into chronological order, `delta` for the j-th kept token is `j` minus its original index — generally
negative, because I am moving each kept key from a larger original index down to a smaller contiguous
one, which is a backward rotation to close the gaps. The values carry no position, so they are just
gathered at the kept indices, no re-rotation. This re-rotation is the piece that makes streaming work
past the pretraining length; skipping it would quietly re-introduce extrapolation failure, and given the
anchor's numbers the workloads most exposed would be the long retrieval ones.

Now I map this onto the three-method hook, and here I have to respect what *this* harness exposes rather
than a generic implementation. `retention_plan` hands `score_tokens` the sink count and the budget —
and I read `compression_ratio` straight from `cache_meta`, because the harness force-overrides it at the
call site anyway, so the policy cannot lie about the budget; declaring it is provenance, the harness
enforces its own value. `score_tokens` produces the static positional mask the harness wants: the
harness keeps the top-`n_kept` by score, so I score 1 everywhere and 0 on the middle block to prune. If
the layer's cache has `k_len` tokens, the retained count is `int(k_len * (1 - ratio))`, so the number to
prune is `n_pruned = k_len - int(k_len * (1 - ratio))`, and the zero slice starts immediately after the
sinks: `[n_sink : n_sink + n_pruned]`. That leaves exactly `n_kept` one-scored positions — the first
`n_sink` tokens and the most recent `n_kept - n_sink` — and an assertion guards that the cache has more
tokens than sinks. One implementation detail particular to this harness: the re-rotation builds its
`gather_idx` from `keys.shape[-1]` (the actual head dimension of the stored key tensor) rather than a
`module.head_dim` attribute, so the gather width always matches the cache, and the rotary table comes
from `module.rotary_emb.inv_freq` which the harness has wired onto every attention module. `select_cache`
then does the top-k, sorts the kept indices chronologically (required for the contiguous re-positioning
to make sense), re-rotates the kept keys by their deltas, and gathers the values — and because this
policy always re-rotates, `rerotate_selected_keys = True` is a class attribute so the harness advances
the decode positions from the re-rotated, contiguous cache length rather than the original sequence
length. The full scaffold module is in the answer.

Let me close on what I expect against the anchor, falsifiably. Retained should land at ~0.20 across all
five workloads (the harness enforces it), so I clear the budget penalty the anchor failed, and runtime
should sit near the anchor's or a touch below since the decode cache is now a fifth the size. The
interesting bets are the accuracy gaps. The sinks-plus-window rule keeps the *recent* context and the
denominator anchor but blindly discards the middle, so I expect it to hold up where the answer lives
near the end or is locally inferable, and to bleed where the answer can sit anywhere in a long passage.
So: passage retrieval and hotpotqa should drop below the anchor's 62.4 and 37.1 by a visible margin —
those are needle-in-a-haystack workloads and a positional rule drops the needle when it is in the middle.
LongBench v2 I expect to roughly hold near 29, because its head-tail truncation and multiple-choice
format are forgiving of a positional keep. The number I am most worried about is gsm8k: the model's
reasoning prefix is the thing being compressed, and if streaming shreds the chain-of-thought, gsm8k
should fall hard from the anchor's 31.8 — possibly to near zero, which would be the loud signal that a
purely positional rule is the wrong shape and that I need a *content-aware* score that can decide which
middle tokens to keep. If instead gsm8k holds, my whole motivation for moving past streaming weakens.
That gsm8k gap, read against the 31.8 anchor, is the falsifiable hinge that decides step 3.
