The anchor told me exactly what I am buying and what I am about to spend. Full attention posted
37.1 on hotpotqa, 62.4 on passage retrieval, 47.6 on repobench, 29.0 on LongBench v2, and 31.8 on
gsm8k — that last one matters, because gsm8k is the workload where the model writes a long
chain-of-thought and the answer depends on the whole reasoning prefix surviving. Those five numbers are
the ceiling, and they came at retained = 1.0, which is four times over the 0.25 budget tolerance, so as
a budgeted submission the anchor is worthless: the penalty zeroes most of what that accuracy earned.
The task is now sharp and unforgiving — get as close to those five accuracies as I can while throwing
away ~80% of every layer's prefill cache.

Before I pick a rule, the anchor's runtime row is worth reading, because it confirms the prefill/decode
split I reasoned about at step 1 and tells me where a compressor's runtime can even move. The five times
are 347, 323, 646, 2589, and 7440 seconds for hotpotqa, passage retrieval, repobench, LongBench v2, and
gsm8k. gsm8k is 21x hotpotqa — it generates the longest chains over the whole test set, so it is
decode-dominated, exactly the regime where a 5x-smaller cache should buy wall-clock time because decode
is bandwidth-bound on reloading K and V. LongBench v2 at 2589 is the next heaviest, and for the opposite
reason: its documents are the longest, so its cost is a quadratic prefill the hook fires *after* and
cannot touch — I should expect a compressor to barely move it. Hotpotqa and passage retrieval at ~330
are cheap and short-decode, and there the per-layer cost of *doing* the compression — a top-K, and for
the rule I am about to pick a gather and a re-rotation, paid on every layer up front — might not be
recouped over a handful of decode steps, so I would not be surprised if those two come in *at or
slightly above* the anchor rather than below. So "runtime near the anchor" is not one expectation; it is
a shape: gsm8k down, LongBench v2 flat because it is prefill-bound, the two cheap retrieval/QA workloads
flat-to-slightly-up because the compression overhead outweighs their thin decode savings. That is the
falsifiable runtime prediction, and I will check it against the row rather than assert compression is
"faster."

So the first compression I reach for should be the cheapest, most robust thing that could possibly work,
run it, and read its gap to the anchor per workload — because *which* workloads collapse will tell me
whether a positional retention rule is even the right shape. What is the cheapest rule that could
plausibly keep quality? Keep the most recent tokens. Recent context is what predicts the next token;
cache the last `L`, evict the oldest as the window rolls. Constant memory, constant latency, and it
needs nothing but position — no attention read, no query, which is exactly what this hook allows (it
hands me keys, values, hidden states, and no attention matrix). This is the obvious move and it ought to
just work. But I know the failure mode of a bare recent-only window, and it is not a gentle forgetting
of old context — it is a cliff. The known perplexity curve for sliding windows is flat and healthy while
the cache still holds everything, and then the instant the window fills and the *first* token of the
sequence gets evicted, perplexity spikes and the model starts producing nonsense. That is bizarre on its
face: the evicted token is thousands of positions away from what is being predicted, the window is full
of perfectly good recent tokens, and yet dropping the ancient first token detonates the model. So this
is not a "lost useful long-range information" failure. The collapse is tied specifically and abruptly to
evicting the *initial* tokens. Something about the first few positions is load-bearing in a way that has
nothing to do with their being recent or relevant. Before I commit to any rule, I have to understand
that, because the diagnosis decides what to keep.

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

Now the cliff makes mechanical sense, and I want to put arithmetic on it rather than leave it as a
picture, because the size of the effect is what decides how hard I must protect the sinks. Write the
attended positions as a sink set `S` plus the ordinary rest `R`. Before eviction a non-sink token's
weight is `exp(x_j)/(sum_S exp + sum_R exp)`. Suppose, concretely, the sinks collectively contribute an
exp-mass of about 50 and the ordinary tokens contribute about 10, so the denominator is 60 and the sinks
hold `50/60 ≈ 83%` of the mass — squarely in the "over half" regime the attention maps show. A
particular non-sink token `j` with `exp(x_j) = 2` then has weight `2/60 ≈ 0.033`. Now evict `S`. The
same logit divides by only `sum_R exp = 10`, so its weight jumps to `2/10 = 0.2` — a 6x amplification,
and *every* surviving weight is multiplied by the same `60/10 = 6`, even though no remaining logit
improved by a hair. Two things break at once: the attention output, a convex combination
`sum_i a_i v_i`, loses the entire sink-value contribution that was 83% of the mix, and it rescales the
surviving 17% up to a full distribution. Downstream layers receive an output vector whose direction and
magnitude are nothing the model ever sees in normal inference. That is the cliff, and it is made of
arithmetic: the sinks were holding the softmax denominator, and the window threw the denominator away
while keeping perfectly good recent tokens. So the sinks are valuable not for content but for the mass
they absorb and the denominator they hold — and the rescue writes itself: do not evict them. Pin a small
fixed set of the very first tokens, permanently, and slide the budget's remaining capacity over the most
recent tokens, dropping the middle. The cache becomes two pieces — sinks plus a recent window — with no
attention read and no query dependence. This is the StreamingLLM skeleton, and it fits this hook
perfectly because the hook forbids exactly the things this rule does not need.

Let me put the budget arithmetic on the two pieces, because the sizes tell me whether "keep the recent
window" is even a meaningful amount of context. The harness computes `n_kept = int(k_len * (1 - ratio))`
with the ratio force-overridden to 0.8, so I keep `~20%` of each layer's tokens. On a 30k-token
LongBench prompt that is `n_kept = 6000`: four sinks plus a recent window of `5996` tokens. So the
recent window is not a peephole, it is six thousand tokens of the most recent context — which is why the
rule holds up wherever the answer lives near the end or is locally inferable. What it discards is the
`~24000` tokens in the middle, blindly. That framing already tells me the shape of the failure: the
budget is generous about *recency* and merciless about the *middle*, so any workload whose answer sits in
the middle of a long context is exactly where this rule will bleed.

It is worth being precise about *why* recency is the right default and where the reasoning stops holding,
because that boundary is what the next rung will have to cross. Keeping the most recent tokens is the
right prior for *language modeling*: the next-token distribution is dominated by local context, adjacent
tokens carry the most mutual information with what comes next, so if all I cared about were perplexity a
recent window plus sinks is close to optimal. But the workloads are not perplexity. A retrieval or QA
task asks the model to attend, at the *end* of the prompt where the question sits, back to a specific
span that could be anywhere — and that span's relevance is mutual information with the *query*, not with
local next-token prediction. Recency and task-relevance are different quantities, and the recent-window
rule only tracks the first. So the rule is exactly right for the part of the model's job that is
autoregressive continuation and exactly blind to the part that is content retrieval, which is why I
expect it to hold on locally-inferable outputs and fail on needle tasks. That split is not a flaw I can
patch with a bigger window; it is the signal that a *positional* rule has reached its ceiling and the
missing ingredient is a per-token notion of relevance.

Before I commit to the recent window, let me at least walk the one alternative fixed geometry that looks
like it might dodge the middle problem: keep tokens on a stride across the *whole* context — every k-th
token — instead of a contiguous recent block. A strided keep covers the middle uniformly, so it might
catch a middle needle the recent window drops. But it pays for that three times over. It breaks the
recency block, and recency is the one thing I *know* the model needs for continuation, so I would be
trading a guaranteed win for a speculative one. It still has to pin the sinks separately or it re-detonates
the denominator cliff, so it is not simpler. And worst, once I re-rotate the kept tokens to contiguous
cache positions — which I must, or the gaps blow up RoPE — a stride places tokens that were originally
hundreds of positions apart at *adjacent* cache slots, telling the model that two semantically distant
tokens are neighbors and corrupting the local structure it reads through relative position. So strided
keep is not a free improvement; it is a different fixed pattern with the same fundamental defect — it has
no notion of *which* tokens matter, only a different guess about *where* they sit. The honest conclusion
is that no fixed geometry solves the middle; the recent window is the best fixed geometry because it at
least nails recency, and the real fix is content-awareness, which is the next rung's problem, not this
one's. So I keep the recent window and let its middle-failure be the clean, interpretable signal.

How many initial tokens do I pin? First note that the sink count is not a budget question. On a
6000-token budget, four sinks is `0.07%` of the kept set and sixteen sinks is `0.27%` — negligible either
way, so pinning a few more sinks costs the window essentially nothing. It is a "how many slots does the
denominator mechanism actually need" question. If the model had been trained with one fixed token always
at position zero, one sink slot would suffice. But this is an off-the-shelf instruct model with no such
dedicated sink, so it spreads the sink role across the first *few* positions, and I would expect one or
two pinned tokens to be too few — the unpinned sink positions still get evicted and the denominator still
partly collapses — while beyond a handful the extra slots pin tokens that were not carrying sink mass, so
they buy nothing. Sweeping the sink count has exactly that shape: one or two does not recover quality,
four does, and beyond four is marginal. So `n_sink = 4` is the knee, and the recent window is whatever
the budget leaves after the four sinks. That is the entire retention decision: it is *positional and
static*, determined before I look at a single attention value, which is the whole appeal — it is the
cheapest rule that respects the diagnosis.

There is one subtlety the naive "keep first four plus last L" rule gets silently wrong, and on this
model it matters a lot, because Qwen uses RoPE. When I keep tokens `[0,1,2,3]` and a recent window at
original positions like `[..., m-2, m-1, m]` while the middle is evicted, the intuitive thing is to leave
each kept token labeled with its original text position. But RoPE makes the query-key inner product
depend only on the *relative* offset: with `q_m = R_m W_q x_m` and `k_n = R_n W_k x_n` and `R` orthogonal
and additive in the index, `q_m^T k_n = x_m^T W_q^T R_{n-m} W_k x_n`, a function of `n - m` alone. So
what the model sees is the *gaps* between kept tokens.

Let me make the failure concrete with a small trace, because the size of the gap is the whole point. Say
the prompt has 100 tokens and I keep 7: four sinks `[0,1,2,3]` and a recent window of three at original
positions `[97,98,99]`. If I leave the kept tokens at their original labels, the relative distance the
model reads between the last sink (original position 3) and the first window token (original position 97)
is `97 - 3 = 94`. On a 30k prompt with the same 20% budget that same
sink-to-window gap is not 94 but roughly `24000` — it grows without bound as the stream advances, and
those huge relative distances are exactly the regime RoPE was never trained on, where it degrades. I
would be re-importing the very length-extrapolation failure I am trying to dodge, through the back door,
and on long LongBench prompts that is precisely where it would hurt.

The fix is to assign positions by index *within the cache*, not by original text position. If the cache
holds `n_kept` tokens, treat them as occupying contiguous positions `0, 1, ..., n_kept-1` regardless of
where they came from in the text. In the seven-token example the kept tokens get new positions
`[0,1,2,3,4,5,6]`, so the sink-to-window gap that was 94 becomes `4 - 3 = 1`, and the largest relative
distance the model ever sees is `n_kept - 1 = 6`, comfortably inside the trained range no matter how far
the stream has gone — the physical gap is simply erased from the position bookkeeping. Concretely, for
RoPE I re-rotate each kept key by the *difference* between its new and old position. The keys in the
cache are stored already rotated to their original positions: a key for original position `p` is
`R_p (W_k x)`, and I want it to behave as if it sat at new cache position `p'`. Let me check that the
re-rotation composes the way I need, on the 2D block for a single frequency `theta`, where `R_p` is the
planar rotation by angle `p*theta`, `[[cos p*theta, -sin p*theta],[sin p*theta, cos p*theta]]`.
Left-multiplying the stored `R_p (W_k x)` by `R_{p'-p}` rotates by `(p'-p)*theta` on top of `p*theta`,
and planar rotations add their angles, so the total is `(p'-p+p)*theta = p'*theta`, which is exactly
`R_{p'}`. Verified: `R_{p'} = R_{p'-p} R_p`, so left-multiplying the stored, already-rotated key by
`R_{p'-p}` — rotate it by `delta = p' - p` — lands it at the key the model would have produced had the
token sat at `p'`, and I never need the un-rotated key. In the efficient RoPE form,
`R_m x = x * cos(m*theta) + rotate_half(x) * sin(m*theta)`, where `rotate_half` swaps the two halves of
the feature vector with a sign flip; to re-rotate by `delta` I form the angles `delta*theta` from the
module's rotary inv-freq table, take their cos and sin, and apply the same formula. The new positions in
cache order are `0..n_kept-1`, so after I sort the kept indices into chronological order, `delta` for the
j-th kept token is `j` minus its original index — generally negative, because I am moving each kept key
from a larger original index down to a smaller contiguous one, which is a backward rotation to close the
gaps (the deltas in the seven-token example are `[0,0,0,0,-93,-93,-93]`). The values carry no position,
so they are just gathered at the kept indices, no re-rotation. This re-rotation is the piece that makes
streaming work past the pretraining length; skipping it would quietly re-introduce extrapolation
failure, and given the anchor's numbers the workloads most exposed would be the long retrieval ones.

Now I map this onto the three-method hook, and here I have to respect what *this* harness exposes rather
than a generic implementation. `retention_plan` hands `score_tokens` the sink count and the budget — and
I read `compression_ratio` straight from `cache_meta`, because the harness force-overrides it at the call
site anyway, so the policy cannot lie about the budget; declaring it is provenance, the harness enforces
its own value. `score_tokens` produces the static positional mask the harness wants: the harness keeps
the top-`n_kept` by score, so I score 1 everywhere and 0 on the middle block to prune. If the layer's
cache has `k_len` tokens, the retained count is `int(k_len * (1 - ratio))`, so the number to prune is
`n_pruned = k_len - int(k_len * (1 - ratio))`, and the zero slice starts immediately after the sinks:
`[n_sink : n_sink + n_pruned]`. That leaves exactly `n_kept` one-scored positions — the first `n_sink`
tokens and the most recent `n_kept - n_sink` — and an assertion guards that the cache has more tokens
than sinks. One implementation detail particular to this harness: the re-rotation builds its `gather_idx`
from `keys.shape[-1]` (the actual head dimension of the stored key tensor) rather than a
`module.head_dim` attribute, so the gather width always matches the cache, and the rotary table comes
from `module.rotary_emb.inv_freq` which the harness has wired onto every attention module. `select_cache`
then does the top-k, sorts the kept indices chronologically (required for the contiguous re-positioning
to make sense), re-rotates the kept keys by their deltas, and gathers the values — and because this
policy always re-rotates, `rerotate_selected_keys = True` is a class attribute so the harness advances
the decode positions from the re-rotated, contiguous cache length rather than the original sequence
length. The full scaffold module is in the answer.

Let me close on what I expect against the anchor, falsifiably. Retained should land at ~0.20 across all
five workloads (the harness enforces it), so I clear the budget penalty the anchor failed, and runtime
should follow the shape I read off the anchor's row — gsm8k down, LongBench v2 flat, the cheap retrieval/QA
pair flat-to-slightly-up. The interesting bets are the accuracy gaps. The sinks-plus-window rule keeps
the *recent* context and the denominator anchor but blindly discards the middle, so I expect it to hold
up where the answer lives near the end or is locally inferable, and to bleed where the answer can sit
anywhere in a long passage. So: passage retrieval and hotpotqa should drop below the anchor's 62.4 and
37.1 by a visible margin — those are needle-in-a-haystack workloads and a positional rule drops the
needle when it is in the middle of the six-thousand-token hole. LongBench v2 I expect to roughly hold
near 29, both because its head-tail truncation and multiple-choice format are forgiving of a positional
keep and because, as I argued at step 1, its ~25 chance floor leaves so little headroom that nothing
moves it much. The number I am most worried about is gsm8k: the model's reasoning prefix is the thing
being compressed, and if streaming shreds the chain-of-thought, gsm8k should fall hard from the anchor's
31.8 — possibly to near zero, which would be the loud signal that a purely positional rule is the wrong
shape and that I need a *content-aware* score that can decide which middle tokens to keep. The mechanism
is that a reasoning trace scatters its load-bearing intermediate steps through the *middle* of the
sequence, not the last 6000 tokens, so the recent window keeps the wrong end of the chain. If instead
gsm8k holds, my whole motivation for moving past streaming weakens. That gsm8k gap, read against the 31.8
anchor, is the falsifiable hinge that decides step 3.
