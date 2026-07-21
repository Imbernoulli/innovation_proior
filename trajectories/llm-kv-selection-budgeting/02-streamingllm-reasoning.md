The anchor told me exactly what I am buying and what I am about to spend. Full attention posted
37.1 on hotpotqa, 62.4 on passage retrieval, 47.6 on repobench, 29.0 on LongBench v2, and 31.8 on
gsm8k — that last one matters, because gsm8k is the workload where the model writes a long
chain-of-thought and the answer depends on the whole reasoning prefix surviving. Those five numbers are
the ceiling, and they came at retained = 1.0, four times over the 0.25 budget tolerance, so as a
budgeted submission the anchor is worthless: the penalty zeroes most of what that accuracy earned. The
task is now sharp — get as close to those five accuracies as I can while throwing away ~80% of every
layer's prefill cache.

The anchor's runtime row also tells me where a compressor's runtime can even move. The five times are
347, 323, 646, 2589, and 7440 seconds. gsm8k is 21x hotpotqa — it generates the longest chains over the
whole test set, so it is decode-dominated, exactly where a 5x-smaller cache should buy wall-clock time
because decode is bandwidth-bound on reloading K and V. LongBench v2 at 2589 is the next heaviest for
the opposite reason: its documents are the longest, so its cost is a quadratic prefill the hook fires
*after* and cannot touch — I expect a compressor to barely move it. Hotpotqa and passage retrieval at
~330 are cheap and short-decode, and there the per-layer cost of *doing* the compression — a top-K,
gather, and re-rotation, paid on every layer up front — might not be recouped over a handful of decode
steps, so those two could come in at or slightly *above* the anchor. So the runtime prediction is a
shape, not one number: gsm8k down, v2 flat, the two cheap workloads flat-to-slightly-up.

So the first compression I reach for is the cheapest, most robust thing that could plausibly keep
quality: keep the most recent tokens. Recent context predicts the next token; cache the last `L`, evict
the oldest as the window rolls. Constant memory, constant latency, and it needs nothing but position —
no attention read, no query, exactly what this hook allows. This is the obvious move and it ought to
just work. But I know the failure mode of a bare recent-only window, and it is not a gentle forgetting
of old context — it is a cliff. The perplexity curve for sliding windows is flat and healthy while the
cache still holds everything, then the instant the window fills and the *first* token of the sequence
gets evicted, perplexity spikes and the model produces nonsense. That is bizarre: the evicted token is
thousands of positions away from what is being predicted, the window is full of good recent tokens, and
yet dropping the ancient first token detonates the model. So this is not a "lost useful long-range
information" failure — the collapse is tied specifically to evicting the *initial* tokens. Something
about the first few positions is load-bearing in a way that has nothing to do with recency or relevance,
and the diagnosis decides what to keep.

Look at what the model does with its attention. Beyond the bottom couple of layers, nearly every head
dumps a huge fraction of its attention mass onto the first few positions — on long sequences the mass
from the last token back to the very first often exceeds *half* the total. Is that because the content
of those tokens matters, or their position? The discriminating test is to replace the first few tokens
with a meaningless filler: if content drove the attention, garbage should not attract it — but it still
does, and reintroducing fillers after an eviction restores quality as well as the originals would. So it
is absolute position, not semantics: the model uses whatever sits in the first few slots as a fixed
destination for attention.

Why would it do that? Within a head the weights are `exp(x_i)/sum_j exp(x_j)` and they sum to one —
there is no "none of the above," no abstain. So a query whose information is essentially self-contained,
with no strong match anywhere, must still produce a distribution summing to one; it dumps the leftover
mass somewhere, and the cheapest place is a token whose value it can mix in harmlessly. The model
manufactures a few such dumping grounds — attention sinks, tokens that collect large attention precisely
because they are *not* semantically important, just to soak up the surplus softmax forces it to
allocate. And why the *initial* tokens? Causal visibility: a query at position `t` attends only to
positions `<= t`, so the only positions visible to *every* later query are the earliest ones. A common
dumping ground every position can reach can only be the first few tokens. So initial-token sinks are
forced by softmax's sum-to-one plus causal masking, not an accident.

The cliff is then arithmetic, and the size of the effect decides how hard I must protect the sinks.
Write the attended positions as a sink set `S` plus the rest `R`. Say the sinks contribute exp-mass ~50
and the ordinary tokens ~10, so the denominator is 60 and the sinks hold `50/60 ≈ 83%` of the mass —
the "over half" regime the attention maps show. A non-sink token `j` with `exp(x_j) = 2` then has weight
`2/60 ≈ 0.033`. Evict `S` and the same logit divides by only `sum_R exp = 10`, so its weight jumps to
`2/10 = 0.2` — a 6x amplification, and *every* surviving weight is multiplied by `60/10 = 6` even though
no remaining logit improved. Two things break at once: the attention output loses the 83% of its mix
that was sink-value, and rescales the surviving 17% up to a full distribution. Downstream layers get an
output vector whose direction and magnitude are nothing the model sees in normal inference. The window
threw away the softmax denominator while keeping perfectly good recent tokens — so the rescue writes
itself: do not evict the sinks. Pin a small fixed set of the very first tokens permanently and slide the
budget's remaining capacity over the most recent tokens, dropping the middle. Two pieces — sinks plus a
recent window — with no attention read and no query dependence. This is StreamingLLM, and it fits this
hook because the hook forbids exactly the things this rule does not need.

The budget arithmetic tells me whether "keep the recent window" is even a meaningful amount of context.
The harness keeps `n_kept = int(k_len * (1 - ratio))` with ratio 0.8, so ~20% per layer. On a 30k-token
prompt that is `n_kept = 6000`: four sinks plus a recent window of 5996 tokens — not a peephole, six
thousand tokens of the most recent context, which is why the rule holds wherever the answer lives near
the end or is locally inferable. What it discards is the ~24000 tokens in the middle, blindly. So the
budget is generous about recency and merciless about the middle: any workload whose answer sits in the
middle of a long context is where this rule bleeds.

That boundary is worth being precise about, because it is what the next rung must cross. Keeping recent
tokens is the right prior for *language modeling* — adjacent tokens carry the most mutual information
with the next token, so for perplexity a recent window plus sinks is near-optimal. But the workloads are
not perplexity. A retrieval or QA task asks the model, at the *end* of the prompt where the question
sits, to attend back to a specific span that could be anywhere — and that span's relevance is mutual
information with the *query*, not with local next-token prediction. The recent-window rule tracks recency
and is blind to task-relevance. That is not a flaw a bigger window patches; it is the signal that a
*positional* rule has a ceiling and the missing ingredient is a per-token notion of relevance.

I do consider one alternative fixed geometry — keep tokens on a stride across the whole context rather
than a contiguous recent block — since it would cover the middle uniformly. But it breaks the recency
block (the one thing I *know* the model needs), still has to pin the sinks separately, and once I
re-rotate the kept tokens to contiguous positions it places originally-distant tokens at adjacent cache
slots, telling the model that semantically distant tokens are neighbors. No fixed geometry solves the
middle; the recent window is the best fixed geometry because it at least nails recency, and the real fix
is content-awareness — the next rung's problem. So I keep the recent window and let its middle-failure be
the clean signal.

How many initial tokens to pin is not a budget question — on a 6000-token budget four sinks is 0.07% and
sixteen is 0.27%, negligible either way. It is a "how many slots does the denominator mechanism need"
question. A model trained with one fixed token at position zero would need one sink slot; this is an
off-the-shelf instruct model with no dedicated sink, so it spreads the sink role across the first *few*
positions. One or two pinned tokens leave unpinned sink positions to be evicted and the denominator
partly collapses; beyond a handful the extra slots pin tokens carrying no sink mass. So the sweep has a
knee at `n_sink = 4`, and the recent window is whatever the budget leaves. The retention decision is
positional and static, fixed before I read a single attention value — the whole appeal.

There is one subtlety the naive "keep first four plus last L" gets wrong, and on this RoPE model it
matters. RoPE makes the query-key inner product depend only on the *relative* offset: with `q_m = R_m
W_q x_m`, `k_n = R_n W_k x_n`, and `R` orthogonal and additive in the index, `q_m^T k_n = x_m^T W_q^T
R_{n-m} W_k x_n`, a function of `n - m` alone. So what the model sees is the *gaps* between kept tokens.
Keep 7 tokens from a 100-token prompt — sinks `[0,1,2,3]` and a window at `[97,98,99]` — and at their
original labels the sink-to-window gap is `97 - 3 = 94`; on a 30k prompt at 20% budget that same gap is
~24000 and grows without bound as the stream advances. Those huge relative distances are exactly the
regime RoPE was never trained on, so I would re-import the length-extrapolation failure I am trying to
dodge, and on long retrieval prompts that is precisely where it hurts.

The fix is to assign positions by index *within the cache*: `n_kept` kept tokens occupy contiguous
positions `0..n_kept-1` regardless of where they came from. In the seven-token example the sink-to-window
gap of 94 becomes `4 - 3 = 1`, and the largest relative distance is `n_kept - 1 = 6`, comfortably inside
the trained range. Concretely I re-rotate each kept key by the difference between its new and old
position. A stored key for original position `p` is `R_p (W_k x)`; since planar rotations add angles,
`R_{p'-p} R_p = R_{p'}`, so left-multiplying the stored, already-rotated key by `R_{delta}` with
`delta = p' - p` lands it at the key the model would have produced had the token sat at `p'`, and I never
need the un-rotated key. In the efficient form `R_m x = x*cos(m*theta) + rotate_half(x)*sin(m*theta)`,
I form the angles `delta*theta` from the module's rotary inv-freq table and apply the same formula. New
positions in cache order are `0..n_kept-1`, so after sorting the kept indices chronologically, `delta`
for the j-th kept token is `j` minus its original index — generally negative, since I move each kept key
from a larger original index down to a smaller contiguous one (in the seven-token example the deltas are
`[0,0,0,0,-93,-93,-93]`). Values carry no position, so they are just gathered.

Mapping onto the three-method hook, respecting what this harness exposes: `retention_plan` hands
`score_tokens` the sink count and budget, and I read `compression_ratio` from `cache_meta` because the
harness force-overrides it anyway, so declaring it is provenance and the harness enforces its own value.
`score_tokens` produces the static positional mask the harness wants — it keeps the top-`n_kept` by
score, so I score 1 everywhere and 0 on the middle block `[n_sink : n_sink + n_pruned]` with `n_pruned =
k_len - int(k_len*(1-ratio))`, leaving exactly the first `n_sink` and the most recent `n_kept - n_sink`
one-scored, guarded by an assertion that the cache has more tokens than sinks. One harness detail: the
re-rotation builds its `gather_idx` from `keys.shape[-1]` (the actual stored head dimension) rather than
a `module.head_dim` attribute, and the rotary table comes from `module.rotary_emb.inv_freq` which the
harness has wired onto every module. `select_cache` does the top-k, sorts the kept indices
chronologically (required for the contiguous re-positioning to make sense), re-rotates the kept keys by
their deltas, and gathers the values — and because this policy always re-rotates,
`rerotate_selected_keys = True` so the harness advances decode positions from the re-rotated cache length
rather than the original sequence length. The full scaffold module is in the answer.

What I expect against the anchor. Retained lands at ~0.20 (the harness enforces it), so I clear the
budget penalty, and runtime should follow the shape I read off the anchor's row. The accuracy bets are
the interesting part: the sinks-plus-window rule keeps recent context and the denominator anchor but
discards the middle blindly, so passage retrieval and hotpotqa should drop below 62.4 and 37.1 by a
visible margin — needle tasks where a positional rule drops the needle when it is in the middle of the
six-thousand-token hole. LongBench v2 should roughly hold near 29, both from its forgiving format and
its chance-floored headroom. The number I am most worried about is gsm8k: a reasoning trace scatters its
load-bearing intermediate steps through the *middle* of the sequence, not the last 6000 tokens, so the
recent window keeps the wrong end of the chain and gsm8k could fall hard from 31.8, possibly to near
zero. That would be the loud signal that a purely positional rule is the wrong shape and I need a
content-aware score that decides *which* middle tokens to keep. If instead gsm8k holds, my motivation
for moving past streaming weakens — so that gsm8k gap against 31.8 is what decides step 3.
