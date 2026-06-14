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
not transferable from a paper: LongBench F1 depends on the harness's normalization, repobench's
code-similarity depends on which line of the generation it compares, gsm8k depends on the boxed-answer
extraction and numeric normalization in this scaffold. So the only trustworthy reference is one
produced *by this harness*, and that is what running the uncompressed policy here gives me. Without it
I would be guessing at headroom.

Let me be exact about what "keep everything" is, because it is not a no-op — it is a deliberate policy
on the same hook as every other rung. The harness runs a standard full-attention prefill: the model
sees the whole prompt at once, fills the per-layer KV cache with `k_1..k_p, v_1..v_p`, and the
per-layer forward hook fires with that filled cache. At that moment the controller gets to decide the
layer's retention. The reason the full cache is the principled reference, and not merely a lazy
default, comes straight from the causal structure of the decoder. In a masked self-attention layer the
key `k_j` and value `v_j` of position `j` are computed from token `j` and the layers beneath it, and
the mask means information flows only forward in position — no later position `i > j` can reach back and
change `k_j` or `v_j`. So once position `j` has been processed, its `(k_j, v_j)` is frozen for the rest
of the sequence. Keeping the entire frozen set means the attention read at every decode step is over
the genuine, complete history: the decoded sequence follows the same mathematical computation as
re-running the model on the full prefix at each step. No token is dropped, nothing is approximated, so
whatever the model can do with full context, it does here. That is the ceiling, and it is exact by
construction — there is no scoring rule clever enough to *exceed* it on quality, only rules that try to
approach it with a fraction of the cache.

Now the cost, because the ceiling is precisely the thing I am being asked to give up. The full cache's
size is `b * h * (k_dim + v_dim) * L * n` — linear in sequence length `n` and depth `L`. On a 30k-token
LongBench prompt across the model's layers, that is the memory wall the task exists to attack. And the
per-step decode cost is dominated by streaming that growing cache from memory: the memory-to-arithmetic
ratio of incremental decode rises toward 1 as `n` approaches the model width, so a long-context decode
is bandwidth-bound on reloading `K, V`. The honest ledger of exactness is a state and a per-step
bandwidth that both grow with the sequence. Compression trades a slice of the quality ceiling for a
bounded cache; this rung is the point on that trade where the slice given up is zero and the cache is
unbounded — the opposite extreme from where any deployable controller must live.

I should also be explicit about what makes *this* the right anchor rather than some "cheap but lossless"
trick, because it is tempting to think there is a free lunch. There are two superficially attractive
alternatives that are not lossless. Windowing — cap attention to the last `L` positions — bounds memory
and bandwidth, but a query can no longer see anything outside the window, so any dependency longer than
`L` is silently dropped; the model's output diverges from its full-context output. Memory compression —
pool past positions into fewer key/value slots — has the same character: the query attends over fewer,
summarized slots instead of the true set, so the output again diverges. Both are legitimate
speed/quality trades, and indeed the later rungs *are* exactly such trades. But neither is the model's
actual full-prefix computation, so neither can serve as the quality reference. The full KV cache, by
contrast, retains every key and every value and evicts nothing, so the attention read at every step is
over the genuine complete history and the decoded sequence follows the same mathematical computation as
the naive full-prefix re-run — only without the recomputation. Exactness, full context, no eviction:
that is precisely the property that qualifies it as the ceiling, and disqualifies it as a budgeted
submission.

So how do I land "keep everything" on this specific hook? The contract has three methods —
`retention_plan`, `score_tokens`, `select_cache`. The cleanest way to express no compression is not to
return a score that happens to keep all tokens (the harness still computes `n_kept = int(k_len * (1 -
ratio))` and would force a top-K down to ~20%), but to *tell the harness not to compress at all*. The
plan field `disable_compression` is the lever: when it is `True`, the harness skips `score_tokens` and
`select_cache` entirely, leaves the cache untouched, and records the retained fraction as exactly 1.0.
This is the one plan field besides the budget that the harness genuinely enforces — it is observable
from the post-selection cache state — so setting it is not advisory bookkeeping, it is the actual
no-compression switch. `score_tokens` then returns `None` (the harness reads `None` as "keep
everything," another path to the same retained = 1.0) and `select_cache` returns the keys and values
unchanged. I set `rerotate_selected_keys = False` because nothing is being re-positioned: the cache
keeps its original token positions, so the RoPE phases the prefill computed are already correct and the
decode positions continue from the true sequence length. There is no re-rotation, no gather, no top-K —
the policy's whole content is the declaration that this layer is not to be compressed.

Let me be clear-eyed about where this lands on the scoreboard, because that is the entire point of
running it. The leaderboard combines, per workload, accuracy at weight 6, runtime at weight 2, and
cache reduction at weight 2, then takes a geometric mean across the five workloads — and it applies a
soft upper-bound penalty the moment `mean_retained_fraction` exceeds 0.25. The full cache retains 1.0,
which is four times over that tolerance, so the penalty fires hard. This is deliberate in the task's
design: the no-compression anchor is kept *visible* as a reference row but is explicitly not a valid
budgeted submission. So I expect this rung to post the highest raw accuracy of anything I will run —
the ceiling — and simultaneously the worst *budgeted* score, because the reduction term is at its floor
(it kept everything) and the budget constraint zeroes most of what the accuracy earned. It is the
weakest submission precisely because it refuses the task's central constraint, while being the strongest
on the one axis (quality) that the constraint exists to protect. That tension is the thing the next
rungs have to resolve: get as close as possible to this accuracy while living at ~20% retained.

What I will watch in the numbers, and what they have to set up for step 2. First, the per-workload
accuracy spread: I expect the retrieval and QA workloads (passage retrieval, hotpotqa) to be where full
context helps most, because the answer can sit anywhere in a long passage, and the reasoning workload
(gsm8k) to show whatever this 3B instruct model can do with its full chain-of-thought intact —
critically, gsm8k accuracy under the full cache is the headroom number, because a compressor that
shreds the reasoning prefix will crater there first. Second, the runtime row: the full cache is the
*slowest* per-token decode (largest cache to stream), so it also calibrates the runtime term that
compressed methods will improve on. Third, and most important for the ladder: this row is the accuracy
target. When I drop to a sink-plus-window rule at step 2, the gap between its accuracy and this row's
accuracy — per workload — is exactly the quality I paid for the 5x cache reduction, and which workloads
lose the most will tell me whether a *positional* retention rule is even the right shape, or whether I
need a content-aware score. The floor is installed; the only way is to compress, and the first
compression I will try is the cheapest, most robust one — keep the sinks and the recent window — so
that its failures, read against this anchor, point at what a smarter score has to recover. The full
scaffold module for this rung is in the answer.
