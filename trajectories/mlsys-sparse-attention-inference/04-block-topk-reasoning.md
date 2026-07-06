StreamingLLM did exactly what I predicted, and the pattern across the three rungs now isolates the one
remaining problem cleanly. Qasper F1 came up from BigBird's `0.0871` to `0.1103` — `0.1103 / 0.1406 = 0.78`,
about 78% of the oracle recovered, within a hair of the ceiling — and the densities all dropped to a flat
`0.2344`, comfortably under the `0.25 + 0.02` ceiling on every task, where BigBird had been over budget on
two of three and cleared one run by a thousandth. So reclaiming the random budget into a clean sink+window
did what I hoped on the budget side and on Qasper: the static floor is now honest and predictable. But two
numbers refuse to move the way I want. First, NIAH stayed pinned at `0.2` — chance — identical to BigBird.
Two completely different static geometries, the same no-needle floor. That is the decisive control result I
set up: the NIAH failure is not about *which* static pattern I pick or how I size its window; it is about
staticness itself. The sink columns cover the first few tokens, the window covers the recent tokens, and a
needle planted in the middle of an 8K haystack lives in neither, so no fixed mask — sink+window,
global+window+random, any of them — can route to it.

The second number is a quieter clue and I do not want to skip it: MultiFieldQA-EN actually *slipped*, from
BigBird's `0.2298` to `0.213`. That is small, but its direction is informative. Going from a random-diluted
window to a pure, five-times-wider local window *lost* `0.017` of F1 on MultiFieldQA — which means BigBird's
random blocks, for all that they cratered NIAH and wasted budget, happened to catch some distributed
evidence on that task that sits *beyond* StreamingLLM's tail window and that a purely local pattern therefore
drops. So MultiFieldQA has real mid-document evidence outside the last `~1020` tokens, and neither static
pattern reaches it well: BigBird sprayed random edges and hit a little of it by luck; StreamingLLM refused
the spray and missed it entirely. That is a second, independent argument for the same move NIAH demands — a
selection that can *choose* the relevant far blocks rather than sampling them blind or ignoring them. The
residual gap to the oracle, on both the NIAH cliff and the MultiFieldQA distributed-evidence, is entirely a
*routing* gap, and routing means the kept set has to depend on what the query is asking. Time to make the
selection content-adaptive.

So the problem restated under this task's constraints: for each query, out of all preceding keys,
manufacture a small subset to attend to — at most `0.25` of the causal pairs — but now chosen *from the
current query* rather than from a fixed template, cheaply, with no retraining and nothing but the
`q, k, v` of this forward. The loop is `use_cache=False`, so there is no decode cache to mine, and this is
exactly why H2O, SnapKV, and Quest are off the table: their importance signal is keyed to a mutable KV
cache accumulated over decode steps, and here every forward is a fresh full parallel pass with no such
accumulation, so their scores would be computed against a cache that does not exist. The reason adaptivity
should work at all is the concentration premise the oracle established: softmax attention is empirically
sparse, for a given query a small fraction of keys carry almost all the mass, so most of the `O(N²)` work is
on near-zero weights and if I attend only to the keys that matter I lose little. The catch — and the whole
difficulty — is that *which* keys matter is query-dependent and moves, so no static set works (the two
static rungs just proved that), and the obvious way to find them, computing `q·k` for all pairs and keeping
the largest, *is* the dense score matrix I am trying to avoid. I need to find the important keys without
first paying for the full `(N, N)` logits.

The escape is to judge importance at a coarser unit than the individual key, and there are two independent
reasons that is legitimate. First, where attention actually lands is *spatially clustered* — the nonzero
entries come in contiguous runs, neighboring keys share an importance level — so a whole contiguous block of
keys rises and falls together, and one score per block loses little. Second, the hardware: contiguous block
reads are what GPUs are fast at, and the fast attention kernels already tile K/V into blocks; selecting
scattered individual tokens would force non-contiguous gathers and waste the sparsity. Both point the same
way: chop the keys into contiguous blocks of size `BLOCK = 64` (matching the harness block size, 128 blocks
at 8K) and decide importance per block.

How to score a block cheaply, with a frozen model and no learned projection? Summarize each block by a
single representative vector and score the query against that. The block's keys are spatially coherent —
that was the premise — so they point in similar directions and their *mean* is a low-variance summary that a
key inside the block is close to. So the representative is the mean-pooled key of the block, and
`q · mean(k_block)` approximates the aggregate `q·k` over the block. Parameter-free, no training, valid
precisely because the block clusters. And to keep the selection score matrix from being `O(N · n_blocks)` —
still quadratic in `N` — I pool the queries the same way: mean-pool each query block, so the importance
matrix is `(n_blocks × n_blocks)`. Let me price that: `128 × 128 = 16,384` block-pair scores per head versus
the dense `N² = 8192² ≈ 67` million logits — a factor of `BLOCK² = 4096` fewer, which is exactly the
`64 × 64` I collapsed each block-pair into. So the selection step never materializes the `(N, N)` matrix;
it works on a matrix four thousand times smaller. Pooling queries is defensible by the same clustering
argument on the query axis: adjacent queries ask for similar things, so giving a query block one shared
selection is consistent with the structure and is exactly what a block-shaped kernel wants. The proxy is
`s[i, j] = mean_q(block i) · mean_k(block j) · scale`, a small matrix that never touches the dense logits.

I have to be honest about what the mean-pooling costs, because it is the one place this proxy can silently
fail, and NIAH is precisely the case that stresses it. The softmax cares about the *maximum* logit in a
block — a single very relevant key dominates — but the block mean is an *average*, so a block that holds one
sharp needle among 63 filler keys gets its needle diluted by a factor of `BLOCK`. Let me work out whether
the needle still survives. Write the needle's raw logit as `L = q · k_needle` and let the filler dot
products have spread `σ` around zero. The needle block's mean score is `(1/64)(L + Σ_{63} filler) ≈ L/64`
plus noise of order `σ/8`, since a mean of `64` independent `σ`-spread terms has standard deviation
`σ/√64 = σ/8`. A non-needle block's mean score is pure noise of that same order, `σ/8`. So the needle block
outranks the field — wins a top-K slot — only when `L/64 ≳ σ/8`, i.e., when `L ≳ 8σ`, and that threshold is
`√BLOCK · σ` in general. This is the crux: pooling turns retrieval into a *thresholded* test. A needle whose
raw logit is many `σ` above the filler — a strong, sharp match — clears the `8σ` bar even after the `/64`
dilution and gets selected, so its block is kept and the model can retrieve it. A needle whose logit is only
moderately above the field — enough to win the per-key dense softmax but not by `8σ` — gets averaged below
the inter-block noise and its block is missed. So I predict NIAH *rises above the `0.2` floor* for the first
time on the ladder, because the sharp needles are now routable, but I do *not* expect it to reach the
oracle's `1.0`, because the mean-pool dilution filters out the moderate ones. The `√BLOCK` threshold also
tells me why `BLOCK = 64` and not `256`: a larger block raises the retrieval bar to `16σ` and would drop
more needles, while a smaller block costs more selection compute and less hardware-friendly contiguity, so
64 is the compromise the harness block size already suggests.

Before I commit to the mean-pool block proxy I should confirm the alternatives are genuinely worse, not just
different. Per-token top-K — score every key and keep the largest per query — is the ideal, but it needs
either the full `(N, N)` logits (the dense matrix I am avoiding) or a cache-accumulated importance
(unavailable), so it is out on both counts. A learned block scorer — a small projection trained to predict
block relevance — would be more accurate than a raw mean, but there is no training in this task, so it is
out. A hashing or random-projection scheme (LSH-style) could bucket keys by direction and skip the pooling,
but it adds machinery and hyperparameters for a gain over mean-pooling that is not obvious on clustered
blocks where the mean is already a good summary. The mean-pooled block proxy is the one that is
parameter-free, cache-free, avoids the dense matrix, and is justified by the very clustering that makes
block selection legitimate. So it is not a default; it is what survives ruling the others out.

With block scores, the selection rule is top-K: for each query block keep a fixed number of high-scoring key
blocks. Two things have to be handled, and one I only see by thinking about a query block near the start.
Causality first: this is a causal LLM, query block `i` may only see key block `j ≤ i`, so I mask the future
block scores to `−∞` before ranking, and AND the kept set with the causal block mask after, and then inside
the diagonal block enforce the strict token-level triangle `key_pos ≤ query_pos`, because a coarse "keep
block `i`" would otherwise let an early query in the block peek at a later key in the same block — a token
from its own future. Two levels of causal masking — blockwise for selection, token-wise for the final
attention. Second, the diagonal block: local context is almost always the most relevant, and it must always
be kept, but a top-K by the *approximate* proxy gives no guarantee the current block wins a slot — for an
early query block the diagonal might score lower than a content-similar far block under the noisy mean. So I
force-include the diagonal unconditionally: mask the diagonal out of the ranking (set its score `−∞` so
top-K never picks it), take the top `K−1` *off-diagonal* blocks, and append the diagonal by hand. This is
the role StreamingLLM's recent window played — local context always kept — but it collapses into "always
keep your own block," and the rest of the budget goes to query-chosen far blocks instead of a fixed recent
band. That is the precise upgrade over rung 3: the window was a *fixed* local keep; here the local block is
kept *and* the remaining budget routes adaptively, which is what the static rungs could not do and what both
the NIAH cliff and the MultiFieldQA slip need.

This makes the comparison against StreamingLLM a clean controlled experiment, which is worth naming because
it is what makes the NIAH result interpretable rather than confounded. StreamingLLM kept `1024` tokens per
query, which is `16` contiguous blocks. This rung keeps `K = 17` blocks per query — the diagonal plus `16`
adaptive ones. So the *per-query block budget is essentially identical*: about sixteen blocks each way. The
only thing that differs is the *allocation policy* — StreamingLLM places its sixteen blocks in a fixed
recent band, this rung places one on the diagonal and lets the query choose the other sixteen from anywhere
in the causal past. Because the budgets match, any NIAH improvement cannot be attributed to spending more of
the matrix; it can only be attributed to *where* the blocks are placed. So this rung isolates the effect of
adaptivity from the effect of budget, and that is exactly the variable the two static rungs left uncontrolled.

Now the keep count `K` must come from the budget, accounting for causality, because under a causal mask
different query blocks have different numbers of admissible blocks. Index query blocks `i = 0..n−1`; block
`i` has `i+1` causal candidates and keeps `min(K, i+1)`. Total kept block-pairs
`= Σ min(K, i+1) = K(K−1)/2 + (n−K+1)K = K(2n−K+1)/2`. I do not want to trust that closed form blindly, so
let me check it on a tiny case I can enumerate: `n = 3, K = 2` gives `min(2,1)+min(2,2)+min(2,3) = 1+2+2 =
5`, and the formula gives `2·(6−2+1)/2 = 2·5/2 = 5` — agree. `K = 1` gives `1+1+1 = 3 = n`, formula `1·6/2 =
3` — agree, each block keeps only itself. `K = 3` gives `1+2+3 = 6 = n(n+1)/2`, full causal — agree. The
formula is right. The denominator is that full count `n(n+1)/2`, so block-level density is
`K(2n−K+1)/(n(n+1))`. Set equal to the budget `ρ` and solve the quadratic `K² − (2n+1)K + ρ·n(n+1) = 0`; the
kept-pair function is concave and rising over `0 ≤ K ≤ n`, so I take the smaller feasible root
`K = ((2n+1) − √((2n+1)² − 4ρn(n+1)))/2` and floor it to stay at or under budget. Plug in `n = 128`,
`ρ = 0.25`: discriminant `257² − 4·0.25·128·129 = 66049 − 16512 = 49537`, `√ ≈ 222.6`,
`K = (257 − 222.6)/2 ≈ 17`. So 17 of 128 blocks per query block including the diagonal; check the realized
block density `17·240/(128·129) ≈ 0.247`, just under budget. `kk = K − 1 = 16` off-diagonal top-K plus the
diagonal, `K` clamped `≥ 1`, `kk` clamped to `[0, n−1]`. The non-causal fallback would be `K ≈ round(ρ·n)`,
but `is_causal` is always true. One more limit check on the quadratic, because a budget-to-`K` map that does
not recover dense at full budget is a map I should not trust: at `ρ = 1` the discriminant is
`257² − 4·1·128·129 = 66049 − 66048 = 1`, so `K = (257 − 1)/2 = 128 = n`, every block kept, full causal
attention. So the solve interpolates correctly between `K = 1` (keep only the diagonal, the sparsest legal
pattern) at `ρ → 0` and `K = n` (dense) at `ρ = 1`, and `K = 17` at `ρ = 0.25` is a point on that curve
rather than a tuned constant. This is a genuine advantage over BigBird's `0.88`-discounted sizing, which by
construction under-spends and cannot reach dense even at `ρ = 1`; here the budget is spent exactly, adaptively.

I should be honest that this `K` controls the *block-pair* density, while the reported `last_density` is the
*token-level* mask after the strict token triangle trims the half-filled diagonal block and any final
padding block — so the algebra and the realization disagree at the boundary, exactly as in the two static
rungs. I trust the algebra only to *choose* `K`; I *measure* `last_density` from the realized token mask,
which comes in at or below the block-level `0.247` because the diagonal trim only ever removes pairs, so I
expect the reported density to sit just under `0.25`, in the same budget-honest regime StreamingLLM reached.
Shape mechanics: `N` may not be a multiple of `BLOCK`, so pad `q, k` up to `Npad = ceil(N/BLOCK)·BLOCK` with
zeros so the reshape and `mean(dim=3)` are exact; the pad keys make a final padding block, but I trim the
token keep-mask back to `[:N, :N]` and the strict causal triangle kills any padded influence. Block scoring
in float32 (inputs are fp16/bf16 and I am about to `masked_fill` with `−∞` and softmax, far more stable
upcast), output cast back to `q.dtype`.

One GQA note for this task specifically: the kernel-level NSA design sums block importance *across the heads
in a KV group* so the whole group selects the same blocks, or the loaded KV is the union of each head's
picks and the memory sparsity collapses. But the harness has already replicated GQA before my module — I see
12 heads on both Q and K/V, an effectively multi-head view — so I score and select per replicated head,
which is correct in this setting; the group-reduction is only needed against a real GQA cache, which I do not
have here. This is the faithful inference re-expression, not the trainable kernel, and the final attention is
again the masked-softmax over the full logits — gathering the selected contiguous blocks into SRAM is what a
Triton kernel would do for the real speedup, but the masked form computes the identical output and no-Triton
is the task constraint. It is worth being precise about which cost this rung actually reduces, so I do not
oversell it. The *selection* is genuinely cheap: `n_blocks² · D ≈ 16,384 · 128` block-pair arithmetic,
four thousand times smaller than the dense logit count. But the *final attention*, in this masked-softmax
realization, still forms the full `(N, N)` logits and masks them, so on this harness the wall-clock is not
reduced — what I buy is faithful *behavior* under the density budget, identical to the two static rungs. In
a real deployment the selected `K` contiguous blocks would be gathered and the attention run only over them,
turning the `O(N²·D)` attention into `O(K·BLOCK·N·D)`; that is the speedup the density budget is a proxy
for, and it is why the whole ladder measures density rather than latency. The `last_density` I report is the
honest stand-in for that would-be speedup.

There is a clean way to state why this should work at all, and it closes the loop back to the oracle. The
oracle nailed NIAH because the complete causal graph contains a *direct edge* from every query to the
needle, a one-hop route that survives the softmax. The two static rungs failed because their fixed masks
never contain that specific edge unless the needle happens to land in the window or the anchors. What block
selection does is *manufacture that edge on demand*: when the end query asks for the needle, the block score
`mean_q · mean_k` lights up the needle's block, top-K keeps it, and the direct query→needle route is
restored — the exact edge dense had and static lacked, put back for the one pair that needs it, at the cost
of the `√BLOCK` dilution that decides whether the score lights up strongly enough. Query-pooling does not
spoil this for NIAH, because the retrieving query block at the end of the prompt is homogeneous — its tokens
are all the same question — so `mean_q ≈ q_question` is barely diluted, and the whole loss is on the key
side, which the SNR argument already accounted for.

Now the falsifiable expectations against the prior rungs' measured numbers, which is the bar this rung
clears. NIAH is the one that matters: both static rungs sat at `0.2` because they could not route to the
needle, and this rung *can* — the query block whose tokens are asking for the needle should score the
needle's key block highly through `mean_q · mean_k` and select it into the top-K, regardless of where in the
haystack it sits, *as long as the needle's raw logit clears the `√BLOCK · σ ≈ 8σ` dilution threshold*. So I
expect NIAH to break above `0.2` for the first time on the ladder — the single clearest test that
content-adaptive selection is doing what the static rungs could not — while the mean-pool dilution keeps it
well short of the oracle's `1.0`, which is the honest signature of a block proxy rather than a per-key one.
On the QA tasks, the diagonal-force-include gives me the same local floor StreamingLLM had, and the adaptive
top-K adds the query-relevant far spans instead of a fixed window, so I expect Qasper and MultiFieldQA to
land at least at StreamingLLM's level (`0.1103` / `0.213`) and plausibly a touch above on MultiFieldQA in
particular, since that is the task whose distributed evidence the static window demonstrably dropped and
adaptive routing is built to recover — though I hold that loosely, because the same mean-pool dilution that
caps NIAH also makes the far-block selection coarser than a guaranteed contiguous window, and if the pooling
imprecision outweighs the routing gain the QA numbers could hold flat or even dip slightly rather than rise.
The QA columns will tell me which effect dominates. And the densities should sit just under `0.25`, because
`K` is solved from the budget quadratic and measured from the trimmed mask. If NIAH rises while
Qasper/MultiFieldQA hold near StreamingLLM's level and density stays under budget, that is the confirmation
that the routing gap — the only thing separating the static rungs from the oracle's NIAH `1.0` — is what
content-adaptive block selection closes. The distilled module, the literal scaffold fill with the
mean-pooled block scoring, the forced diagonal, and the budget quadratic, is in the answer.
