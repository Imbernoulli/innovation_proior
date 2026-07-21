StreamingLLM did what I predicted, and the three rungs now isolate the remaining problem cleanly. Qasper
came up from BigBird's `0.0871` to `0.1103` — `0.1103/0.1406 = 0.78`, within a hair of the ceiling — and
the densities all dropped to a flat `0.2344`, comfortably under budget on every task where BigBird had been
over on two of three and cleared one run by a thousandth. The static floor is now honest and predictable.
But two numbers refuse to move. First, NIAH stayed pinned at `0.2`, identical to BigBird: two completely
different static geometries, the same no-needle floor. That is the decisive control result — the NIAH
failure is not about *which* static pattern I pick or how I size its window, it is about staticness itself,
because the sink columns cover the first tokens, the window covers the recent tokens, and a needle in the
middle of an 8K haystack lives in neither.

The second number is a quieter clue: MultiFieldQA-EN actually *slipped*, from `0.2298` to `0.213`. Small,
but its direction is informative — going from a random-diluted window to a pure, five-times-wider local one
*lost* `0.017` of F1, which means BigBird's random blocks, for all that they cratered NIAH and wasted
budget, happened to catch some distributed evidence *beyond* StreamingLLM's tail window that a purely local
pattern drops. So MultiFieldQA has real mid-document evidence outside the last `~1020` tokens, and neither
static pattern reaches it well: BigBird sprayed random edges and hit a little by luck, StreamingLLM refused
the spray and missed it entirely. That is a second, independent argument for the same move NIAH demands — a
selection that can *choose* the relevant far blocks rather than sampling them blind or ignoring them. The
residual gap to the oracle, on both the NIAH cliff and the MultiFieldQA distributed-evidence, is entirely a
*routing* gap: the kept set has to depend on what the query is asking.

So the problem under this task's constraints: for each query, out of all preceding keys, manufacture a
small subset to attend to — at most `0.25` of the causal pairs — chosen *from the current query* rather than
a fixed template, cheaply, with no retraining and nothing but this forward's `q, k, v`. The loop is
`use_cache=False`, which is exactly why H2O, SnapKV, and Quest are off the table: their importance signal is
keyed to a mutable KV cache accumulated over decode steps, and here every forward is a fresh full parallel
pass with no such accumulation, so their scores would run against a cache that does not exist. Adaptivity
should work because of the concentration premise the oracle established — softmax attention is empirically
sparse, a small fraction of keys carry almost all the mass — so attending only to the keys that matter loses
little. The catch is that *which* keys matter is query-dependent and moves (the two static rungs proved no
fixed set works), and the obvious way to find them, computing `q·k` for all pairs and keeping the largest,
*is* the dense score matrix I am avoiding. I need the important keys without first paying for the full
`(N, N)` logits.

The escape is to judge importance at a coarser unit than the individual key, legitimate for two independent
reasons. Where attention lands is spatially clustered — the nonzero entries come in contiguous runs,
neighboring keys share an importance level — so a whole contiguous block of keys rises and falls together
and one score per block loses little. And the hardware: contiguous block reads are what GPUs are fast at,
and selecting scattered individual tokens would force non-contiguous gathers that waste the sparsity. Both
say chop the keys into contiguous blocks of `BLOCK = 64` (128 blocks at 8K) and decide importance per
block. To score a block cheaply with a frozen model and no learned projection, summarize it by a single
representative vector: the block's keys are spatially coherent, so they point in similar directions and
their *mean* is a low-variance summary that a member key is close to, and `q · mean(k_block)` approximates
the aggregate `q·k` over the block — parameter-free, valid precisely because the block clusters. To keep the
selection matrix from being `O(N · n_blocks)`, I pool the queries the same way, so the importance matrix is
`(n_blocks × n_blocks) = 128 × 128 = 16{,}384` scores per head versus the dense `~67M` logits — `BLOCK² =
4096` times fewer, exactly the `64 × 64` each block-pair collapses. Pooling queries is defensible by the
same clustering on the query axis: adjacent queries ask for similar things, and a shared per-block selection
is what a block-shaped kernel wants. The proxy `s[i, j] = mean_q(i) · mean_k(j) · scale` never touches the
dense logits.

I have to be honest about what mean-pooling costs, because NIAH stresses it exactly. The softmax cares about
the *maximum* logit in a block — a single very relevant key dominates — but the block mean is an *average*,
so a block holding one sharp needle among 63 filler keys dilutes the needle by `BLOCK`. Whether it survives
is an SNR question. Write the needle's raw logit `L = q · k_needle` and let the filler dot products have
spread `σ` around zero. The needle block's mean score is `(1/64)(L + Σ_{63} filler) ≈ L/64` plus noise of
order `σ/8` (a mean of 64 independent `σ`-spread terms has standard deviation `σ/√64`); a non-needle block's
mean is pure noise of that same order. So the needle block outranks the field — wins a top-K slot — only
when `L/64 ≳ σ/8`, i.e. `L ≳ 8σ`, and the threshold is `√BLOCK · σ` in general. Pooling turns retrieval into
a *thresholded* test: a needle whose raw logit is many `σ` above the filler clears the `8σ` bar even after
the `/64` dilution and gets selected, while a needle only moderately above the field — enough to win the
per-key dense softmax but not by `8σ` — gets averaged below the inter-block noise and its block is missed.
So I predict NIAH *rises above the `0.2` floor* for the first time on the ladder, because the sharp needles
are now routable, but not to the oracle's `1.0`, because the dilution filters out the moderate ones. The
`√BLOCK` threshold also fixes `BLOCK = 64` over `256`: a larger block raises the bar to `16σ` and drops more
needles, a smaller one costs more selection compute and less contiguity, so 64 is the compromise the harness
block size already suggests.

The alternatives are genuinely worse, not just different. Per-token top-K is the ideal but needs either the
full `(N, N)` logits (the dense matrix I am avoiding) or a cache-accumulated importance (unavailable), out
on both counts. A learned block scorer would beat a raw mean but there is no training here. An LSH-style
hashing scheme could bucket keys by direction, but it adds machinery and hyperparameters for a gain that is
not obvious on clustered blocks where the mean is already a good summary. The mean-pooled block proxy is
what survives ruling the others out.

The selection rule is top-K per query block, with two things to handle. Causality: query block `i` may only
see `j ≤ i`, so I mask the future block scores to `−∞` before ranking, AND the kept set with the block
causal mask after, and then inside the diagonal block enforce the strict token triangle `key_pos ≤
query_pos` — two levels, blockwise for selection and token-wise for the final attention, because a coarse
"keep block `i`" would let an early query in the block peek at a later key in the same block, a token from
its own future. And the diagonal block: local context is almost always most relevant and must always be
kept, but a top-K by the *approximate* proxy gives no guarantee the current block wins a slot — for an early
query block the diagonal might score below a content-similar far block under the noisy mean. So I
force-include it: mask the diagonal out of the ranking (set its score `−∞` so top-K never picks it), take
the top `K−1` off-diagonal blocks, and append the diagonal by hand. That plays StreamingLLM's local-window
role, collapsed to "always keep your own block," while the rest of the budget routes adaptively — the
precise upgrade over rung 3, where the window was a *fixed* local keep. This also makes the comparison a
clean controlled experiment: StreamingLLM kept `1024` tokens = 16 contiguous blocks per query, and this rung
keeps `K = 17` blocks (diagonal plus 16 adaptive), so the per-query budget is essentially identical and the
*only* difference is allocation policy — a fixed recent band versus one on the diagonal plus 16 the query
chooses from anywhere in the causal past. Any NIAH improvement therefore cannot be attributed to spending
more of the matrix; it can only be attributed to *where* the blocks land, which is exactly the variable the
two static rungs left uncontrolled.

`K` comes from the budget under causality. Block `i` has `i+1` causal candidates and keeps `min(K, i+1)`,
so total kept block-pairs `Σ min(K, i+1) = K(2n−K+1)/2`. Checking on a tiny case I can enumerate,
`n = 3, K = 2`: `min(2,1)+min(2,2)+min(2,3) = 1+2+2 = 5`, and the formula gives `2(6−2+1)/2 = 5` — agree;
`K = 1` gives `n`, `K = 3` gives `n(n+1)/2`, both agree, so the closed form is right. Density is
`K(2n−K+1)/(n(n+1)) = ρ`, a quadratic `K² − (2n+1)K + ρn(n+1) = 0` whose kept-pair function is concave and
rising over `0 ≤ K ≤ n`, so I take the smaller feasible root `K = ((2n+1) − √((2n+1)² − 4ρn(n+1)))/2` and
floor it. At `n = 128, ρ = 0.25`: discriminant `257² − 4·0.25·128·129 = 66049 − 16512 = 49537`,
`√ ≈ 222.6`, `K = (257 − 222.6)/2 ≈ 17`; realized block density `17·240/(128·129) ≈ 0.247`, just under
budget, `kk = K − 1 = 16` off-diagonal top-K plus the diagonal. At `ρ = 1` the discriminant is `1` and
`K = 128 = n`, full causal attention — so the solve interpolates between keep-only-the-diagonal at `ρ → 0`
and dense at `ρ = 1`, and `K = 17` is a point on that curve rather than a tuned constant. This is a genuine
advantage over BigBird's `0.88`-discounted sizing, which by construction under-spends and cannot reach dense
even at `ρ = 1`.

`K` controls the *block-pair* density, while the reported `last_density` is the *token-level* mask after the
strict triangle trims the half-filled diagonal block and any padding block — so the algebra and the
realization disagree at the boundary, exactly as in the two static rungs. I trust the algebra only to
*choose* `K` and *measure* `last_density` from the realized token mask, which comes in at or below the
block-level `0.247` since the trim only removes pairs. Shape mechanics: `N` may not be a multiple of `BLOCK`,
so pad `q, k` up to `Npad = ceil(N/BLOCK)·BLOCK` with zeros for a clean reshape and `mean(dim=3)`, then trim
the token keep-mask back to `[:N, :N]` and let the strict causal triangle kill any padded influence; block
scoring in float32, output cast back. One GQA note: the kernel-level NSA design sums block importance across
the heads in a KV group so the whole group selects the same blocks, or the loaded KV is the union of each
head's picks and the memory sparsity collapses — but the harness already replicated GQA before my module, I
see 12 heads on both Q and K/V, so I score per replicated head, which is correct here; the group-reduction
is only needed against a real GQA cache. And the final attention is again masked-softmax over the full
logits, so on this harness the *selection* is genuinely cheap (`16{,}384 · 128` block-pair arithmetic, four
thousand times smaller than the dense logit count) but the wall-clock is not reduced — what I buy is
faithful *behavior* under the density budget. In a real deployment the selected `K` contiguous blocks would
be gathered and attention run only over them, turning `O(N²·D)` into `O(K·BLOCK·N·D)`; that is the speedup
the density budget stands in for, and why the ladder measures density rather than latency.

There is a clean way to state why this works, closing back to the oracle. The oracle nailed NIAH because the
complete causal graph contains a *direct edge* from every query to the needle, a one-hop route that survives
the softmax; the static rungs failed because their fixed masks never contain that specific edge unless the
needle lands in the window or the anchors. Block selection *manufactures that edge on demand*: when the end
query asks for the needle, the block score `mean_q · mean_k` lights up the needle's block, top-K keeps it,
and the direct query→needle route is restored — for the one pair that needs it, at the cost of the `√BLOCK`
dilution that decides whether the score lights up strongly enough. Query-pooling does not spoil this for
NIAH, because the retrieving query block at the end of the prompt is homogeneous — its tokens are all the
same question — so `mean_q ≈ q_question` is barely diluted and the whole loss is on the key side, which the
SNR argument already accounted for.

So the falsifiable expectations against the prior rungs' measured numbers. NIAH is the one that matters:
both static rungs sat at `0.2` because they could not route to the needle, and this rung *can*, so I expect
it to break above `0.2` for the first time on the ladder — the clearest test that content-adaptive selection
does what the static rungs could not — while the mean-pool dilution keeps it well short of the oracle's
`1.0`, the honest signature of a block proxy rather than a per-key one. On the QA tasks the diagonal
force-include gives the same local floor StreamingLLM had, and the adaptive top-K adds query-relevant far
spans instead of a fixed window, so I expect Qasper and MultiFieldQA at least at StreamingLLM's `0.1103` /
`0.213` and plausibly a touch above on MultiFieldQA in particular, the task whose distributed evidence the
static window demonstrably dropped — though I hold that loosely, because the same mean-pool dilution that
caps NIAH makes the far-block selection coarser than a guaranteed contiguous window, so if the pooling
imprecision outweighs the routing gain the QA numbers could hold flat or even dip slightly. And the
densities should sit just under `0.25`, `K` solved from the quadratic and measured from the trimmed mask. If
NIAH rises while Qasper/MultiFieldQA hold near StreamingLLM's level and density stays under budget, that
confirms the routing gap — the only thing separating the static rungs from the oracle's NIAH `1.0` — is what
content-adaptive block selection closes. The distilled module with the mean-pooled block scoring, the forced
diagonal, and the budget quadratic is in the answer.
