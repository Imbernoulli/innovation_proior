StreamingLLM did exactly what I predicted, and the pattern across the three rungs now isolates the one
remaining problem cleanly. Qasper F1 came up from BigBird's `0.0871` to `0.1103` — past the oracle's
`0.1406` reach to within a hair, about 78% recovered — and MultiFieldQA-EN came up from `0.2298` to
`0.213`... actually it slipped slightly there, but it stayed in the same band, and crucially the
densities all dropped to a flat `0.2344`, comfortably under the `0.25 + 0.02` ceiling on every task,
where BigBird had been over budget on two of three. So reclaiming the random budget into a clean
sink+window did what I hoped on the budget side and roughly on Qasper, and the static floor is now
honest. But NIAH stayed pinned at `0.2` — chance — identical to BigBird. Two completely different static
patterns, same NIAH. That is the decisive signal: the NIAH failure is not about *which* static pattern I
pick or how I size its window; it is about staticness itself. The sink columns cover the first few
tokens, the window covers the recent tokens, and a needle planted in the middle of an 8K haystack lives
in neither, so no fixed mask — sink+window, global+window+random, any of them — can route to it. The
oracle hit NIAH `1.0` because every query has a direct edge to the needle; both static rungs sit at `0.2`
because no query has an edge to a position it did not anticipate. The residual gap to the oracle is
entirely a *routing* gap, and routing means the kept set has to depend on what the query is asking. Time
to make the selection content-adaptive.

So the problem restated under this task's constraints: for each query, out of all preceding keys,
manufacture a small subset to attend to — at most `0.25` of the causal pairs — but now chosen *from the
current query* rather than from a fixed template, cheaply, with no retraining and nothing but the
`q, k, v` of this forward (the loop is `use_cache=False`, so there is no decode cache to mine, which is
exactly why H2O/SnapKV/Quest are off the table — their importance signal is keyed to a mutable cache that
does not exist here). The reason adaptivity should work at all is that softmax attention is empirically
sparse: for a given query a small fraction of keys carry almost all the mass. So most of the `O(N²)` work
is on near-zero weights, and if I attend only to the keys that matter I lose little. The catch — and the
whole difficulty — is *which* keys matter is query-dependent and moves, so no static set works (the two
static rungs just proved that), and the obvious way to find them, computing `q·k` for all pairs and
keeping the largest, *is* the dense score matrix I am trying to avoid. I need to find the important keys
without first paying for the full `(N, N)` logits.

The escape is to judge importance at a coarser unit than the individual key, and there are two
independent reasons that is legitimate. First, where attention actually lands is *spatially clustered* —
the nonzero entries come in contiguous runs, neighboring keys share an importance level — so a whole
contiguous block of keys rises and falls together, and one score per block loses little. Second, the
hardware: contiguous block reads are what GPUs are fast at, and the fast attention kernels already tile
K/V into blocks; selecting scattered individual tokens would force non-contiguous gathers and waste the
sparsity. Both point the same way: chop the keys into contiguous blocks of size `BLOCK = 64` (matching the
harness block size, 128 blocks at 8K) and decide importance per block.

How to score a block cheaply, with a frozen model and no learned projection? Summarize each block by a
single representative vector and score the query against that. The block's keys are spatially coherent —
that was the premise — so they point in similar directions and their *mean* is a low-variance summary
that a key inside the block is close to. So the representative is the mean-pooled key of the block, and
`q · mean(k_block)` approximates the aggregate `q·k` over the block. Parameter-free, no training, valid
precisely because the block clusters. And to keep the selection score matrix from being `O(N · n_blocks)`
— still quadratic in `N` — I pool the queries the same way: mean-pool each query block, so the importance
matrix is `(n_blocks × n_blocks)`, a factor `BLOCK²` cheaper than dense. Pooling queries is defensible by
the same clustering argument on the query axis: adjacent queries ask for similar things, so giving a
query block one shared selection is consistent with the structure and is exactly what a block-shaped
kernel wants. So the proxy is `s[i, j] = mean_q(block i) · mean_k(block j) · scale`, a small matrix that
never materializes the dense `(N, N)` logits.

With block scores, the selection rule is top-K: for each query block keep a fixed number of high-scoring
key blocks. Two things have to be handled, and one of them I only see by thinking about a query block near
the start. Causality first: this is a causal LLM, query block `i` may only see key block `j ≤ i`, so I
mask the future block scores to `−∞` before ranking, and AND the kept set with the causal block mask
after, and then inside the diagonal block enforce the strict token-level triangle `key_pos ≤ query_pos`,
because a coarse "keep block `i`" would otherwise let an early query in the block peek at a later key in
the same block. Two levels of causal masking — blockwise for selection, token-wise for the final
attention. Second, the diagonal block: local context is almost always the most relevant, and it must
always be kept, but a top-K by the *approximate* proxy gives no guarantee the current block wins a slot —
for an early query block the diagonal might score lower than a content-similar far block. So I force-
include the diagonal unconditionally: mask the diagonal out of the ranking (set its score `−∞` so it is
never picked by top-K), take the top `K−1` *off-diagonal* blocks, and append the diagonal by hand. This
is the role StreamingLLM's recent window played — local context always kept — but now it collapses into
"always keep your own block," and the rest of the budget goes to query-chosen far blocks instead of a
fixed recent band. That is the precise upgrade over rung 3: the window was a *fixed* local keep; here the
local block is kept *and* the remaining budget routes adaptively, which is what the static rungs could
not do and what NIAH needs.

Now the keep count `K` must come from the budget, accounting for causality, because under a causal mask
different query blocks have different numbers of admissible blocks. Index query blocks `i = 0..n−1`; block
`i` has `i+1` causal candidates and keeps `min(K, i+1)`. Total kept block-pairs `= Σ min(K, i+1) =
K(K−1)/2 + (n−K+1)K = K(2n−K+1)/2`. Sanity: `K=1` gives `n` (each block keeps itself), `K=n` gives
`n(n+1)/2` (full causal). The denominator is that full count `n(n+1)/2`, so block-level density `=
K(2n−K+1)/(n(n+1))`. Set equal to the budget `ρ` and solve the quadratic `K² − (2n+1)K + ρ·n(n+1) = 0`;
the kept-pair function is concave and rising over `0 ≤ K ≤ n`, so I take the smaller feasible root
`K = ((2n+1) − √((2n+1)² − 4ρn(n+1)))/2` and floor it to stay at or under budget. Plug in `n=128`,
`ρ=0.25`: discriminant `257² − 4·0.25·128·129 = 66049 − 16512 = 49537`, `√ ≈ 222.6`, `K = (257−222.6)/2 ≈
17`. So 17 of 128 blocks per query block including the diagonal; check the realized density `17·240/(128·
129) ≈ 0.247`, just under budget. `kk = K−1 = 16` off-diagonal top-K plus the diagonal, `K` clamped ≥ 1,
`kk` clamped to `[0, n−1]`. The non-causal fallback would be `K ≈ round(ρ·n)`, but `is_causal` is always
true.

I should be honest that this `K` controls the *block-pair* density, while the reported `last_density` is
the *token-level* mask after the strict token triangle trims the half-filled diagonal block and any
final padding block — so the algebra and the realization disagree at the boundary. I trust the algebra
only to *choose* `K`; I *measure* `last_density` from the realized token mask, which comes in at or below
the block-level 0.247, honoring the contract. Shape mechanics: `N` may not be a multiple of `BLOCK`, so
pad `q, k` up to `Npad = ceil(N/BLOCK)·BLOCK` with zeros so the reshape and `mean(dim=3)` are exact; the
pad keys make a final padding block, but I trim the token keep-mask back to `[:N, :N]` and the strict
causal triangle kills any padded influence. Block scoring in float32 (inputs are fp16/bf16 and I am about
to `masked_fill` with `−∞` and softmax, far more stable upcast), output cast back to `q.dtype`.

One GQA note for this task specifically: the kernel-level NSA design sums block importance *across the
heads in a KV group* so the whole group selects the same blocks, or the loaded KV is the union of each
head's picks and the memory sparsity collapses. But the harness has already replicated GQA before my
module — I see 12 heads on both Q and K/V, an effectively multi-head view — so I score and select per
replicated head, which is correct in this setting; the group-reduction is only needed against a real GQA
cache, which I do not have here. This is the faithful inference re-expression, not the trainable kernel.

The recipe end to end: pad to a block multiple; mean-pool `q` and `k` per block; score block pairs in
fp32; causal-mask the block scores; mask the diagonal out of the ranking; solve the budget quadratic for
`K`; take `kk = K−1` off-diagonal top-K plus the diagonal; AND with the causal block mask; expand to a
token keep-mask, trim to `N`, AND with the strict token triangle; report the measured density; and run a
masked-softmax over the full logits (gathering the selected contiguous blocks into SRAM is what a Triton
kernel would do for the real speedup, but the masked form computes the identical output, and no-Triton is
the task constraint).

Now the falsifiable expectations against the prior rungs' measured numbers, which is the bar this rung
clears. NIAH is the one that matters: both static rungs sat at `0.2` because they could not route to the
needle, and this rung *can* — the query block whose tokens are asking for the needle should score the
needle's key block highly through `mean_q · mean_k` and select it into the top-K, regardless of where in
the haystack it sits. So I expect NIAH to *break above* `0.2` for the first time on the ladder — the
single clearest test that content-adaptive selection is doing what the static rungs could not. On the QA
tasks, the diagonal-force-include gives me the same local floor StreamingLLM had, and the adaptive top-K
adds the query-relevant far spans instead of a fixed window, so I expect Qasper and MultiFieldQA to land
at least at StreamingLLM's level (`0.1103` / `0.213`) and plausibly a touch above, since the routed far
blocks are chosen for the question. And the densities should sit just under `0.25`, the same budget-honest
regime StreamingLLM reached, because `K` is solved from the budget quadratic and the density is measured
from the trimmed mask. If NIAH rises while Qasper/MultiFieldQA hold and density stays under budget, that
is the confirmation that the routing gap — the only thing separating the static rungs from the oracle's
NIAH `1.0` — is what content-adaptive block selection closes. The distilled module, the literal scaffold
fill with the mean-pooled block scoring, the forced diagonal, and the budget quadratic, is in the answer.
