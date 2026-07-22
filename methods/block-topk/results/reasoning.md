Let me start from what is actually killing me at inference. I have a frozen causal LLM and a long context, and for every query token `q_t` the attention has to score `q_t` against all `t` preceding keys and softmax-weight the values. That is `O(t)` per query, `O(t^2)` over the sequence, and at 64k context the breakdowns I trust say attention is something like 70-80% of decode latency. So the quadratic is exactly where it hurts, right where long context is the point. I am not allowed to retrain or touch the weights — I get to replace the attention forward and nothing else — and I have a hard knob: a density budget, a cap on the fraction of `(q, k)` pairs that may get nonzero attention. Default 0.25. The harness replays my module on the full prefix every forward, so whatever I decide has to be computable from the `q` and `k` sitting in front of me right now, not from some cache I accumulate across steps. Out of the full set of preceding keys, manufacture for each query a small subset to attend to, cheaply, and keep the quality.

The reason I think this is even possible: softmax attention is empirically sparse. For a given query a small fraction of the keys carry almost all the mass — people have measured that in a 128k attention matrix the top few thousand columns recall the overwhelming majority of the total attention. So most of the `O(t^2)` work is spent on near-zero weights. If I could attend to just the keys that matter, I would pay far less and lose almost nothing. The catch is *which* keys matter. That set is query-dependent and it moves: take the top columns you found on one prompt and apply them to a different prompt and the recall craters. So I cannot precompute a fixed mask once and reuse it — the choice has to come from the current query. And here is the trap that makes this hard rather than trivial: to know which `(q, k)` pairs are big, the obvious thing is to compute `q . k` for all of them and keep the largest. But that *is* the dense score I was trying to avoid. `O(t^2)` again. So I need to find the important keys without first paying for the full score matrix. That is the whole problem in one sentence.

First instinct, the cheap content-blind patterns, because they cost almost nothing. A sliding window: each query attends to the last `w` keys. Contiguous, trivial, fast. But it is blind — anything older than `w` is simply gone, so a needle sitting far back in the context is invisible. Patch it the way the streaming-attention people did: keep a few initial "sink" tokens too. There is a real reason those help — softmax has to put its weight somewhere even when nothing in the past is relevant, and the first tokens end up absorbing that leftover mass as a kind of no-op sink; keeping their KV around stabilizes the distribution and lets you stream past the training length. So sinks plus a recent window recovers most of windowed attention. But it is still a *static* mask: sinks at the front, a window at the back, and a hole in the middle. If the fact I need is in the middle of a 100k document, this pattern cannot route to it, because the pattern does not depend on what the query is asking. I could throw in random blocks per query the way the expander-graph people do to guarantee connectivity, but random is still not *chosen for this query* — on a specific retrieval the relevant block gets hit only by luck unless I blow the budget. That confirms what the dynamic measurement was already telling me: a fixed pattern, however clever, cannot be the answer. The selection has to be content-adaptive — driven by the current `q`.

So content-adaptive it is, but back to the cost trap: I cannot afford the full `q . k` matrix. I need to score importance at a coarser unit than the individual key. Is that legitimate, or am I throwing away the resolution I need? Look again at the structure of where attention actually lands. It is not only sparse and dynamic — it is *spatially clustered*. When people measure the distance from a nonzero attention entry to its nearest nonzero neighbor, that distance concentrates around five; the important entries do not scatter uniformly, they come in contiguous runs, two-dimensional blocks. Neighboring keys tend to share an importance level. That is the structural gift. If importance is approximately a property of a contiguous *block* of keys rather than of each key in isolation, then I do not need a per-key score — I can score a whole block at once and lose very little, because the keys inside a block rise and fall together. And there is a second, independent reason to go blockwise that has nothing to do with accuracy: the hardware. GPUs deliver far more throughput on contiguous block reads than on random index reads, and Tensor Cores want dense block-shaped operands — that is exactly why the fast attention kernels tile K/V into contiguous blocks loaded once into SRAM. If I selected scattered individual tokens, I would force non-contiguous gathers from the KV cache and fall back to terrible utilization, eating the speedup I was chasing. Both the data (clustering) and the machine (contiguous reads) point the same way: select at block granularity. So: chop the keys into contiguous blocks of size `B`, and decide importance per block.

Now, how do I score a block cheaply? I want one number per (query, key-block) pair that approximates how much that query wants that block, without touching the individual keys inside it. Summarize each block by a single representative vector and score against that. What representative? Nothing with learned parameters — the model is frozen and I am at inference, so a learned projection is off the table, and that leaves two parameter-free options people have actually shipped. One keeps, per block, the channel-wise min and max of its keys, and bounds a query's best possible logit against the whole block by `sum_c max(q_c*min_c, q_c*max_c)` — this is `>= q . k` for every key in the block regardless of the sign of each query channel, so it never silently under-ranks a block. But it costs two summary vectors instead of one, and, because the bound's value turns on the sign pattern of the query itself, two heads that share the same keys can rank the very same block differently — used per head, as originally proposed, that is exactly the independent per-head selection that inflates GQA memory. The other option keeps a handful of individual "representative" tokens per block and dots the query against those — cheap, but it is a bet that a few discrete tokens stand in for the whole block, with no structural guarantee they do. I want something with a firmer foundation than either.

The block's keys are spatially coherent — that was the whole premise, they cluster — so they point in similar directions, which suggests their *average* is a reasonable summary of the block's direction. Let me make sure the average is the *right* summary and not just a convenient one. The thing I actually want the proxy to track is the aggregate of `q . k` over the keys in the block, `sum_{k in block} q . k`. Dot product is linear in `k`, so `sum_{k in block} q . k = q . (sum_{k in block} k) = Bk * (q . mean(k_block))`. So the mean key is not an approximation of the block's aggregate logit at all — up to the constant factor `Bk` it *is* the aggregate, exactly. That is a stronger statement than "the keys point the same way": even if a block were internally spread out, `q . mean(k_block)` still equals the average over the block of `q . k`. The clustering buys me something different — it makes the *max* logit in the block close to that average, so a block that scores high on the mean is a block that actually contains a high-scoring key rather than a lukewarm crowd. Both pieces matter, but the linearity is what makes the mean principled rather than a heuristic, and it pays off again later: because a dot product is linear where a max is not, if two heads sharing a KV ever need to agree on one selection, I can force that agreement just by pooling their queries too before scoring — the max-bound has no equally cheap move, since the max of two heads' bounds is not the bound of their pooled query. Averaging keys as the block representative is a known competitive choice against fancier learned or top-token representatives. Good — mean-pooled keys are my block representatives.

What about the query side? I am scoring per block, so I want one score per (query-block, key-block) pair, not per (query, key-block) — otherwise I still pay `O(t)` per query times `O(t/B)` blocks, which is `O(t^2 / B)`, better but still quadratic in `t`. If I also pool the queries into blocks of size `B` and use the mean query of a block, then the score matrix is `(t/B) x (t/B)` — `O((t/B)^2)`, a factor `B^2` cheaper than dense. Is pooling the queries defensible? Same clustering argument, now on the query axis: adjacent query positions are asking for similar things and want similar blocks, so giving a whole query block one shared selection is consistent with the spatial structure, and it is exactly the regime where a block-shaped kernel is happy — load all the queries at a position together, let them share the same selected KV blocks. So mean-pool the queries too. Now the importance score is

  `s[i, j] = (mean of q over query-block i) . (mean of k over key-block j) * scale`,

a small `(n_blocks x n_blocks)` matrix, `scale = 1/sqrt(D)` as usual. That is the cheap proxy for "how much does query-block `i` want key-block `j`," and the selection step never materializes the dense `(t x t)` logits.

With block scores in hand, the selection rule writes itself from the dynamic-sparsity fact: for each query block keep a fixed number of high-scoring key blocks, and attend exactly to those. Ranking by score because that is the cheapest content-adaptive rule there is, and content-adaptive because reusing a static set fails. So far so good, but two things have to be handled before this is correct, and one of them I only see when I think about a query block near the start.

Causality first. This is a causal LLM: query-block `i` may only see key-block `j` with `j <= i`. So before I rank, I have to forbid the future blocks — set `s[i, j] = -inf` for `j > i` so they can never be picked — and at the end I have to AND the kept set with the causal block mask again, and then, inside the diagonal block where query and key blocks coincide, enforce the strict token-level triangle `key_pos <= query_pos`, because a coarse "keep block `i` for query-block `i`" would otherwise let an early query in the block peek at a later key in the same block. Two levels of causal masking: blockwise for the selection, token-wise for the final attention.

Second, the thing I nearly missed: the diagonal block. Local context — the keys right next to the query — is almost always the most relevant, and it had better always be attended to. But if I just take top-`n` by the proxy score, there is no guarantee the current block wins a slot; the mean-pooled proxy is an approximation, and for an early query block the diagonal might score lower than some content-similar block far away. And there is a degenerate corner: a query block could select none of its own neighborhood, which is exactly the local information the sliding-window baselines were built to protect. So I force-include the diagonal (current) block unconditionally. The clean way to do it without wasting a top-`n` slot on a block I am going to keep anyway: when I run the top-`k` ranking, mask the diagonal out (set its score to `-inf` so it never gets picked by the ranking), pick the top-`k` *off-diagonal* blocks, and then append the diagonal index by hand. So if I want `K` blocks per query in total including the diagonal, I take `kk = K - 1` off-diagonal blocks from the rank and add the one diagonal block. This is the same role the dedicated local/sliding-window branch plays in the trainable version — here it collapses into "always keep your own block."

Now the budget. The harness will abort me if the mean density across layers exceeds 0.25 (plus a hair of slack), so I cannot just pick a hand-set `n`; I have to derive the per-query-block keep count `K` from the budget, and I have to do it accounting for causality, because under a causal mask different query blocks have different numbers of admissible blocks. Let me actually count. Index query blocks `i = 0 .. n-1`. Query block `i` has `i + 1` causal candidate key blocks (`j = 0 .. i`). If I keep `K` blocks per query block, block `i` actually keeps `min(K, i + 1)` — an early block simply has fewer than `K` blocks available. So the total number of kept block-pairs is `sum_{i=0}^{n-1} min(K, i + 1)`. For the blocks with `i + 1 >= K` (that is `i >= K - 1`) it contributes `K` each; for the early blocks `i < K - 1` it contributes `1, 2, ..., K-1`. So

  `sum_i min(K, i+1) = (1 + 2 + ... + (K-1)) + (n - (K-1)) * K`
                      `= K(K-1)/2 + (n - K + 1)K`
                      `= K * [ (K-1)/2 + n - K + 1 ]`
                      `= K * [ n + 1 - (K-1)/2 ]`
                      `= K * (2n - K + 1) / 2`.

Let me sanity check that closed form. `K = 1`: every query block keeps just itself, `1*(2n-1+1)/2 = n` — yes, `n` blocks. `K = n`: keep everything causal, `n*(2n - n + 1)/2 = n(n+1)/2` — yes, the full causal count. Good, `K(2n - K + 1)/2` is exact for `K <= n`. The denominator — the total number of admissible causal block-pairs — is that same `K = n` value, `n(n+1)/2`. So the *block-level* density is

  `density = K(2n - K + 1) / (n(n + 1))`.

I want this equal to the budget `rho`. Rearrange: `K(2n + 1 - K) = rho * n(n + 1)`, i.e.

  `K^2 - (2n + 1)K + rho * n(n + 1) = 0`,

a quadratic in `K` with roots `K = [ (2n + 1) +/- sqrt((2n+1)^2 - 4 rho n(n+1)) ] / 2`. Two roots — which one? The kept-pair function `K(2n+1-K)` is concave: it rises over the feasible range `0 <= K <= n`, reaching the full causal count at `K = n`, while the other root of the quadratic lies beyond the feasible range. I want the smaller feasible root, then I floor it so the integer keep count stays at or under the budget, so I take the minus root,

  `K = ( (2n + 1) - sqrt((2n + 1)^2 - 4 rho n(n + 1)) ) / 2`.

The plus root would be `> n`, i.e. "keep almost everything," the wrong end. Let me plug in the real numbers to be sure: `B = 64`, `N = 8k`, so `n = N/B = 128` query blocks, `rho = 0.25`. Discriminant `(257)^2 - 4 * 0.25 * 128 * 129 = 66049 - 16512 = 49537`, `sqrt ~ 222.6`, so `K = (257 - 222.6)/2 ~ 17.2`. Floor to 17 blocks per query block including the diagonal. Check the density that actually produces: `17 * (256 - 17 + 1)/(128*129) = 17 * 240 / 16512 = 4080/16512 ~ 0.247`, comfortably under 0.25. So `K = 17`, `kk = 16` off-diagonal top-`k` plus the diagonal. The non-causal fallback, if I ever needed it, is just `K ~ round(rho * n)` since every query block then sees all `n` blocks — but here `is_causal` is always true. I clamp `K` to at least 1 so a query block always keeps at least its own diagonal, and clamp `kk` to `[0, n-1]`.

There is a subtlety I should be honest about: this `K` controls the *block-pair* density, and my reported `last_density` will be computed on the actual *token-level* keep mask after I expand blocks to tokens and apply the strict token causal triangle. Those are not identical at the boundary, because the diagonal block is only half-filled under the token triangle, and the last block may be padding. So I will not trust the algebra for reporting — I will compute the true fraction from the realized token mask. The algebra is for *choosing* `K`; the reported density is *measured* from the mask. At `n = 128` the block-level density I computed (0.247) sits comfortably under budget, and the token-level density, which only trims the diagonal block to its lower triangle, comes in at or below that, so I stay within budget there. That margin is not free at every length, though: forcing the diagonal block is mandatory, and the diagonal alone already costs block-level density `2/(n+1)` (`n` forced blocks out of `n(n+1)/2` causal pairs), which only reaches at or under a `0.25` budget once `n >= 7` — for a handful of blocks the forced local context can legitimately outweigh the budget, and the harness's contexts (8k and up, so `n` in the hundreds) are nowhere near that edge.

Let me also deal with the shape mechanics so the pooling is clean. `N` may not be a multiple of `B`. Pad `q` and `k` up to `Npad = ceil(N/B) * B` with zeros on the sequence axis so the reshape `(B, H, n, Bk, D)` is exact and `mean(dim=3)` gives the block means. `n = Npad / B`. The pad keys contribute to a final padding block, but I trim the token keep-mask back to `[:N, :N]` afterward and the strict causal triangle plus the slice kills any influence of the padded positions, so they cannot leak into a real query's output. (The padded *queries*, rows `>= N`, I just drop in the slice.) I do the block scoring in float32 — `q, k` arrive in fp16/bf16 and I am about to `masked_fill` with `-inf` and softmax, which is far more stable in fp32 — and cast the output back to `q.dtype` at the very end.

Now expand the block decision to a token keep-mask and finish. I have `block_keep[b, h, i, j]` true iff query-block `i` keeps key-block `j` (top-`kk` off-diagonal scattered to true, diagonal scattered to true, then ANDed with the causal block mask `i >= j`). Map each token to its block via integer division `token // B`, and gather: `token_keep[b, h, p, r] = block_keep[b, h, p//B, r//B]`. Trim to `[:N, :N]`. Then AND with the strict token causal mask `query_pos >= key_pos` so the half-diagonal-block and same-block ordering are correct. From this realized mask the density is just `token_keep[0,0].sum() / (N(N+1)/2)` for the causal case (`N*N` otherwise) — measured, per the contract, finite and in `[0,1]`. Finally the attention itself, on the *full* dense logits but masked: compute `q . k^T * scale` in float32, set the non-kept entries to `-inf`, softmax along the keys, replace any NaN rows (a row that somehow kept nothing would softmax `-inf`s to NaN) with zeros, multiply by `v`, cast back. I keep the full `q.k^T` here rather than gathering only the selected blocks because this is the pure-PyTorch reference form — no Triton — and the point of the baseline is faithful *behavior* under the budget, not the kernel-level memory savings; a Triton kernel would gather the selected contiguous blocks into SRAM exactly as the block decision prescribes, which is where the realized speedup comes from, but the masked-softmax form computes the identical output.

I should also check I have not quietly reintroduced the GQA failure mode I ruled out for the max-bound alternative earlier. Left alone, a per-head mean score has the same symptom as the max-bound: each head has its own query, so nothing stops two heads in a group from ranking the same block differently. For *my* linear proxy the fix is free: `sum_h (q_h_blk . k_blk) = (sum_h q_h_blk) . k_blk`, so forcing agreement is just pooling the per-head queries before the one dot product, no per-head score ever needs computing separately. The max-bound has no equivalent shortcut — the max of several heads' bounds is not the bound of their pooled query, so making it group-consistent means computing every head's bound and reconciling them after the fact, exactly the cost that leaves Quest, as originally proposed, selecting per head and loading the union of the group's picks. (The real trainable design behind this baseline sums each head's actual importance score across the group instead, which is the same group-consistency fix but paid per head, because its scores come from a softmax and are not linear in the query the way my mean-based proxy is.) Here the inference harness has already replicated GQA so my module sees 12 heads on both Q and K/V, i.e. it is handed an effectively multi-head view, and in this setting I score and select per replicated head. If I were writing the kernel-level version against the real GQA cache instead, I would use the free move and pool the per-head queries within a group before scoring, to force one shared selection.

One more risk worth naming: is block-pooling the query silently dropping resolution the task needs? A query block whose tokens want genuinely different blocks would have its averaged query pick a compromise that serves none of them. But the spatial-clustering premise says adjacent queries want similar blocks, and the diagonal-force-include guarantees every query block always has its own local neighborhood regardless of the proxy; the off-diagonal top-`k` then adds the shared long-range blocks the block collectively wants. The remaining failure case — a query block that is internally heterogeneous *and* needs a far block only some of its tokens want — is exactly the case the budget could not afford to resolve at token granularity anyway, so block selection is the right trade under the cap, and it avoids the specific problems the two alternatives already ruled out: the max-bound selector's per-head inconsistency under GQA, and the representative-token approach's dependence on a few discrete tokens actually standing in for the block. Mean-pooled-key block scoring with a forced diagonal is the cleanest fit for a frozen model, a parallel forward, a fixed budget, and contiguous reads.

So the recipe, end to end: pad to a block multiple; mean-pool `q` and `k` into per-block vectors; score block pairs `s = q_blk . k_blk^T * scale` in fp32; causal-mask the block scores; mask the diagonal out of the ranking; solve the budget quadratic for `K`, take `kk = K - 1` off-diagonal top-`k` blocks and append the diagonal; AND with the causal block mask; expand to a token keep-mask, trim to `N`, AND with the strict token causal triangle; report the measured density; and do a masked-softmax attention over the full logits. Here is the module, filling the one empty slot in the harness:

```python
import math
import torch
import torch.nn as nn


class SparseAttention(nn.Module):
    """Content-adaptive block-sparse top-K attention for a frozen causal LLM.

    For each query block, score all causal key blocks by mean-pooled-query .
    mean-pooled-key, keep K total blocks by taking K-1 top off-diagonal
    blocks plus the always-kept diagonal block, and run masked-softmax attention
    over the kept (q, k) pairs.
    """

    BLOCK = 64

    def __init__(self, head_dim, num_heads, block_size=64, density_budget=0.25):
        super().__init__()
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.block_size = block_size
        self.density_budget = density_budget
        self.last_density = None

    def forward(self, q, k, v, is_causal=False, scale=None):
        B, H, N, D = q.shape
        Bk = self.BLOCK
        scale = scale if scale is not None else 1.0 / math.sqrt(D)

        # Pad sequence up to a multiple of BLOCK so block-pooling is exact.
        Npad = ((N + Bk - 1) // Bk) * Bk
        if Npad != N:
            pad = Npad - N
            qp = torch.nn.functional.pad(q, (0, 0, 0, pad))
            kp = torch.nn.functional.pad(k, (0, 0, 0, pad))
        else:
            qp, kp = q, k
        n_blocks = Npad // Bk

        # Block representatives: mean-pooled q / k per block -> (B, H, n_blocks, D).
        q_blocks = qp.view(B, H, n_blocks, Bk, D).mean(dim=3)
        k_blocks = kp.view(B, H, n_blocks, Bk, D).mean(dim=3)

        # Cheap block-pair importance proxy (fp32 for stable masking/softmax).
        scores = torch.einsum('bhmd,bhnd->bhmn',
                              q_blocks.float(), k_blocks.float()) * scale

        idx = torch.arange(n_blocks, device=q.device)
        if is_causal:                                   # forbid future blocks j > i
            causal_blk = idx[:, None] >= idx[None, :]
            scores = scores.masked_fill(~causal_blk, float('-inf'))

        # Remove the diagonal from the ranking (it is force-included below).
        diag_mask = torch.zeros(n_blocks, n_blocks, dtype=torch.bool, device=q.device)
        diag_mask.fill_diagonal_(True)
        scores_nodiag = scores.masked_fill(diag_mask, float('-inf'))

        # Derive per-query-block keep count K from the density budget. With per-row
        # keep K under causal AND, kept block-pairs = K(2n-K+1)/2, denom = n(n+1)/2,
        # so density = K(2n-K+1)/(n(n+1)); solve the quadratic, take the smaller root.
        n_b = n_blocks
        if is_causal:
            disc = max(0.0, (2 * n_b + 1) ** 2
                      - 4 * self.density_budget * n_b * (n_b + 1))
            K_top = max(1, int(((2 * n_b + 1) - math.sqrt(disc)) / 2))
        else:
            K_top = max(1, int(round(self.density_budget * n_b)))
        kk = max(0, min(K_top - 1, n_b - 1))            # off-diagonal slots (diagonal is +1)

        if kk > 0:
            topk_idx = scores_nodiag.topk(kk, dim=-1).indices       # (B,H,n_b,kk)
        else:
            topk_idx = torch.empty(B, H, n_blocks, 0,
                                   dtype=torch.long, device=q.device)
        diag_idx = idx.view(1, 1, n_blocks, 1).expand(B, H, n_blocks, 1)
        sel = torch.cat([topk_idx, diag_idx], dim=-1)              # top-k + diagonal

        block_keep = torch.zeros(B, H, n_blocks, n_blocks,
                                 dtype=torch.bool, device=q.device)
        block_keep.scatter_(-1, sel, True)
        if is_causal:
            block_keep = block_keep & causal_blk

        # Expand block decision to a token-level keep mask, trim padding.
        tok_blk = torch.arange(Npad, device=q.device) // Bk
        token_keep = block_keep[:, :, tok_blk, :][:, :, :, tok_blk]
        token_keep = token_keep[:, :, :N, :N]
        if is_causal:                                   # strict token causal triangle
            tidx = torch.arange(N, device=q.device)
            token_keep = token_keep & (tidx[:, None] >= tidx[None, :])

        # Report density measured from the realized mask (per the harness contract).
        denom = (N * (N + 1) / 2.0) if is_causal else float(N * N)
        self.last_density = float(token_keep[0, 0].sum().item()) / max(denom, 1.0)

        # Masked-softmax attention over the kept (q, k) pairs (full logits, masked).
        attn = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
        attn = attn.masked_fill(~token_keep, float('-inf'))
        attn = torch.softmax(attn, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)          # rows that kept nothing -> 0
        out = torch.matmul(attn, v.float())
        return out.to(q.dtype)
```
